# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import datetime
import json
import random
import time
from typing import Iterable
from ipdb import set_trace
import torch
import utils
import util
import torch.nn.functional as F
import numpy as np


all_class_name = ['BaseballPitch',
                  'BasketballDunk',
                  'Billiards',
                  'CleanAndJerk',
                  'CliffDiving',
                  'CricketBowling',
                  'CricketShot',
                  'Diving',
                  'FrisbeeCatch',
                  'GolfSwing',
                  'HammerThrow',
                  'HighJump',
                  'JavelinThrow',
                  'LongJump',
                  'PoleVault',
                  'Shotput',
                  'SoccerPenalty',
                  'TennisSwing',
                  'ThrowDiscus',
                  'VolleyballSpiking']


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500
    num_class = 22
    for camera_inputs, motion_inputs, enc_target, distance_target, class_h_target, dec_target in metric_logger.log_every(data_loader, print_freq, header):

        camera_inputs = camera_inputs.to(device)
        motion_inputs = motion_inputs.to(device)

        enc_target = enc_target.to(device)
        distance_target = distance_target.to(device)
        class_h_target = class_h_target.to(device)
        dec_target = dec_target.to(device)

        enc_score_p0, dec_scores = \
            model(camera_inputs, motion_inputs)

        outputs = {
            'labels_encoder': enc_score_p0,  # [128, 22]
            'labels_decoder': dec_scores.view(-1, num_class),  # [128, 8, 22]

        }
        targets = {
            'labels_encoder': class_h_target.view(-1, num_class),
            'labels_decoder': dec_target.view(-1, num_class),

        }

        loss_dict = criterion(outputs, targets)
        # loss_dict_decoder = criterion(outputs_decoder, targets_decoder)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, logger, args, epoch, nprocs=4):

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    all_probs, all_classes = [], []
    dec_score_metrics = []
    dec_score_metrics_every = {}
    dec_target_metrics_every = {}
    num_query = args.query_num
    for iii in range(num_query):
        dec_score_metrics_every[str(iii)] = []
        dec_target_metrics_every[str(iii)] = []
    #
    dec_target_metrics = []

    num_class = args.numclass
    feat_type = args.feature

    for camera_inputs_val, motion_inputs_val, enc_target_val, distance_target_val, class_h_target_val, dec_target in metric_logger.log_every(data_loader, 500, header):
        camera_inputs = camera_inputs_val.to(device)
        motion_inputs = motion_inputs_val.to(device)

        enc_target = enc_target_val.to(device)
        distance_target = distance_target_val.to(device)
        class_h_target = class_h_target_val.to(device)
        dec_target = dec_target.to(device)

        enc_score_p0, dec_scores = \
            model(camera_inputs, motion_inputs)
        # set_trace()

        outputs = {
            'labels_encoder': enc_score_p0,  # [128, 22]
            'labels_decoder': dec_scores.view(-1, num_class),  # [128, 8, 22]

        }
        targets = {
            'labels_encoder': class_h_target.view(-1, num_class),
            'labels_decoder': dec_target.view(-1, num_class),

        }

        loss_dict = criterion(outputs, targets)
        # loss_dict_decoder = criterion(outputs_decoder, targets_decoder)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}

        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        if args.distributed:
            # torch.distributed.barrier()
            logits_gather_list = [torch.zeros_like(enc_score_p0) for _ in range(nprocs)]
            torch.distributed.all_gather(logits_gather_list, enc_score_p0)
            enc_score_p0 = torch.cat(logits_gather_list, dim=0)

            targets_gather_list = [torch.zeros_like(class_h_target) for _ in range(nprocs)]
            torch.distributed.all_gather(targets_gather_list, class_h_target)
            class_h_target = torch.cat(targets_gather_list, dim=0)
            # prob_val = enc_score_p0.detach().cpu().numpy()  # enc_score_p0[:, :21].cpu().numpy()
            all_probs.extend(enc_score_p0[:, :21].detach().cpu().numpy())  # (89, 21)  # all_probs += list(prob_val)
            # t0_class_batch = class_h_target.detach().cpu().numpy()  # class_h_target[:, :21].cpu().numpy()
            all_classes.extend(class_h_target[:, :21].detach().cpu().numpy())
            # set_trace()
        else:
            # prob_val = enc_score_p0[:, :21].cpu().numpy()
            prob_val = F.softmax(enc_score_p0[:, :21], dim=-1).cpu().numpy()
            # prob_val = F.softmax(enc_score_p0, dim=-1)[:, :21].cpu().numpy()  
            all_probs += list(prob_val)  # (89, 21)
            # dec_score_metrics = dec_scores[:, :21].cpu().numpy()
            dec_score_metrics += list(F.softmax(dec_scores.view(-1, num_class)[:, :21], dim=-1).cpu().numpy())
            # dec_score_metrics += list(F.softmax(dec_scores.view(-1, num_class), dim=-1)[:, :21].cpu().numpy())
            t0_class_batch = class_h_target[:, :21].cpu().numpy()
            all_classes += list(t0_class_batch)
            dec_target_metrics += list(dec_target.view(-1, num_class)[:, :21].cpu().numpy())
            for iii in range(num_query):
                dec_score_metrics_every[str(iii)] += list(F.softmax(dec_scores[:,iii,:].view(-1, num_class)[:, :21], dim=-1).cpu().numpy())
                dec_target_metrics_every[str(iii)] += list(dec_target[:,iii,:].view(-1, num_class)[:, :21].cpu().numpy())
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if args.distributed:
        # results
        if utils.is_main_process():
            all_probs = np.asarray(all_probs).T
            logger.output_print(str(all_probs.shape))  # (21, 180489)

            all_classes = np.asarray(all_classes).T
            logger.output_print(str(all_classes.shape))  # (21, 180489)
            results = {'probs': all_probs, 'labels': all_classes}

            map, aps, _, _ = utils.frame_level_map_n_cap(results)
            logger.output_print('[Epoch-{}] [IDU-{}] mAP: {:.4f}\n'.format(epoch, feat_type, map))

            for i, ap in enumerate(aps):
                cls_name = all_class_name[i]
                logger.output_print('{}: {:.4f}'.format(cls_name, ap))
    else:
        # results
        all_probs = np.asarray(all_probs).T
        logger.output_print(str(all_probs.shape))  # (21, 180489)

        all_classes = np.asarray(all_classes).T
        logger.output_print(str(all_classes.shape))  # (21, 180489)
        results = {'probs': all_probs, 'labels': all_classes}

        map, aps, _, _ = utils.frame_level_map_n_cap(results)
        logger.output_print('[Epoch-{}] [IDU-{}] mAP: {:.4f}\n'.format(epoch, feat_type, map))

        results_dec = {}
        results_dec['probs'] = np.asarray(dec_score_metrics).T
        results_dec['labels'] = np.asarray(dec_target_metrics).T
        dec_map_2, dec_aps_2, _, _ = util.frame_level_map_n_cap_thumos(results_dec)
        logger.output_print('dec_mAP all together: | {} |.'.format(dec_map_2))
        all_decoder = 0.
        for iii in range(num_query):
            results_dec = {}
            results_dec['probs'] = np.asarray(dec_score_metrics_every[str(iii)]).T
            results_dec['labels'] = np.asarray(dec_target_metrics_every[str(iii)]).T
            dec_map_2, dec_aps_2, _, _ = util.frame_level_map_n_cap_thumos(results_dec)
            logger.output_print('dec_mAP_pred | {} : {} |.'.format(iii, dec_map_2))
            all_decoder += dec_map_2
        logger.output_print('{}: | {:.4f} |.'.format('all decoder map', all_decoder/num_query))

        for i, ap in enumerate(aps):
            cls_name = all_class_name[i]
            logger.output_print('{}: {:.4f}'.format(cls_name, ap))
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats
