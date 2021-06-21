import os
import os.path as osp
import json
from collections import OrderedDict
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix

__all__ = [
    'compute_result_multilabel',
    'compute_result_multilabel_test',
    'compute_result',
    'frame_level_map_n_cap_tvseries',
    'frame_level_map_n_cap_thumos',
    'compute_result_multilabel_tvseries',
]


def compute_result_multilabel_tvseries(class_index, score_metrics, target_metrics, save_dir, result_file,
                              ignore_class=[0], save=True, verbose=False, smooth=False, switch=False):
    result = OrderedDict()
    score_metrics = np.array(score_metrics)
    pred_metrics = np.argmax(score_metrics, axis=1)
    target_metrics = np.array(target_metrics)

    ###################################################################################################################
    # We follow (Shou et al., 2017) and adopt their per-frame evaluation method of THUMOS'14 datset.
    # Source: https://bitbucket.org/columbiadvmm/cdc/src/master/THUMOS14/eval/PreFrameLabeling/compute_framelevel_mAP.m
    ###################################################################################################################

    # Simple temporal smoothing via NMS of 5-frames window
    if smooth:
        prob = np.copy(score_metrics)
        prob1 = prob.reshape(1, prob.shape[0], prob.shape[1])
        prob2 = np.append(prob[0, :].reshape(1, -1), prob[0:-1, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob3 = np.append(prob[1:, :], prob[-1, :].reshape(1, -1), axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob4 = np.append(prob[0:2, :], prob[0:-2, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob5 = np.append(prob[2:, :], prob[-2:, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        probsmooth = np.squeeze(np.max(np.concatenate((prob1, prob2, prob3, prob4, prob5), axis=0), axis=0))
        score_metrics = np.copy(probsmooth)

    # Assign cliff diving (5) as diving (8)
    if switch:
        switch_index = np.where(score_metrics[:, 5] > score_metrics[:, 8])[0]
        score_metrics[switch_index, 8] = score_metrics[switch_index, 5]

    # Remove ambiguous (21)
    # valid_index = np.where(target_metrics[:, 21]!=1)[0]

    # Compute AP
    result['AP'] = OrderedDict()
    for cls in range(len(class_index)):
        if cls not in ignore_class:
            result['AP'][class_index[cls]] = average_precision_score(
                (target_metrics[:, cls]==1).astype(np.int),
                score_metrics[:, cls])
            if verbose:
                print('{} AP: {:.5f}'.format(class_index[cls], result['AP'][class_index[cls]]))

    # Compute mAP
    result['mAP'] = np.mean(list(result['AP'].values()))
    if verbose:
        print('mAP: {:.5f}'.format(result['mAP']))

    # Save
    if save:
        if not osp.isdir(save_dir):
            os.makedirs(save_dir)
        with open(osp.join(save_dir, result_file), 'w') as f:
            json.dump(result, f)
        if verbose:
            print('Saved the result to {}'.format(osp.join(save_dir, result_file)))

    return result['mAP']


def frame_level_map_n_cap_thumos(results):
    all_probs = results['probs']
    all_labels = results['labels']

    n_classes = all_labels.shape[0]
    all_cls_ap, all_cls_acp = list(), list()
    for i in range(1, n_classes):
        this_cls_prob = all_probs[i, :]
        this_cls_gt = all_labels[i, :]
        w = np.sum(this_cls_gt == 0) / np.sum(this_cls_gt == 1)

        indices = np.argsort(-this_cls_prob)
        tp, psum, cpsum = 0, 0., 0.
        for k, idx in enumerate(indices):
            if this_cls_gt[idx] == 1:
                tp += 1
                wtp = w * tp
                fp = (k + 1) - tp
                psum += tp / (tp + fp)
                cpsum += wtp / (wtp + fp)
        this_cls_ap = psum/np.sum(this_cls_gt)
        this_cls_acp = cpsum / np.sum(this_cls_gt)

        all_cls_ap.append(this_cls_ap)
        all_cls_acp.append(this_cls_acp)

    map = sum(all_cls_ap) / len(all_cls_ap)
    cap = sum(all_cls_acp) / len(all_cls_acp)
    return map, all_cls_ap, cap, all_cls_acp


def frame_level_map_n_cap_tvseries(results):
    all_probs = results['probs']
    all_labels = results['labels']
    # for i in range(1, all_probs.shape[1] - 1):
    #     all_probs[:, i] = np.mean(all_probs[:, i - 1:i + 1], axis=1)
    n_classes = all_labels.shape[0]
    all_cls_ap, all_cls_acp = list(), list()
    for i in range(1, n_classes):
        this_cls_prob = all_probs[i, :]
        this_cls_gt = all_labels[i, :]
        w = np.sum(this_cls_gt == 0) / np.sum(this_cls_gt == 1)

        indices = np.argsort(-this_cls_prob)
        tp, psum, cpsum = 0, 0., 0.
        for k, idx in enumerate(indices):
            if this_cls_gt[idx] == 1:
                tp += 1
                wtp = w * tp
                fp = (k + 1) - tp
                psum += tp / (tp + fp)
                cpsum += wtp / (wtp + fp)
        this_cls_ap = psum/np.sum(this_cls_gt)
        this_cls_acp = cpsum / np.sum(this_cls_gt)

        all_cls_ap.append(this_cls_ap)
        all_cls_acp.append(this_cls_acp)

    map = sum(all_cls_ap) / len(all_cls_ap)
    cap = sum(all_cls_acp) / len(all_cls_acp)
    return map, all_cls_ap, cap, all_cls_acp


def compute_result_multilabel_test(class_index, score_metrics, target_metrics, save_dir, result_file,
                              ignore_class=[0], save=True, verbose=False, smooth=False, switch=False, logger=None, step=None):
    
    result = OrderedDict()
    score_metrics = np.array(score_metrics)
    pred_metrics = np.argmax(score_metrics, axis=1)
    target_metrics = np.array(target_metrics)

    ###################################################################################################################
    # We follow (Shou et al., 2017) and adopt their per-frame evaluation method of THUMOS'14 datset.
    # Source: https://bitbucket.org/columbiadvmm/cdc/src/master/THUMOS14/eval/PreFrameLabeling/compute_framelevel_mAP.m
    ###################################################################################################################

    # Simple temporal smoothing via NMS of 5-frames window
    if smooth:
        prob = np.copy(score_metrics)
        prob1 = prob.reshape(1, prob.shape[0], prob.shape[1])
        prob2 = np.append(prob[0, :].reshape(1, -1), prob[0:-1, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob3 = np.append(prob[1:, :], prob[-1, :].reshape(1, -1), axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob4 = np.append(prob[0:2, :], prob[0:-2, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob5 = np.append(prob[2:, :], prob[-2:, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        probsmooth = np.squeeze(np.max(np.concatenate((prob1, prob2, prob3, prob4, prob5), axis=0), axis=0))
        score_metrics = np.copy(probsmooth)

    # Assign cliff diving (5) as diving (8)
    if switch:
        switch_index = np.where(score_metrics[:, 5] > score_metrics[:, 8])[0]
        score_metrics[switch_index, 8] = score_metrics[switch_index, 5]

    # Remove ambiguous (21)
    valid_index = np.where(target_metrics[:, 21]!=1)[0]

    # Compute AP
    result['AP'] = OrderedDict()
    for cls in range(len(class_index)):
        if cls not in ignore_class:
            result['AP'][class_index[cls]] = average_precision_score(
                (target_metrics[valid_index, cls]==1).astype(np.int),
                score_metrics[valid_index, cls])
            if verbose:
                print('{} AP: {:.5f}'.format(class_index[cls], result['AP'][class_index[cls]]))
                logger.output_test_results('{} AP: {:.5f}'.format(class_index[cls], result['AP'][class_index[cls]]))

    # Compute mAP
    result['mAP'] = np.mean(list(result['AP'].values()))
    if verbose:
        print('mAP: {:.5f}'.format(result['mAP']))
        if step is not None:
            logger.output_test_results('step {} results : !'.format(step))
        logger.output_test_results('mAP: {:.5f} \n'.format(result['mAP']))

    # Save
    if save:
        if not osp.isdir(save_dir):
            os.makedirs(save_dir)
        with open(osp.join(save_dir, result_file), 'w') as f:
            json.dump(result, f)
        if verbose:
            print('Saved the result to {}'.format(osp.join(save_dir, result_file)))
            logger.output_test_results('Saved the result to {}'.format(osp.join(save_dir, result_file)))

    return result['mAP']


def compute_result_multilabel(class_index, score_metrics, target_metrics, save_dir, result_file,
                              ignore_class=[0], save=True, verbose=False, smooth=False, switch=False):
    result = OrderedDict()
    score_metrics = np.array(score_metrics)
    pred_metrics = np.argmax(score_metrics, axis=1)
    target_metrics = np.array(target_metrics)

    ###################################################################################################################
    # We follow (Shou et al., 2017) and adopt their per-frame evaluation method of THUMOS'14 datset.
    # Source: https://bitbucket.org/columbiadvmm/cdc/src/master/THUMOS14/eval/PreFrameLabeling/compute_framelevel_mAP.m
    ###################################################################################################################

    # Simple temporal smoothing via NMS of 5-frames window
    if smooth:
        prob = np.copy(score_metrics)
        prob1 = prob.reshape(1, prob.shape[0], prob.shape[1])
        prob2 = np.append(prob[0, :].reshape(1, -1), prob[0:-1, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob3 = np.append(prob[1:, :], prob[-1, :].reshape(1, -1), axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob4 = np.append(prob[0:2, :], prob[0:-2, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob5 = np.append(prob[2:, :], prob[-2:, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        probsmooth = np.squeeze(np.max(np.concatenate((prob1, prob2, prob3, prob4, prob5), axis=0), axis=0))
        score_metrics = np.copy(probsmooth)

    # Assign cliff diving (5) as diving (8)
    if switch:
        switch_index = np.where(score_metrics[:, 5] > score_metrics[:, 8])[0]
        score_metrics[switch_index, 8] = score_metrics[switch_index, 5]

    # Remove ambiguous (21)
    valid_index = np.where(target_metrics[:, 21]!=1)[0]

    # Compute AP
    result['AP'] = OrderedDict()
    for cls in range(len(class_index)):
        if cls not in ignore_class:
            result['AP'][class_index[cls]] = average_precision_score(
                (target_metrics[valid_index, cls]==1).astype(np.int),
                score_metrics[valid_index, cls])
            if verbose:
                print('{} AP: {:.5f}'.format(class_index[cls], result['AP'][class_index[cls]]))

    # Compute mAP
    result['mAP'] = np.mean(list(result['AP'].values()))
    if verbose:
        print('mAP: {:.5f}'.format(result['mAP']))

    # Save
    if save:
        if not osp.isdir(save_dir):
            os.makedirs(save_dir)
        with open(osp.join(save_dir, result_file), 'w') as f:
            json.dump(result, f)
        if verbose:
            print('Saved the result to {}'.format(osp.join(save_dir, result_file)))

    return result['mAP']


def compute_result(class_index, score_metrics, target_metrics, save_dir, result_file,
                   ignore_class=[0], save=True, verbose=False):
    result = OrderedDict()
    score_metrics = np.array(score_metrics)
    pred_metrics = np.argmax(score_metrics, axis=1)
    target_metrics = np.array(target_metrics)

    # Compute ACC
    correct = np.sum((target_metrics!=0) & (target_metrics==pred_metrics))
    total = np.sum(target_metrics!=0)
    result['ACC'] = correct / total
    if verbose:
        print('ACC: {:.5f}'.format(result['ACC']))

    # Compute confusion matrix
    result['confusion_matrix'] = \
            confusion_matrix(target_metrics, pred_metrics).tolist()

    # Compute AP
    result['AP'] = OrderedDict()
    for cls in range(len(class_index)):
        if cls not in ignore_class:
            result['AP'][class_index[cls]] = average_precision_score(
                (target_metrics==cls).astype(np.int),
                score_metrics[:, cls])
            if verbose:
                print('{} AP: {:.5f}'.format(class_index[cls], result['AP'][class_index[cls]]))

    # Compute mAP
    result['mAP'] = np.mean(list(result['AP'].values()))
    if verbose:
        print('mAP: {:.5f}'.format(result['mAP']))

    # Save
    if save:
        if not osp.isdir(save_dir):
            os.makedirs(save_dir)
        with open(osp.join(save_dir, result_file), 'w') as f:
            json.dump(result, f)
        if verbose:
            print('Saved the result to {}'.format(osp.join(save_dir, result_file)))

    return result['mAP']
