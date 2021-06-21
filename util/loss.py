import torch
import torch.nn.functional as F
from torch import nn
from ipdb import set_trace


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, losses, args):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.classification_x_loss_coef = args.classification_x_loss_coef
        self.classification_h_loss_coef = args.classification_h_loss_coef
        self.similar_loss_coef = args.similar_loss_coef
        self.weight_dict = {
            'labels_encoder': self.classification_h_loss_coef,
            'labels_decoder': args.classification_pred_loss_coef,
            'labels_x0': self.classification_x_loss_coef,
            'labels_xt': self.classification_x_loss_coef,
            'distance': self.similar_loss_coef,
        }
        self.losses = losses
        self.ignore_index = 21
        self.margin = args.margin
        self.size_average = True
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def loss_labels(self, input, targets, name):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # assert 'pred_logits' in outputs
        # src_logits = outputs['pred_logits']
        #
        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # target_classes = torch.full(src_logits.shape[:2], self.num_classes,
        #                             dtype=torch.int64, device=src_logits.device)
        # target_classes[idx] = target_classes_o

        # loss_ce = F.cross_entropy(outputs, targets, ignore_index=21)
        target = targets.float()
        # logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        if self.ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
            output = torch.sum(-target[:, notice_index] * self.logsoftmax(input[:, notice_index]), 1)
            if output.sum() == 0:   # 全为 ignore 类
                loss_ce = torch.tensor(0.).to(input.device).type_as(target)
            else:
                loss_ce = torch.mean(output[target[:, self.ignore_index] != 1])
        else:
            output = torch.sum(-target * self.logsoftmax(input), 1)
            if self.size_average:
                loss_ce = torch.mean(output)
            else:
                loss_ce = torch.sum(output)
        if torch.isnan(loss_ce).sum()>0:
            set_trace()
        losses = {name: loss_ce}

        return losses

    def loss_labels_decoder(self, input, targets, name):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # assert 'pred_logits' in outputs
        # src_logits = outputs['pred_logits']
        #
        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # target_classes = torch.full(src_logits.shape[:2], self.num_classes,
        #                             dtype=torch.int64, device=src_logits.device)
        # target_classes[idx] = target_classes_o

        # loss_ce = F.cross_entropy(outputs, targets, ignore_index=21)
        target = targets.float()
        # logsoftmax = nn.LogSoftmax(dim=1).to(input.device)
        ignore_index = 21  # -1 改为21 更好一点
        if ignore_index >= 0:
            notice_index = [i for i in range(target.shape[-1]) if i != self.ignore_index]
            output = torch.sum(-target[:, notice_index] * self.logsoftmax(input[:, notice_index]), 1)
            if output.sum() == 0:   # 全为 ignore 类
                loss_ce = torch.tensor(0.).to(input.device).type_as(target)
            else:
                loss_ce = torch.mean(output[target[:, self.ignore_index] != 1])
        else:
            output = torch.sum(-target * self.logsoftmax(input), 1)
            if self.size_average:
                loss_ce = torch.mean(output)
            else:
                loss_ce = torch.sum(output)
        if torch.isnan(loss_ce).sum()>0:
            set_trace()
        losses = {name: loss_ce}

        return losses

    def contrastive_loss(self, output, label, name):
        """
        Contrastive loss function.
        Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        """
        output1, output2 = output
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1.-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        if torch.isnan(loss_contrastive).sum()>0:
            set_trace()
        losses = {name: loss_contrastive.double()}
        return losses

    def get_loss(self, loss, outputs, targets):
        loss_map = {
            'labels_encoder': self.loss_labels,
            'labels_decoder': self.loss_labels_decoder,
            'labels_x0': self.loss_labels,
            'labels_xt': self.loss_labels,
            'distance': self.contrastive_loss,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, name=loss)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_boxes = sum(len(t["labels"]) for t in targets)
        # num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs[loss], targets[loss]))

        return losses