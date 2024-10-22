import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class GenvisSetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, sim_use_clip):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.sim_use_clip = sim_use_clip
    
    # def loss_texts(self, outputs, targets, indices, num_masks):
    #     """
    #     Contrastive Loss between text embedding and query projection
    #     """
    #     assert "pred_text_embed" in outputs
    #     all_src_texts = outputs["pred_text_embed"] # L, B, cQ, Ct
    #     # src_masks = outputs["pred_masks"]
        
    #     idx = self._get_src_permutation_idx(indices)
    #     L, B, cQ, Ct = all_src_texts.shape
        
    #     src_texts = all_src_texts.reshape(L*B, cQ, Ct)
    #     # src_masks = src_masks.reshape(L*B, cQ, T, H, W)

    #     # src_masks = src_masks[idx] # Nt x T x Hp x Wp
    #     src_texts = src_texts[idx] # Nt  x Ct
    #     # target_masks = []
    #     valid_targets = []
    #     target_texts = []
    #     for t in (targets * L):
    #         valid_targets.append(t["valid_inst"])
    #         # target_masks.append(t["masks"])
    #         target_texts.append(t["texts"])
            
    #     # target_masks = torch.cat(target_masks).to(src_masks) # Nt x T x Ht x Wt
    #     target_texts = torch.cat(target_texts) # Nt x Ct
    #     valid_targets = torch.cat(valid_targets) # Nt
        
    #     target_texts = target_texts.reshape(L, B, Ct)
        
    #     # src_masks = src_masks[valid_targets]
    #     # target_masks = target_masks[valid_targets]
    #     # matched_src_texts = src_texts[valid_targets]
        
    #     cos_dist = torch.einsum('lbqc, lbc -> blq', F.normalize(all_src_texts, dim=-1), F.normalize(target_texts, dim=-1)) # B, L, cQ
    #     labels = torch.nn.functional.one_hot(torch.tensor(indices)[:,0].reshape(L,B), cQ)
    #     loss = torch.abs(cos_dist - labels.float())**2
    #     loss_text = loss.mean(-1).sum()
        
    #     losses = {'loss_genvis_text': loss_text}
        
    #     return losses
    def loss_fusion(self, pred_masks, video_targets, num_masks):
        """

        """
        cN,B,T,Hp,Wp = pred_masks.shape
        
        target_masks = []
        for t in (video_targets):
            target_masks.append(t['merged_masks'])
        target_masks = torch.cat(target_masks) # B*cN*T, Ht, Wt
        BcNT, Ht, Wt = target_masks.shape
        
        pred_masks = pred_masks.permute(1,0,2,3,4).reshape(cN*B*T, 1, Hp, Wp)
        target_masks = target_masks.reshape(BcNT, 1, Ht, Wt).to(dtype=torch.float16)

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                pred_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            pred_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)
        
        
        point_logits = point_logits.view(cN*B, T * self.num_points)
        point_labels = point_labels.view(cN*B, T * self.num_points)
        
        losses = {
            "loss_fusion_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_fusion_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del pred_masks
        del target_masks
        return losses
    
    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs['pred_logits']
        L, B, cQ, _ = src_logits.shape
        src_logits = src_logits.reshape(L*B, cQ, self.num_classes+1)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = []
        valid_targets = []
        for t in (targets * L):
            valid_targets.append(t["valid_inst"])
            target_classes_o.append(t["labels"])

        valid_targets = torch.cat(valid_targets)
        target_classes_o = torch.cat(target_classes_o)
        
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        ) # LB, cQ
        valid_idx = (idx[0][valid_targets], idx[1][valid_targets])
        target_classes[valid_idx] = target_classes_o[valid_targets]

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_genvis_ce': loss_ce}

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        L, B, cQ, T, H, W = src_masks.shape
        src_masks = src_masks.reshape(L*B, cQ, T, H, W)

        src_masks = src_masks[idx] # Nt x T x Hp x Wp
        target_masks = []
        valid_targets = []
        for t in (targets * L):
            valid_targets.append(t["valid_inst"])
            target_masks.append(t["masks"])

        target_masks = torch.cat(target_masks).to(src_masks)
        valid_targets = torch.cat(valid_targets)
        # Nt x T x Ht x Wt
        
        src_masks = src_masks[valid_targets]
        target_masks = target_masks[valid_targets]

        Nt = target_masks.shape[0]

        src_masks = src_masks.flatten(0, 1)[:, None]
        target_masks = target_masks.flatten(0, 1)[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        # Nt*T, randN -> Nt, T*randN
        point_logits = point_logits.view(Nt, T * self.num_points)
        point_labels = point_labels.view(Nt, T * self.num_points)

        losses = {
            "loss_genvis_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_genvis_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_fg_sim(
        self, outputs, clip_targets, frame_targets,
        clip_indices, frame_indices, num_masks, MULTIPLIER=1000
    ):
        total_src_q, total_tgt_ids, total_batch_idx = [], [], []

        # Frame
        src_fq = outputs["pred_fq_embed"]   # L, B, T, fQ, C
        # L = number of frame_decoder layers
        L, B, T, fQ, C = src_fq.shape
        src_fq = src_fq.flatten(0, 2)       # LBT, fQ, C

        frame_indices = sum(frame_indices, [])
        frame_src_idx = self._get_src_permutation_idx(frame_indices)    # len = LBT
        src_fq = src_fq[frame_src_idx]      # Nf, C
        target_frame_ids = torch.cat(
            [t["ids"][J] for t, (_, J) in zip(frame_targets * L, frame_indices)]
        )
        frame_batch_idx = torch.div(frame_src_idx[0].to(device=src_fq.device), T, rounding_mode="floor")
        is_frame_valid = target_frame_ids != -1
        target_frame_ids += frame_batch_idx * MULTIPLIER

        total_src_q.append(src_fq[is_frame_valid])
        total_tgt_ids.append(target_frame_ids[is_frame_valid])
        total_batch_idx.append(frame_batch_idx[is_frame_valid])

        # Clip
        if self.sim_use_clip:
            src_cq = outputs["pred_cq_embed"]   # L, B, cQ, C
            src_cq = src_cq.flatten(0, 1)       # LB , cQ, C

            clip_src_idx = self._get_src_permutation_idx(clip_indices)      # len = LB

            target_clip_ids = []
            valid_targets = []
            for t in (clip_targets * L):
                valid_targets.append(t["valid_inst"])
                target_clip_ids.append(t["ids"])
            
            valid_targets = torch.cat(valid_targets)
            target_clip_ids = torch.cat(target_clip_ids)[valid_targets].amax(dim=1)

            valid_src_idx = (clip_src_idx[0][valid_targets], clip_src_idx[1][valid_targets])

            src_cq = src_cq[valid_src_idx]

            clip_batch_idx = valid_src_idx[0].to(device=src_fq.device)
            is_clip_valid = target_clip_ids != -1
            target_clip_ids += clip_batch_idx * MULTIPLIER

            total_src_q.append(src_cq[is_clip_valid])
            total_tgt_ids.append(target_clip_ids[is_clip_valid])
            total_batch_idx.append(clip_batch_idx[is_clip_valid])

        # Clip + Frame
        total_src_q = torch.cat(total_src_q)            # Nc+Nf, C
        total_tgt_ids = torch.cat(total_tgt_ids)        # Nc+Nf
        total_batch_idx = torch.cat(total_batch_idx)    # Nc+Nf

        sim_pred_logits = torch.matmul(total_src_q, total_src_q.T)          # Nc+Nf, Nc+Nf
        sim_tgt = (total_tgt_ids[:, None] == total_tgt_ids[None]).float()   # Nc+Nf, Nc+Nf

        same_clip = (total_batch_idx[:, None] == total_batch_idx[None]).float()
        loss = F.binary_cross_entropy_with_logits(sim_pred_logits, sim_tgt, reduction='none')

        loss = loss * same_clip
        loss_clip_sim = loss.sum() / (same_clip.sum() + 1e-6)

        return {"loss_genvis_sim": loss_clip_sim}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(
        self, loss, outputs, clip_targets, frame_targets, clip_indices, frame_indices, num_masks
    ):
        loss_map = {
            'genvis_masks': self.loss_masks,
            'genvis_fusion': self.loss_fusion,
            'genvis_labels': self.loss_labels,
            'fg_sim': self.loss_fg_sim,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        if loss == 'fg_sim':
            return loss_map[loss](
                outputs, clip_targets, frame_targets, clip_indices, frame_indices, num_masks
            )
        if loss == 'genvis_fusion':
            return loss_map[loss](outputs['pred_masks_fused'], clip_targets, num_masks)

        return loss_map[loss](outputs, clip_targets, clip_indices, num_masks)

    def forward(self, outputs, clip_targets, frame_targets, frame_indices=None, prev_clip_indices=None, prev_aux_clip_indices=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        new_clip_indices = self.matcher(outputs_without_aux, clip_targets, prev_clip_indices)
        L = len(new_clip_indices)

        if prev_clip_indices is None: # first clip
            clip_indices = new_clip_indices
        else:
            # merge indices
            clip_indices = []
            for t, p_i, n_i in zip(clip_targets*L, prev_clip_indices, new_clip_indices):
                new_inst = t["new_inst"]
                merged_src_idx = p_i[0]
                merged_tgt_idx = p_i[1]

                merged_src_idx[new_inst] = n_i[0][new_inst]
                merged_tgt_idx[new_inst] = n_i[1][new_inst]

                clip_indices.append((merged_src_idx, merged_tgt_idx))
 
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        #num_masks = sum(len(t["labels"]) for t in clip_targets) * len(outputs_without_aux["pred_masks"])
        num_masks = sum(t["valid_inst"].sum() for t in clip_targets) * len(outputs_without_aux["pred_masks"])
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss, outputs, clip_targets, frame_targets, clip_indices, frame_indices, num_masks
                )
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            aux_clip_indices_list = []
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                if prev_aux_clip_indices is None:
                    new_aux_clip_indices = self.matcher(aux_outputs, clip_targets)
                else:
                    new_aux_clip_indices = self.matcher(aux_outputs, clip_targets, prev_aux_clip_indices[i])
                if prev_aux_clip_indices is None:
                    aux_clip_indices = new_aux_clip_indices
                else:
                    # merge indices
                    aux_clip_indices = []
                    for t, p_i, n_i in zip(clip_targets*L, prev_aux_clip_indices[i], new_aux_clip_indices):
                        new_inst = t["new_inst"]
                        merged_src_idx = p_i[0]
                        merged_tgt_idx = p_i[1]

                        merged_src_idx[new_inst] = n_i[0][new_inst]
                        merged_tgt_idx[new_inst] = n_i[1][new_inst]

                        aux_clip_indices.append((merged_src_idx, merged_tgt_idx))

                for loss in self.losses:
                    if loss == "fg_sim" or loss == "genvis_fusion":
                        continue
                    l_dict = self.get_loss(
                        loss, aux_outputs, clip_targets, frame_targets, aux_clip_indices, frame_indices, num_masks
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

                aux_clip_indices_list.append(aux_clip_indices)

        else: # TODO: make clear
            aux_clip_indices_list = None

        return losses, clip_indices, aux_clip_indices_list

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
