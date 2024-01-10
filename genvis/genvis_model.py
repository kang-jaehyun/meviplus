import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom

from vita.vita_model import Vita
from .modeling.genvis_criterion import GenvisSetCriterion
from .modeling.genvis_matcher import GenvisHungarianMatcher
from .modeling.genvis import GenVIS
from detectron2.layers import Conv2d, get_norm

from transformers import BertModel, RobertaModel
# import clip

from vita.modeling.transformer_decoder.vita import FFNLayer, CrossAttentionLayer, SelfAttentionLayer, MLP, _get_activation_fn

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from typing import Optional
from groundingdino.util.inference import load_model, load_image, predict, annotate
import gc

@META_ARCH_REGISTRY.register()
class Genvis(Vita):

    @configurable
    def __init__(
        self,
        len_clip_window: int,
        genvis_criterion: nn.Module,
        motion2text: nn.Module,
        text2text: nn.Module,
        dino: nn.Module,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.len_clip_window = len_clip_window
        self.genvis_criterion = genvis_criterion
        self.freeze_detector = kwargs["freeze_detector"]
        self.motion2text = motion2text
        self.text2text = text2text
        
        hidden_dim = 256 # TODO : configurable
        self.num_decoder_layers = 3 # TODO : configurable
        self.feature_utilize_level = 2 # TODO : configurable
        self.feature_proj = nn.Conv2d(96, 256, kernel_size=1, bias=False) # TODO : configurable

        lateral_norm = get_norm("GN", hidden_dim)
        output_norm = get_norm("GN", hidden_dim)
        
        self.lateral_conv = Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False, norm=lateral_norm)
        self.output_conv = Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False, norm=output_norm, activation=F.relu)
        
        # self.mask_features = Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.multiscale_feature_proj = nn.ModuleList()
        for i in range(self.feature_utilize_level):
            self.multiscale_feature_proj.append(Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False))

        self.dino = dino
        self.convlstm = ConvLSTMCell(input_dim=96, hidden_dim=256, kernel_size=(3,3), bias=True)
        self.feature_shape = (14,14)
        
        for p in self.dino.parameters():
            p.requires_grad_(False)
            
    @classmethod
    def from_config(cls, cfg):
        rets = Vita.from_config(cfg)

        # genvis
        rets["vita_module"] = GenVIS(cfg=cfg)

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight  = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight  = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        sim_weight   = cfg.MODEL.VITA.SIM_WEIGHT
        grounding_weight = cfg.MODEL.GENVIS.GROUNDING_WEIGHT
        
        genvis_matcher = GenvisHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        genvis_weight_dict = {
            "loss_genvis_ce": class_weight, 
            "loss_genvis_mask": mask_weight, 
            "loss_genvis_dice": dice_weight,

            "loss_grounding": grounding_weight,
        }
        if sim_weight > 0.0:
            genvis_weight_dict["loss_genvis_sim"] = sim_weight

        if cfg.MODEL.VITA.DEEP_SUPERVISION:
            aux_weight_dict = {}             
            for i in range(cfg.MODEL.VITA.DEC_LAYERS - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in genvis_weight_dict.items()})
            genvis_weight_dict.update(aux_weight_dict)
        
        genvis_losses = [
            "genvis_labels", 
            "genvis_masks",
            # "genvis_grounding"
            # "genvis_fusion",
        ]
        
        if sim_weight > 0.0:
            genvis_losses.append("fg_sim")

        num_classes = rets["sem_seg_head"].num_classes
        genvis_criterion = GenvisSetCriterion(
            num_classes, 
            matcher=genvis_matcher, 
            weight_dict=genvis_weight_dict,
            eos_coef=cfg.MODEL.VITA.NO_OBJECT_WEIGHT,
            losses=genvis_losses, 
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            sim_use_clip=cfg.MODEL.VITA.SIM_USE_CLIP,
        )
        motion2text = MLP(
            input_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            hidden_dim = cfg.MODEL.GENVIS.TEXT_HIDDEN_DIM,
            output_dim = cfg.MODEL.GENVIS.TEXT_HIDDEN_DIM,
            num_layers = cfg.MODEL.GENVIS.PROJ_LAYERS,
        )
        text2text = MLP(
            input_dim = cfg.MODEL.GENVIS.BERT_DIM,
            hidden_dim = cfg.MODEL.GENVIS.TEXT_HIDDEN_DIM,
            output_dim = cfg.MODEL.GENVIS.TEXT_HIDDEN_DIM,
            num_layers = cfg.MODEL.GENVIS.PROJ_LAYERS,
        )

        rets.update({
            "len_clip_window": cfg.MODEL.GENVIS.LEN_CLIP_WINDOW,
            "genvis_criterion": genvis_criterion,
            "motion2text": motion2text,
            "text2text": text2text,
            "dino": load_model("/workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/workspace/GroundingDINO/weights/groundingdino_swint_ogc.pth"),  # TODO : hard coded
        })

        return rets

    def train_model(self, batched_inputs):
        num_frames = len(batched_inputs[0]['image'])
        pre_memory = {"k": [], "v": [], "motion": []}
        num_clips = num_frames // self.len_clip_window
        
        assert num_frames % self.len_clip_window == 0, f"num_frames: {num_frames}, len_clip_window: {self.len_clip_window}"
        
        B = len(batched_inputs)
        L = 3 # TODO : configurable
        cQ = 20 # TODO : configurable
        
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
            
        # mask classification target
        frame_targets_all, video_targets_full = self.prepare_targets(batched_inputs, images)
        frame_targets = self.split_frame_targets(frame_targets_all, B)
        video_targets = self.split_video_targets(video_targets_full)
        
        positive_indices = [set() for i in range(B)]
        
        prev_clip_indices = None
        prev_aux_clip_indices = None
        output_q = self.vita_module.query_feat.weight.unsqueeze(1).repeat(1, L*B, 1) # cQ, LB, C
        losses = dict()
        
        h, c = self.convlstm.init_hidden(B*cQ, self.feature_shape) # roi aligned feature
        for c_i in range(num_clips):
            images = []
            for video in batched_inputs:
                for frame in video["image"][c_i*self.len_clip_window : (c_i+1)*self.len_clip_window]:
                    images.append(frame.to(self.device))
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            texts = []
            for video in batched_inputs:
                sentence = video["expressions"]
                sentence = sentence + '.' if not sentence.endswith('.') else sentence
                texts.extend([sentence]*self.len_clip_window)

            frame_targets_per_clip = frame_targets[c_i]
            clip_targets  = video_targets[c_i]
            
            with torch.no_grad():
                dino_outputs = self.dino(images.tensor, captions=texts)
                
            enhanced_features = dino_outputs['enhanced_features']
            backbone_features = dino_outputs['backbone_features']
            logits = dino_outputs['pred_logits'][None] # 1, B, 900, C
            
            T = num_frames
            BcT = len(images)
            cT = self.len_clip_window
            B = BcT // cT
            frame_queries = torch.stack(dino_outputs['hs'][-self.num_decoder_layers:], dim=0)
            encoded_sentence = dino_outputs['encoded_sentence'][::cT, :] # B, C
            
            _, _, _, C = frame_queries.shape
            # top_indices = torch.topk(frame_queries.max(-1)[0], 100, dim=2)[1] # TODO : query topk method
            top_indices = torch.topk(logits.max(-1)[0], 100, dim=2)[1]
            # frame_queries = torch.gather(frame_queries, 2, top_indices.unsqueeze(-1).repeat(1,1,1,C))
            frame_queries = torch.gather(frame_queries, 2, top_indices.unsqueeze(-1).repeat(self.num_decoder_layers,1,1,C)) 
            L, BcT, fQ, C = frame_queries.shape
            
            _mask_features, multi_scale_features = self.mask_features_from_gdino(enhanced_features, backbone_features)
            _mask_features = self.vita_module.vita_mask_features(_mask_features)
            _mask_features = _mask_features.view(B, cT, *_mask_features.shape[-3:])
            
            # bipartite matching-based loss
            # assert self.freeze_detector, "Detector should be frozen"
            
            vita_outputs, output_q = self.vita_module(frame_queries, pre_memory, output_q)
            vita_outputs["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", vita_outputs["pred_mask_embed"], _mask_features)
            
            for out in vita_outputs["aux_outputs"]:
                out["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", out["pred_mask_embed"], _mask_features)

            roi_features, bbox = self.get_roi_features(vita_outputs, backbone_features, feature_shape=self.feature_shape) # cQ*B, T, C, H, W
            
            for f in range(self.len_clip_window):
                h, c = self.convlstm(roi_features[:, f], (h, c), bbox[:, f])
            
            genvis_loss_dict, out_clip_indices, aux_clip_indices_list = self.genvis_criterion(
                                                                            outputs=vita_outputs,
                                                                            clip_targets=clip_targets,
                                                                            frame_targets=frame_targets_per_clip, 
                                                                            # frame_indices=fg_indices, 
                                                                            prev_clip_indices=prev_clip_indices, 
                                                                            prev_aux_clip_indices=prev_aux_clip_indices,
                                                                        )

            for i, (s_i, t_i) in enumerate(out_clip_indices[-B:]):
                    for s in s_i:
                        positive_indices[i].add(s.item())
            
            
            genvis_weight_dict = self.genvis_criterion.weight_dict
            
            loss_dict_keys = list(genvis_loss_dict.keys())
            for k in loss_dict_keys:
                if k in genvis_weight_dict:
                    genvis_loss = genvis_loss_dict.pop(k)
                    genvis_loss_dict[f"{k}_clip{c_i}"] = genvis_loss * genvis_weight_dict[k]
            losses.update(genvis_loss_dict)
            
            # update memory
            pre_memory["k"].append(vita_outputs["pre_memory"]["k"])
            pre_memory["v"].append(vita_outputs["pre_memory"]["v"])
            # pre_memory["motion"].append(vita_outputs["pre_memory"]["motion"])
            
            # update clip indices
            prev_clip_indices = out_clip_indices
            prev_aux_clip_indices = aux_clip_indices_list
        
        # motion_query = torch.cat(pre_memory['motion']).mean(dim=(0,1)).permute(1,0,2) # cQ, B, C 
        # motion_emb = self.motion2text(motion_query) # cQ, B, tC
        text_emb = self.text2text(encoded_sentence) # B, tC/
        last_hidden_state = nn.AdaptiveAvgPool2d((1,1))(h.reshape(cQ, B, *h.shape[-3:])).reshape(cQ, B, -1) # cQ, B, flattened feature_shape
        motion_emb = self.motion2text(last_hidden_state) # cQ, B, C
        # motion_emb = self.motion2text(h.reshape(cQ, B, h.shape[-3], feature_shape[0]*feature_shape[1]))

        sim = F.cosine_similarity(motion_emb, text_emb[None], dim=-1).permute(1,0) # B, cQ
        labels = torch.zeros_like(sim)
        for i, ind_set in enumerate(positive_indices):
            labels[i][list(ind_set)] = 1
        
        pos_weight =  ((B*cQ - labels.sum()) / labels.sum()).clamp(max=10)
        grounding_loss_dict = {"loss_grounding" : F.binary_cross_entropy_with_logits(sim, labels, pos_weight=pos_weight)}
        grounding_loss_dict_keys = list(grounding_loss_dict.keys())
        
        for k in grounding_loss_dict_keys:
            if k in genvis_weight_dict:
                grounding_loss_dict[k] *= genvis_weight_dict[k]
        losses.update(grounding_loss_dict)
        
        return losses
    
    def get_roi_features(self, outputs, backbone_features):
        bmask = outputs['pred_masks'][-1] > 0
        B, cQ, T, H, W = bmask.shape
        bmask = bmask.permute(1,0,2,3,4).flatten(0,2) # cQ*B*T, H, W
        bbox = torch.zeros(B*cQ*T, 4).to(bmask.device)
        mask_sum = bmask.sum(dim=(-1,-2))
        non_zero_indices = mask_sum.nonzero(as_tuple=True)
        nonzero_bmask = bmask[non_zero_indices]
        bbox[non_zero_indices] = torchvision.ops.masks_to_boxes(nonzero_bmask)
        denominator = torch.tensor([W, H, W, H]).to(bbox.device)
        
        # normalize with H,W
        bbox = bbox.reshape(cQ*B, T, 4)
        bbox = torch.cat((bbox.min(dim=1)[0][:,:2], bbox.max(dim=1)[0][:,2:]), dim=1)[:,None].repeat(1, T, 1).flatten(0,1)
        bbox = bbox / denominator[None]
        
        # x1, y1, x2, y2 -> cx, cy, dx, dy
        cx, cy = bbox[:, [0,2]].mean(dim=1), bbox[:, [1,3]].mean(dim=1)
        dx = bbox[:, 2] - cx
        dy = bbox[:, 3] - cy
        center_bbox = torch.stack((cx, cy, dx, dy), dim=1)
        
        ind = torch.arange(B*cQ*T).to(bbox.device)
        ind_bbox = torch.concat((ind[:,None], bbox), dim=1) # cQ*B*T, 5 (Batch + bbox)

        _roi_features = torchvision.ops.roi_align(backbone_features[0].decompose()[0].repeat(cQ, 1, 1, 1), ind_bbox, output_size=self.feature_shape)
        roi_features = torch.zeros_like(_roi_features)
        roi_features[non_zero_indices] = _roi_features[non_zero_indices]
        
        roi_features = roi_features.reshape(cQ*B, T, *roi_features.shape[-3:])
        center_bbox = center_bbox.reshape(cQ*B, T, 4)
        return roi_features, center_bbox
        
    def mask_features_from_gdino(self, enhanced_features, backbone_features):
        x = self.feature_proj(backbone_features[0].decompose()[0]).float()
        cur_fpn = self.lateral_conv(x)
        
        for i in range(self.feature_utilize_level):
            cur_fpn = cur_fpn + F.interpolate(self.multiscale_feature_proj[i](enhanced_features[i].permute(0,3,1,2).contiguous()), size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
        y = self.output_conv(cur_fpn)
        
        return y, enhanced_features
        
    def split_frame_targets(self, frame_targets, batch_size):
        T = self.num_frames
        W = self.len_clip_window
        num_clips = T // W

        frame_targets = [frame_targets[b_i*T:(b_i+1)*T] for b_i in range(batch_size)]

        frame_targets_splits = dict()
        for frame_targets_per_batch in frame_targets:
            for clip_idx in range(num_clips):
                if not clip_idx in frame_targets_splits:
                    frame_targets_splits[clip_idx] = []
        
                frame_targets_splits[clip_idx] += frame_targets_per_batch[clip_idx*W:(clip_idx+1)*W]
        
        return list(frame_targets_splits.values())

    def split_video_targets(self, clip_targets):
        clip_len = self.len_clip_window

        clip_target_splits = dict()
        for targets_per_video in clip_targets:
            labels = targets_per_video["labels"] # Ni (number of instances)
            # texts = targets_per_video["texts"] # 1, Ct
            ids = targets_per_video["ids"] # Ni, T 
            merged_masks = targets_per_video["merged_masks"] # T, H, W
            masks = targets_per_video["masks"] # Ni, T, H, W
            frame_idx = targets_per_video["frame_idx"] # T

            # merged_masks_splits = merged_masks.split(clip_len, dim=0)
            masks_splits = masks.split(clip_len, dim=1)
            ids_splits = ids.split(clip_len, dim=1)

            prev_valid = torch.zeros_like(labels).bool()
            for clip_idx, (_masks, _ids) in enumerate(zip(masks_splits, ids_splits)):
                valid_inst = _masks.sum(dim=(1,2,3)) > 0.
                new_inst   = (prev_valid == False) & (valid_inst == True)

                if not clip_idx in clip_target_splits:
                    clip_target_splits[clip_idx] = []

                clip_target_splits[clip_idx].append(
                    {
                        # "texts" : texts,
                        "merged_masks": merged_masks[clip_idx*clip_len:(clip_idx+1)*clip_len],
                        "labels": labels, "ids": _ids, "masks": _masks,
                        "video_len": targets_per_video["video_len"], 
                        "frame_idx": frame_idx[clip_idx*clip_len:(clip_idx+1)*clip_len],
                        "valid_inst": valid_inst,
                        "new_inst": new_inst,
                    }
                )

                prev_valid = prev_valid | valid_inst

        return list(clip_target_splits.values())

    def split_fg_indices(self, fg_indices, batch_size):
        L = len(fg_indices)
        T = self.num_frames
        W = self.len_clip_window
        num_clips = T // W

        fg_indices_splits = []
        for L_i in range(L):
            fg_indices_Li = [fg_indices[L_i][b_i*T:(b_i+1)*T] for b_i in range(batch_size)]
            fg_indices_Li_splits = dict()
            for b_i in range(batch_size):
                for clip_idx in range(num_clips):
                    if not clip_idx in fg_indices_Li_splits:
                        fg_indices_Li_splits[clip_idx] = []
                    fg_indices_Li_splits[clip_idx] += fg_indices_Li[b_i][clip_idx*W:(clip_idx+1)*W]
            fg_indices_splits.append(fg_indices_Li_splits)

        fg_indices_splits_clips = []
        for clip_idx in range(num_clips):
            per_clip = []
            for L_i in range(L):
                per_clip.append(fg_indices_splits[L_i][clip_idx])
            fg_indices_splits_clips.append(per_clip)
            
        return fg_indices_splits_clips

    def inference(self, batched_inputs):

        """
        For best speed, use GPU as the primary storage device.
        However, when inferring long videos (e.g., OVIS), you will need a GPU of 32G or higher. 
        If you have less than 24G GPUs to infer OVIS, please uncomment the code line below.
        """
        # to_store = self.device if num_frames <= 36 else "cpu"
        
        mask_features = []
        num_frames = len(batched_inputs["image"])
        to_store = self.device if num_frames <= 72 else "cpu"
        
        pre_memory = {"k": [], "v": []}

        output_q = self.vita_module.query_feat.weight.unsqueeze(1).repeat(1, 1, 1) # cQ, LB, C, note L=1 B=1
        cQ, LB, C = output_q.shape


        clip_mask_embed = []
        mask_features = []
        
        h, c = self.convlstm.init_hidden(1*cQ, self.feature_shape) # roi aligned feature
        for i in range(math.ceil(num_frames / self.len_clip_window)):
            images = batched_inputs["image"][i*self.len_clip_window : (i+1)*self.len_clip_window]
            images = [(x.to(self.device) - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            
            T = images.tensor.shape[0]
            texts = []
            sentence = batched_inputs["expressions"]
            sentence = sentence + '.' if not sentence.endswith('.') else sentence
            texts.extend([sentence]*T)

            dino_outputs = self.dino(images.tensor, captions=texts)
            
            enhanced_features = dino_outputs['enhanced_features']
            backbone_features = dino_outputs['backbone_features']
            encoded_sentence = dino_outputs['encoded_sentence'][0, :][None]
            logits = dino_outputs['pred_logits'][None] # 1, B, 900, C
            
            frame_queries = torch.stack(dino_outputs['hs'][-self.num_decoder_layers:], dim=0) # last 3 layer from gdino
            _, _, _, C = frame_queries.shape
            top_indices = torch.topk(logits.max(-1)[0], 100, dim=2)[1]
            # top_indices = torch.topk(frame_queries.max(-1)[0], 100, dim=2)[1]
            frame_queries = torch.gather(frame_queries, 2, top_indices.unsqueeze(-1).repeat(self.num_decoder_layers,1,1,C)) 
            # frame_queries = torch.gather(frame_queries, 2, top_indices.unsqueeze(-1).repeat(1,1,1,C)) 
            L, BT, fQ, C = frame_queries.shape
            
            _mask_features, multi_scale_features = self.mask_features_from_gdino(enhanced_features, backbone_features)
            _mask_features = self.vita_module.vita_mask_features(_mask_features)
            _mask_features = _mask_features.view(1, T, *_mask_features.shape[-3:])
            mask_features.append(_mask_features.squeeze(0))
            
            # output_q = output_q + encoded_sentence[None]
            vita_outputs, output_q = self.vita_module(frame_queries, pre_memory, output_q)
            clip_mask_embed.append(vita_outputs["pred_mask_embed"].squeeze(1)) # squeeze batch
            vita_outputs["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", vita_outputs["pred_mask_embed"], _mask_features)
            
            roi_features, bbox = self.get_roi_features(vita_outputs, backbone_features) # cQ*B, T, C, H, W
            
            for f in range(T):
                h, c = self.convlstm(roi_features[:, f], (h, c), bbox[:, f])
                
            # update memory
            pre_memory["k"].append(vita_outputs["pre_memory"]["k"])
            pre_memory["v"].append(vita_outputs["pre_memory"]["v"])
            
            del vita_outputs

        interim_size = images.tensor.shape[-2:]
        image_size   = images.image_sizes[0]  # image size without padding after data augmentation

        out_height = batched_inputs.get("height", image_size[0])  # raw image size before data augmentation
        out_width  = batched_inputs.get("width", image_size[1])

        # del outputs, batched_inputs
        
        last_hidden_state = nn.AdaptiveAvgPool2d((1,1))(h.reshape(cQ, 1, *h.shape[-3:])).reshape(cQ, 1, -1) # cQ, B, flattened feature_shape
        motion_emb = self.motion2text(last_hidden_state) # cQ, B, C
        text_emb = self.text2text(encoded_sentence) # B, tC
        sim = F.cosine_similarity(motion_emb, text_emb[None], dim=-1).permute(1,0) # B, cQ
        
        stacked_mask_embed = torch.stack(clip_mask_embed) # nC, B(1), cQ, C
        
        where = sim > 0.
        indices = where.nonzero(as_tuple=False)[:,1]
        
        # memory explodes when all indices are selected
        if indices.numel() > 5:
            indices = sim.topk(5)[1].flatten()
        # indices = sim.argmax()
        selected_mask_embed = stacked_mask_embed.permute(2,0,1,3)[indices] # qnum, nC, B(1), C
        # selected_mask_embed = stacked_mask_embed.permute(2,0,1,3)[indices].unsqueeze(0)
        
        video_mask = []
        for i, mf in enumerate(mask_features):
            # mf: T, C, H, W
            clip_mask = torch.einsum("qbc,tchw->qthw", selected_mask_embed[:,i,:,:], mf) # qnum, T, H, W
            # upsample masks
            video_mask.append(clip_mask)
            
        mask_pred = torch.cat(video_mask, dim=1).float()
        mask_pred = retry_if_cuda_oom(F.interpolate)(
            mask_pred,
            size=interim_size,
            mode="bilinear",
            align_corners=False,
        ) # Q, T, H, W
                
        del clip_mask_embed, mask_features
        
        mask_pred = mask_pred[:, :, : image_size[0], : image_size[1]]
        mask_pred = retry_if_cuda_oom(F.interpolate)(
            mask_pred, size=(out_height, out_width), mode="bilinear", 
            align_corners=False
        ) > 0.
        mask_pred = retry_if_cuda_oom(torch.sum)(mask_pred, dim=0).clamp(max=1)[None]
    

        processed_results = {
            "image_size": (out_height, out_width),
            "pred_masks": mask_pred.to(self.device),
        }

        return processed_results
    

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Sequential(
            Conv2d(
                in_channels=self.input_dim + self.hidden_dim,
                out_channels=2 * self.hidden_dim,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=self.bias,
                norm = get_norm("GN", 2 * self.hidden_dim),
            ),
            Conv2d(
                in_channels=2 * self.hidden_dim,
                out_channels=4 * self.hidden_dim,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=self.bias,
                norm = get_norm("GN", 4 * self.hidden_dim),
                activation=F.relu,
            )
        )

        self.bbox_enc = Conv2d(in_channels=4, out_channels=self.input_dim, kernel_size=1, bias=False)
        
    def forward(self, input_tensor, cur_state, bbox):
        h_cur, c_cur = cur_state
        bbox = self.bbox_enc(bbox[:,:,None, None])
        input_tensor = input_tensor + bbox
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.bbox_enc.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.bbox_enc.weight.device))