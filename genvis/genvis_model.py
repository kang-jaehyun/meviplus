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

from typing import Any, Optional, Tuple, Type
import numpy as np

from xdecoder.modeling.BaseModel import BaseModel
from xdecoder.modeling import build_model
from positional_encodings.torch_encodings import PositionalEncoding2D

@META_ARCH_REGISTRY.register()
class Genvis(Vita):

    @configurable
    def __init__(
        self,
        len_clip_window: int,
        genvis_criterion: nn.Module,
        motion2text: nn.Module,
        xdecoder: nn.Module,
        score_decoder: nn.Module,
        mask2former_hidden_dim: int,
        xdecoder_hidden_dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.eps = 1e-6
        self.len_clip_window = len_clip_window
        self.genvis_criterion = genvis_criterion
        self.freeze_detector = kwargs["freeze_detector"]
        # self.motion2text = motion2text
        
        self.mask2former_hidden_dim = mask2former_hidden_dim
        self.xdecoder_hidden_dim = xdecoder_hidden_dim

        self.query_proj = MLP(
                            input_dim=xdecoder_hidden_dim, 
                            hidden_dim=xdecoder_hidden_dim, 
                            output_dim=mask2former_hidden_dim,
                            num_layers=1
                            )
        self.cls_proj = MLP(
                            input_dim=xdecoder_hidden_dim, 
                            hidden_dim=xdecoder_hidden_dim, 
                            output_dim=mask2former_hidden_dim,
                            num_layers=1
                            )
                                        
        self.iou_predictor_head = MLP(
                                    input_dim=mask2former_hidden_dim,
                                    hidden_dim=mask2former_hidden_dim,
                                    output_dim=20, # TODO
                                    num_layers=3
                                    )

        # LSTM
        lstm_hidden_dim = mask2former_hidden_dim // 2
        self.lstm = nn.LSTM(
            input_size=mask2former_hidden_dim, 
            hidden_size=lstm_hidden_dim, 
            num_layers=1,
            batch_first=False,
            bidirectional=True,
        )
        self.initial_hidden = nn.Embedding(2, lstm_hidden_dim)
        self.initial_cell = nn.Embedding(2, lstm_hidden_dim)
        
        ##########################
        # for roi, box embedding #
        ##########################
        self.roi_feature_shape = (14,14)
        self.valid_roi_emb = nn.Embedding(1, 256)
        self.invalid_roi_emb = nn.Embedding(1, 256)
        
        self.valid_box_emb = nn.Embedding(1, 256)
        self.invalid_box_emb = nn.Embedding(1, 256)
        
        self.pe_layer = PositionEmbeddingRandom()
        
        self.roi_embed = nn.Sequential(
            Conv2d(xdecoder_hidden_dim*3, mask2former_hidden_dim*4, kernel_size=self.roi_feature_shape, bias=False, norm=get_norm("GN", mask2former_hidden_dim*4)),
            # nn.ReLU(),
            Conv2d(mask2former_hidden_dim*4, mask2former_hidden_dim, kernel_size=1, bias=False, norm=get_norm("GN", mask2former_hidden_dim)),
            # nn.ReLU(),
        )
        

        self.score_decoder = score_decoder
        self.xdecoder = xdecoder.eval().cuda()
        
        # freeze xdecoder
        for param in self.xdecoder.parameters():
            param.requires_grad = False
        
        # for default token init - no effect on result
        self.xdecoder.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(["background", "background"], is_eval=False)
        
    @classmethod
    def from_config(cls, cfg):
        rets = Vita.from_config(cfg)

        # genvis
        rets["vita_module"] = GenVIS(cfg=cfg)

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight  = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight  = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        score_weight = cfg.MODEL.GENVIS.GROUNDING_WEIGHT
        
        sim_weight   = cfg.MODEL.VITA.SIM_WEIGHT
        
        
        genvis_matcher = GenvisHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            cost_score=score_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        genvis_weight_dict = {
            "loss_genvis_ce": class_weight, 
            "loss_genvis_mask": mask_weight, 
            "loss_genvis_dice": dice_weight,

            "loss_grounding": score_weight,
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

        score_decoder = ScoreDecoder(
            nheads=cfg.MODEL.SCORE_DECODER.NHEADS,
            dim_feedforward=cfg.MODEL.SCORE_DECODER.DIM_FEEDFORWARD,
            enc_layers=cfg.MODEL.SCORE_DECODER.ENC_LAYERS,
            dec_layers=cfg.MODEL.SCORE_DECODER.DEC_LAYERS,
            hidden_dim=cfg.MODEL.SCORE_DECODER.HIDDEN_DIM,
            text_dim=cfg.XDECODER.MODEL.ENCODER.MASK_DIM,
            num_frames=cfg.INPUT.SAMPLING_FRAME_NUM,
        )
        
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
            input_dim = cfg.MODEL.VITA.HIDDEN_DIM,
            hidden_dim = cfg.MODEL.VITA.HIDDEN_DIM,
            output_dim = cfg.XDECODER.MODEL.ENCODER.MASK_DIM,
            num_layers = cfg.MODEL.GENVIS.PROJ_LAYERS,
        )
        xdecoder = BaseModel(cfg.XDECODER, build_model(cfg.XDECODER)).from_pretrained(cfg.XDECODER.MODEL.WEIGHTS)
        
        rets.update({
            "len_clip_window": cfg.MODEL.GENVIS.LEN_CLIP_WINDOW,
            "genvis_criterion": genvis_criterion,
            "motion2text": motion2text,
            "xdecoder": xdecoder,
            "mask2former_hidden_dim": cfg.MODEL.VITA.HIDDEN_DIM,
            "xdecoder_hidden_dim": cfg.XDECODER.MODEL.ENCODER.MASK_DIM,
            "score_decoder": score_decoder,
        })

        return rets
    
    def train_model(self, batched_inputs):
        num_frames = len(batched_inputs[0]['image'])
        pre_memory = {"k": [], "v": []}
        num_clips = num_frames // self.len_clip_window
        
        assert num_frames % self.len_clip_window == 0, f"num_frames: {num_frames}, len_clip_window: {self.len_clip_window}"
        
        B = len(batched_inputs)
        L = 3 # TODO : configurable
        cQ = 20 # TODO : configurable
        
        # load image to prepare targets
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
        
        # positive_indices = [set() for i in range(B)]
        
        prev_clip_indices = None
        prev_aux_clip_indices = None
        output_q = self.vita_module.query_feat.weight.unsqueeze(1).repeat(1, L*B, 1) # cQ, LB, C
        losses = dict()
        
        # h, c = self.convlstm.init_hidden(B*cQ, self.roi_feature_shape, self.device) # roi aligned feature
        clip_queries_list = []
        global_memory = []
        roi_list = []
        box_list = []
        
        for c_i in range(num_clips):
            images = []
            for video in batched_inputs:
                for frame in video["image"][c_i*self.len_clip_window : (c_i+1)*self.len_clip_window]:
                    images.append(frame.to(self.device))
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            extra = {}
            texts = []
            cls_token = []
            for video in batched_inputs:
                sentence = video["expressions"]
                gtext = self.xdecoder.model.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings([sentence], name='grounding', token=False, norm=False)
                token_emb = gtext['token_emb']
                tokens = gtext['tokens']
                query_emb = token_emb[tokens['attention_mask'].bool()]
                texts.append(query_emb)
                cls_token.append(gtext['class_emb'])
                
            grounding_tokens = nn.utils.rnn.pad_sequence(texts)
            cls_token = torch.cat(cls_token, dim=0)
            extra['grounding_tokens'] = grounding_tokens.repeat_interleave(self.len_clip_window, dim=1)

            frame_targets_per_clip = frame_targets[c_i]
            clip_targets  = video_targets[c_i]
            
            with torch.no_grad():
                features = self.xdecoder.model.backbone(images.tensor)
                _, frame_queries, _mask_features, multi_scale_features = self.xdecoder.model.sem_seg_head(features, extra=extra, task='grounding_eval')
            
            frame_queries = self.query_proj(frame_queries)
            
            T = num_frames
            BcT = len(images)
            cT = self.len_clip_window
            B = BcT // cT

            L, BcT, fQ, C = frame_queries.shape
            
            _mask_features = self.vita_module.vita_mask_features(_mask_features)
            _mask_features = _mask_features.view(B, cT, *_mask_features.shape[-3:])
            
            # bipartite matching-based loss
            # assert self.freeze_detector, "Detector should be frozen"
            # _, fg_indices = self.criterion(outputs, frame_targets[c_i])
            
            # if self.freeze_detector:
            # losses = dict()
            
            vita_outputs, output_q = self.vita_module(frame_queries, pre_memory, output_q, cls_token)
            vita_outputs["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", vita_outputs["pred_mask_embed"], _mask_features)
            
            
            for out in vita_outputs["aux_outputs"]:
                out["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", out["pred_mask_embed"], _mask_features)

            roi_emb, box_emb = self.get_roi_embeds(vita_outputs, multi_scale_features) # cQ*B, T, C, H, W
            roi_list.append(roi_emb)
            box_list.append(box_emb)
            
            clip_queries_list.append(output_q.reshape(cQ, L*B, C))
            global_memory.append(vita_outputs)
            
            # update memory
            pre_memory["k"].append(vita_outputs["pre_memory"]["k"])
            pre_memory["v"].append(vita_outputs["pre_memory"]["v"])
            
            # pre_memory["motion"].append(vita_outputs["pre_memory"]["motion"])
            
            # update clip indices
            # prev_clip_indices = out_clip_indices
            # prev_aux_clip_indices = aux_clip_indices_list
        
        global_outputs = self.unify_memory(global_memory)
        clip_queries = torch.stack(clip_queries_list, dim=1)
        cQ, cN, LB, C = clip_queries.shape
        clip_queries = clip_queries.permute(1,0,2,3) # cN, cQ, LB, C
        
        rois = torch.cat(roi_list, dim=2) # cQ, LB, cNcT, C
        boxes = torch.cat(box_list, dim=2) # cQ, LB, cNcT, C
        
        object_feature = (rois + boxes).flatten(0,1).permute(1,0,2) # cNcT, cQ * LB, C
        lstm_output, (hn, cn) = self.lstm(object_feature, (self.initial_hidden.weight.unsqueeze(1).repeat_interleave(LB*cQ, dim=1), self.initial_cell.weight.unsqueeze(1).repeat_interleave(LB*cQ, dim=1)))
        # video_iou = torch.stack(ious_list, dim=0).mean(dim=0)

        output_q = output_q + self.cls_proj(cls_token)[None].repeat(cQ, L, 1)
        output = self.score_decoder(lstm_output.reshape(cN*cT, cQ, LB, C), cls_token)
        scores = self.iou_predictor_head(output[-1])
        global_outputs['scores'] = scores
        
        genvis_loss_dict, out_clip_indices, aux_clip_indices_list, iou = self.genvis_criterion(
                                                                        outputs=global_outputs,
                                                                        clip_targets=video_targets_full,
                                                                        frame_targets=frame_targets_all, 
                                                                        # frame_indices=fg_indices, 
                                                                        prev_clip_indices=prev_clip_indices, 
                                                                        prev_aux_clip_indices=prev_aux_clip_indices,
                                                                    )
        
        
        
        genvis_weight_dict = self.genvis_criterion.weight_dict
        
        loss_dict_keys = list(genvis_loss_dict.keys())
        for k in loss_dict_keys:
            if k in genvis_weight_dict:
                genvis_loss = genvis_loss_dict.pop(k)
                genvis_loss_dict[f"{k}_video"] = genvis_loss * genvis_weight_dict[k]
        losses.update(genvis_loss_dict)
        
        
        fusion_loss_dict = {"loss_grounding": F.mse_loss(scores, iou)}
        for k in fusion_loss_dict.keys():
            if k in genvis_weight_dict:
                fusion_loss_dict[k] *= genvis_weight_dict[k]
                
        losses.update(fusion_loss_dict)
        
        return losses

    def unify_memory(self, global_memory):
        outputs = {}
        pred_logits = torch.stack([c['pred_logits'] for c in global_memory], dim=0)
        pred_logits = pred_logits.mean(dim=0)
        
        pred_mask_embed = torch.stack([c['pred_mask_embed'] for c in global_memory], dim=0)
        pred_masks = torch.cat([c['pred_masks'] for c in global_memory], dim=3) # L, B, cQ, T, H, W
        
        outputs['pred_logits'] = pred_logits
        outputs['pred_mask_embed'] = pred_mask_embed
        outputs['pred_masks'] = pred_masks
        
        if 'aux_outputs' in global_memory[0]:
            outputs['aux_outputs'] = []
            for i,_ in enumerate(global_memory[0]['aux_outputs']):
                aux_outputs = [c['aux_outputs'][i] for c in global_memory]
                aux_outputs = self.unify_memory(aux_outputs)
                outputs['aux_outputs'].append(aux_outputs)
    
        return outputs

    def _embed_boxes(self, bbox: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        bbox = bbox + 0.5  # Shift to center of pixel
        coords = bbox.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, image_size)
        
        return corner_embedding
    
    def get_roi_embeds(self, outputs, multi_scale_features):
        L = 3
        sig_mask = outputs['pred_masks'].flatten(0,1).sigmoid() # LB, cQ, T, H, W
        objectness_logits = outputs['pred_logits'].flatten(0,1).softmax(-1)[..., 0]
        calibrated_mask = objectness_logits[..., None, None, None] * sig_mask
        calibrated_mask = calibrated_mask > 0.5
        LB, cQ, T, H, W = calibrated_mask.shape
        calibrated_mask = calibrated_mask.permute(1,0,2,3,4).flatten(0,2) # cQ*LB*T, H, W
        bbox = torch.zeros(LB*cQ*T, 4).to(calibrated_mask.device)
        mask_sum = calibrated_mask.sum(dim=(-1,-2))
        valid_indices = mask_sum.nonzero(as_tuple=True)
        nonzero_calibrated_mask = calibrated_mask[valid_indices]
        bbox[valid_indices] = torchvision.ops.masks_to_boxes(nonzero_calibrated_mask)
        
        objectness_embed = self.invalid_roi_emb.weight.repeat(cQ*LB*T, 1)
        objectness_embed[valid_indices] = self.valid_roi_emb.weight
        
        boxness_embed = self.invalid_box_emb.weight.repeat(cQ*LB*T, 1)
        boxness_embed[valid_indices] = self.valid_box_emb.weight
        
        roi_features = []
        for f in multi_scale_features:
            _H, _W = f.shape[-2:]
            _roi_features = torchvision.ops.roi_align(
                                f.repeat(cQ*L, 1, 1, 1), # cQ*LB*T, C, _H, _W
                                bbox[None].unbind(1), 
                                output_size=self.roi_feature_shape, 
                                spatial_scale=(_H/H)
                            )
            roi_features.append(_roi_features)
            
        roi_features = torch.cat(roi_features, dim=1) # cQ*LB*T, C*3, _H, _W
        roi_embed = self.roi_embed(roi_features).reshape(cQ*LB*T, self.mask2former_hidden_dim)
        roi_embed += objectness_embed
        
        box_embed = self._embed_boxes(bbox, image_size=(H,W)).reshape(cQ*LB*T, -1)
        box_embed += boxness_embed
        
        return roi_embed.view(cQ, LB, T, -1), box_embed.view(cQ, LB, T, -1)
        
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
        
        pre_memory = {"k": [], "v": [], "motion": []}

        output_q = self.vita_module.query_feat.weight.unsqueeze(1).repeat(1, 1, 1) # cQ, LB, C, note L=1 B=1
        cQ, LB, C = output_q.shape


        clip_mask_embed = []
        mask_features = []
        clip_queries_list = []
        global_memory = []
        roi_list = []
        box_list = []
        
        for i in range(math.ceil(num_frames / self.len_clip_window)):
            images = batched_inputs["image"][i*self.len_clip_window : (i+1)*self.len_clip_window]
            images = [(x.to(self.device) - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            T = images.tensor.shape[0]
            
            extra = {}
            texts = []
            cls_token = []
            # for video in batched_inputs:
            sentence = batched_inputs["expressions"]
            gtext = self.xdecoder.model.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings([sentence], name='grounding', token=False, norm=False)
            token_emb = gtext['token_emb']
            tokens = gtext['tokens']
            query_emb = token_emb[tokens['attention_mask'].bool()]
            texts.append(query_emb)
            cls_token.append(gtext['class_emb'])
            
            grounding_tokens = nn.utils.rnn.pad_sequence(texts)
            cls_token = torch.cat(cls_token, dim=0)
            extra['grounding_tokens'] = grounding_tokens.repeat_interleave(T, dim=1)

            with torch.no_grad():
                features = self.xdecoder.model.backbone(images.tensor)
            outputs, frame_queries, _mask_features, multi_scale_features = self.xdecoder.model.sem_seg_head(features, extra=extra, task='grounding_eval')
            frame_queries = self.query_proj(frame_queries)

            L, BT, fQ, C = frame_queries.shape
            
            output_q = output_q + self.cls_proj(cls_token)[None].repeat(cQ, 1, 1)
            vita_outputs, output_q = self.vita_module(frame_queries, pre_memory, output_q, cls_token)
            clip_mask_embed.append(vita_outputs["pred_mask_embed"].squeeze(1)) # squeeze batch
            clip_queries_list.append(output_q)
            
            _mask_features = self.vita_module.vita_mask_features(_mask_features)
            _mask_features = _mask_features.view(1, T, *_mask_features.shape[-3:])
            mask_features.append(_mask_features.squeeze(0))
                
            
            vita_outputs["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", vita_outputs["pred_mask_embed"], _mask_features)
            roi_emb, box_emb = self.get_roi_embeds(vita_outputs, multi_scale_features) # cQ*B, T, C, H, W
            
            global_memory.append(vita_outputs)
            roi_list.append(roi_emb)
            box_list.append(box_emb)

            # update memory
            pre_memory["k"].append(vita_outputs["pre_memory"]["k"])
            pre_memory["v"].append(vita_outputs["pre_memory"]["v"])
            
            del vita_outputs


        interim_size = images.tensor.shape[-2:]
        image_size   = images.image_sizes[0]  # image size without padding after data augmentation

        out_height = batched_inputs.get("height", image_size[0])  # raw image size before data augmentation
        out_width  = batched_inputs.get("width", image_size[1])

        del outputs, images, batched_inputs
        
        # clip_queries = torch.stack(clip_queries_list, dim=1)
        # cQ, cN, B, C = clip_queries.shape
        # clip_queries = clip_queries.permute(1,0,2,3) # cN, cQ, B, C
        global_outputs = self.unify_memory(global_memory)
        
        rois = torch.cat(roi_list, dim=2) # cQ, LB, cNcT, C
        boxes = torch.cat(box_list, dim=2) # cQ, LB, cNcT, C
        
        object_feature = (rois + boxes).flatten(0,1).permute(1,0,2) # cNcT, cQ * LB, C
        lstm_output, (hn, cn) = self.lstm(object_feature, (self.initial_hidden.weight.unsqueeze(1).repeat_interleave(LB*cQ, dim=1), self.initial_cell.weight.unsqueeze(1).repeat_interleave(LB*cQ, dim=1)))
        
        output = self.score_decoder(lstm_output.reshape(num_frames, cQ, LB, C), cls_token)
        iou_pred = self.iou_predictor_head(output[-1]).clamp(min=self.eps)
        query_logit = global_outputs['pred_logits'].softmax(-1)[..., 0][0]
        
        final_score = gmean(torch.cat([iou_pred, query_logit], dim=0), dim=0)
        # where = sim > 0.5
        # indices = where.nonzero(as_tuple=False)[:,1]
    
        indices = final_score.topk(1)[1].flatten()
        # indices = sim.argmax()
        
        stacked_mask_embed = torch.stack(clip_mask_embed).permute(2,0,1,3) # nC, B(1), cQ, C -> cQ, nC, B(1), C
        video_mask = []
        for i, mf in enumerate(mask_features):
            # mf: T, C, H, W
            all_clip_mask = torch.einsum("qbc,tchw->qthw", stacked_mask_embed[:,i,:,:], mf) # qnum, T, H, W
            # upsample masks
            video_mask.append(all_clip_mask)
        
            
        mask_pred = torch.cat(video_mask, dim=1).float()
        H_cur, W_cur = mask_pred.shape[-2:]
        H_small, W_small = image_size[0] * H_cur // interim_size[0], image_size[1] * W_cur // interim_size[1]
        all_mask_pred = mask_pred[:,:, :H_small, :W_small] > 0.
        mask_pred = mask_pred[indices]
        
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
            "all_pred_masks": all_mask_pred.to(self.device),
            "image_small_size": (H_small, W_small),
        }
    
        return processed_results
    
class ScoreDecoder(nn.Module):
    def __init__(
        self, 
        nheads: int,
        dim_feedforward: int,
        enc_layers: int,
        dec_layers: int,
        hidden_dim: int,
        text_dim: int,
        num_frames: int,
        ):
        """
        
        Score Decoder
        
        """
        
        super().__init__()
        self.num_heads = nheads
        self.enc_layers = enc_layers
        self.num_layers = dec_layers

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_frames = num_frames
        
        self.src_embed = nn.Identity()
        self.text_embed = MLP(
            input_dim=text_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=3,
        )
        
        self.iou_token = nn.Embedding(1, hidden_dim) 
        
        cQ = 20
        self.query_pos = nn.Embedding(cQ, hidden_dim)
        
        pre_norm = False
        if enc_layers > 0:
            self.enc_self_attn = nn.ModuleList()
            self.enc_ffn = nn.ModuleList()
            for _ in range(self.enc_layers):
                self.enc_self_attn.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    ),
                )
                self.enc_ffn.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )
                
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
    def encode_clip_query(self, clip_q):
        """
        input shape (frame_query)   : cQ, LB, tC
        output shape (frame_query)  : cQ, LB, tC
        """


        for i in range(self.enc_layers):
            clip_q = self.enc_self_attn[i](clip_q)
            clip_q = self.enc_ffn[i](clip_q)
            
        return clip_q
        
    def forward(self, clip_q, cls_token):
        # clip_q : cN, cQ, LB, C
        # cls_token : B, T, tC 
        if not self.training:
            clip_q = clip_q[:,:,-1:,:]
        
        cN, cQ, LB, C = clip_q.shape
        B, tC = cls_token.shape
        L = LB // B

        cls_token = cls_token.repeat(L,1)
        
        clip_q = self.encode_clip_query(clip_q.reshape(cN, cQ*LB, C)).reshape(cN*cQ, LB, C) # cNcQ, B, C
        src = self.src_embed(clip_q) # cNcQ, LB, C
        cls_token = self.text_embed(cls_token) # LB, C
        output = torch.cat([cls_token[None].repeat_interleave(cQ, dim=0) + self.query_pos.weight.unsqueeze(1), self.iou_token.weight[None].repeat(1,LB,1)], dim=0) # cQ, LB, C
        output_pos = output
        # each Query attends to each embeddings
        mask = (~torch.cat([torch.eye(cQ), torch.ones(1, cQ)], dim=0).to(dtype=torch.bool, device=cls_token.device)).repeat_interleave(cN, dim=1) # cN should be number of embeddings releated to each Q
        
        # decoder_outputs = []
        for i in range(self.num_layers):
            output += output_pos
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=None,
            )
            
            output += output_pos
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src,
                memory_mask=mask,
                memory_key_padding_mask=None,
                pos=None, query_pos=None
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            output = self.decoder_norm(output)
        
        return output

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
    
def gmean(input_x, dim):
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))