import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom

from vita.vita_model import Vita
from .modeling.genvis_criterion import GenvisSetCriterion
from .modeling.genvis_matcher import GenvisHungarianMatcher
from .modeling.genvis import GenVIS

from transformers import BertModel, RobertaModel
# import clip

from vita.modeling.transformer_decoder.vita import FFNLayer, CrossAttentionLayer, SelfAttentionLayer, MLP, _get_activation_fn

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from typing import Optional
from groundingdino.util.inference import load_model, load_image, predict, annotate

@META_ARCH_REGISTRY.register()
class Genvis(Vita):

    @configurable
    def __init__(
        self,
        len_clip_window: int,
        genvis_criterion: nn.Module,
        # text_encoder: nn.Module,
        # vis2text: nn.Module,
        text_decoder: nn.Module,
        freeze_text_encoder: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.len_clip_window = len_clip_window
        self.genvis_criterion = genvis_criterion
        self.freeze_detector = kwargs["freeze_detector"]
        # self.vis2text = vis2text
        # self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        self.feature_proj = nn.ModuleList()
        for i in range(4):
            self.feature_proj.append(nn.Conv2d(96*(2**i), 256*(2**i), kernel_size=1, bias=False))
        
        

        self.dino = load_model("/workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

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
        fusion_mask_weight = cfg.MODEL.GENVIS.FUSION_MASK_WEIGHT
        fusion_dice_weight = cfg.MODEL.GENVIS.FUSION_DICE_WEIGHT
        
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
            "loss_fusion_mask": fusion_mask_weight,
            "loss_fusion_dice": fusion_dice_weight,
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
        # text_encoder = RobertaModel.from_pretrained('roberta-base')
        text_decoder = TextDecoder(
            nheads=cfg.MODEL.TEXT_DECODER.NHEADS,
            dim_feedforward=cfg.MODEL.TEXT_DECODER.DIM_FEEDFORWARD,
            enc_layers=cfg.MODEL.TEXT_DECODER.ENC_LAYERS,
            dec_layers=cfg.MODEL.TEXT_DECODER.DEC_LAYERS,
            hidden_dim=cfg.MODEL.TEXT_DECODER.HIDDEN_DIM,
            mask_dim=cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
            num_frames=cfg.INPUT.SAMPLING_FRAME_NUM,
        )
        # resizer = FeatureResizer(
        #     input_feat_size=768,
        #     output_feat_size=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
        #     dropout=0.1,
        # )
        
        # vis2text = MLP(
        #     input_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
        #     hidden_dim = cfg.MODEL.TEXT_DECODER.TEXT_DIM,
        #     output_dim = cfg.MODEL.TEXT_DECODER.TEXT_DIM,
        #     num_layers = cfg.MODEL.TEXT_DECODER.PROJ_LAYERS,
        # )

        rets.update({
            "len_clip_window": cfg.MODEL.GENVIS.LEN_CLIP_WINDOW,
            "genvis_criterion": genvis_criterion,
            # "text_encoder": text_encoder,
            # "vis2text": vis2text,
            "text_decoder": text_decoder,
            "freeze_text_encoder": cfg.MODEL.GENVIS.FREEZE_TEXT_ENCODER,
        })

        return rets

    def train_model(self, batched_inputs):
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        texts = []
        for video in batched_inputs:
            sentence = video["expressions"]
            sentence = sentence + '.' if not sentence.endswith('.') else sentence
            texts.extend([sentence]*self.num_frames)
            
        # text_emb = [x['lang_tokens'].to(self.device) for x in batched_inputs]
        # text_emb = torch.cat(text_emb, dim=0)

        # text_mask = [x['lang_mask'].to(self.device) for x in batched_inputs]
        # text_mask = torch.cat(text_mask, dim=0)

        # with torch.no_grad():
        #     text_features_all = self.text_encoder(text_emb, attention_mask=text_mask)
        #     text_features = text_features_all.pooler_output # B, tC

        

        dino_outputs = self.dino(images.tensor, captions=texts)
        # features = self.backbone(images.tensor)
        features = {}
        for i,r in enumerate(['res2', 'res3', 'res4', 'res5']):
            features[r] = self.feature_proj[i](dino_outputs['backbone_features'][i].decompose()[0])
        
        BT = len(images)
        T = self.num_frames if self.training else BT 
        B = BT // T
        frame_queries = torch.stack(dino_outputs['hs'][-3:], dim=0)
        encoded_sentence = dino_outputs['encoded_sentence']
        
        del dino_outputs
        
        outputs, _, mask_features = self.sem_seg_head(features) # 30, 256, 96, 96
        
        _, _, _, C = frame_queries.shape
        top_indices = torch.topk(frame_queries.max(-1)[0], 100, dim=2)[1]
        frame_queries = torch.gather(frame_queries, 2, top_indices.unsqueeze(-1).repeat(1,1,1,C))
        L, BT, fQ, C = frame_queries.shape
        
        del features

        mask_features = self.vita_module.vita_mask_features(mask_features)
        mask_features = mask_features.view(B, self.num_frames, *mask_features.shape[-3:])

        # mask classification target
        frame_targets, video_targets_full = self.prepare_targets(batched_inputs, images)

        # bipartite matching-based loss
        losses, fg_indices = self.criterion(outputs, frame_targets)

        if self.freeze_detector:
            losses = dict()
        
        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        num_clips = T // self.len_clip_window

        frame_targets = self.split_frame_targets(frame_targets, B)
        video_targets = self.split_video_targets(video_targets_full)
        fg_indices    = self.split_fg_indices(fg_indices, B)

        frame_queries = frame_queries.reshape(L, B, T, fQ, C)
        frame_queries = frame_queries.split(self.len_clip_window, dim=2)
        mask_features = mask_features.split(self.len_clip_window, dim=1)
        
        pre_memory = {"k": [], "v": []}

        prev_clip_indices = None
        prev_aux_clip_indices = None
        output_q = self.vita_module.query_feat.weight.unsqueeze(1).repeat(1, L*B, 1) # cQ, LB, C
        cQ, LB, C = output_q.shape
        # text_q = self.text_proj(text_features)[None].repeat(1, L, 1) # 1, LB, C
        # text_q = text_features[None] # 1, 1, tC
        clip_queries_list = []
        
        for c_i in range(num_clips):
            clip_targets  = video_targets[c_i]
            frame_targets_per_clip = frame_targets[c_i]
            frame_queries_per_clip = frame_queries[c_i]
            mask_features_per_clip = mask_features[c_i]
            fg_indices_per_clip = fg_indices[c_i]

            vita_outputs, output_q = self.vita_module(frame_queries_per_clip.flatten(1,2), pre_memory, output_q)
            clip_queries_list.append(output_q.reshape(cQ, L, B, C)[:, -1:, :, :])
            vita_outputs["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", vita_outputs["pred_mask_embed"], mask_features_per_clip)
            
            for out in vita_outputs["aux_outputs"]:
                out["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", out["pred_mask_embed"], mask_features_per_clip)
                
            genvis_loss_dict, out_clip_indices, aux_clip_indices_list = self.genvis_criterion(
                                                                            outputs=vita_outputs, 
                                                                            clip_targets=clip_targets, 
                                                                            frame_targets=frame_targets_per_clip, 
                                                                            frame_indices=fg_indices_per_clip, 
                                                                            prev_clip_indices=prev_clip_indices, 
                                                                            prev_aux_clip_indices=prev_aux_clip_indices,
                                                                        )
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

            # update clip indices
            prev_clip_indices = out_clip_indices
            prev_aux_clip_indices = aux_clip_indices_list
        
        clip_queries = torch.cat(clip_queries_list, dim=1)
        cQ, cN, B, C = clip_queries.shape
        clip_queries = clip_queries.reshape(cQ*cN, B, C)
        sentence_emb = encoded_sentence.reshape(B,T,-1)
        output, fused_sentence_emb = self.text_decoder(clip_queries, sentence_emb)
        
        mask_features_video = torch.stack(mask_features) # cN, B, T, C, H, W
        pred_mask_embed = output["pred_mask_embed"].reshape(B,C) # 1, B, C
        pred_masks_fused = torch.einsum("bc,nbtchw->nbthw", pred_mask_embed, mask_features_video) #  cN, B, T, H, W

        fusion_loss_dict = self.genvis_criterion.loss_fusion(pred_masks_fused, video_targets_full, B)
        for k in fusion_loss_dict.keys():
            if k in genvis_weight_dict:
                fusion_loss_dict[k] *= genvis_weight_dict[k]
        losses.update(fusion_loss_dict)
        
        # output_q_proj = self.vis2text(output_q) # LB, tC
        # text_outputs, text_q = self.text_decoder(output_q_proj, text_q)
        # pred_mask_embed_fused = text_outputs["pred_mask_embed"].reshape(L,B,C) 
        
        # vita_outputs['pred_masks_fused'] = torch.einsum("lbc,btchw->lbthw", pred_mask_embed_fused, mask_features_per_clip) # L, B, T, H, W
        return losses

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
        to_store = self.device
        
        mask_cls, mask_embed = [], []
        pre_memory = {"k": [], "v": []}

        output_q = self.vita_module.query_feat.weight.unsqueeze(1).repeat(1, 1, 1) # cQ, LB, C, note L=1 B=1
        cQ, LB, C = output_q.shape


        pred_masks = []
        clip_queries_list = []
        mask_features_list = []
        for i in range(math.ceil(num_frames / self.len_clip_window)):
            images = batched_inputs["image"][i*self.len_clip_window : (i+1)*self.len_clip_window]
            images = [(x.to(self.device) - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            
            texts = []
            sentence = batched_inputs["expressions"]
            sentence = sentence + '.' if not sentence.endswith('.') else sentence
            texts.extend([sentence]*images.tensor.shape[0])

            dino_outputs = self.dino(images.tensor, captions=texts)
            # features = self.backbone(images.tensor)
            features = {}
            for i,r in enumerate(['res2', 'res3', 'res4', 'res5']):
                features[r] = self.feature_proj[i](dino_outputs['backbone_features'][i].decompose()[0])
            # features = self.backbone(images.tensor)
            
            encoded_sentence = dino_outputs['encoded_sentence']
            frame_queries = torch.stack(dino_outputs['hs'][-3:], dim=0) # last 3 layer from gdino
            
            outputs, _, _mask_features = self.sem_seg_head(features)
            _, _, _, C = frame_queries.shape
            top_indices = torch.topk(frame_queries.max(-1)[0], 100, dim=2)[1]
            frame_queries = torch.gather(frame_queries, 2, top_indices.unsqueeze(-1).repeat(1,1,1,C))
            L, BT, fQ, C = frame_queries.shape
            del features
            
            
            vita_outputs, output_q = self.vita_module(frame_queries, pre_memory, output_q)
            
            clip_queries_list.append(output_q.reshape(cQ, -1, 1, C)[:, -1:, :, :])
            
            # BT is 1 as runs per frame
            _mask_features = self.vita_module.vita_mask_features(_mask_features)
            mask_features_list.append(_mask_features.unsqueeze(1))
            # mask_features.append(_mask_features)  # T', C, H, W
            
            # mask_cls.append(vita_outputs["pred_logits"][-1])       # 1, cQ, K+1
            # mask_embed.append(vita_outputs["pred_mask_embed"][-1]) # 1, cQ, C

            # update memory
            pre_memory["k"].append(vita_outputs["pre_memory"]["k"])
            pre_memory["v"].append(vita_outputs["pre_memory"]["v"])
            # pred_masks.append(pred_masks_fused)
            
            del vita_outputs

        interim_size = images.tensor.shape[-2:]
        image_size   = images.image_sizes[0]  # image size without padding after data augmentation

        out_height = batched_inputs.get("height", image_size[0])  # raw image size before data augmentation
        out_width  = batched_inputs.get("width", image_size[1])

        del outputs, images, batched_inputs
        
        mask_features_video = torch.cat(mask_features_list, dim=0) # T, B, C, H, W
        T, B, C, H, W = mask_features_video.shape
        
        clip_queries = torch.cat(clip_queries_list, dim=1)
        cQ, cN, B, C = clip_queries.shape
        clip_queries = clip_queries.reshape(cQ*cN, B, C)
        sentence_emb = dino_outputs['encoded_sentence'][0][None, None].repeat(1,T,1)
        output, fused_sentence_emb = self.text_decoder(clip_queries, sentence_emb)
        

        pred_mask_embed = output["pred_mask_embed"][0] # B, C = 1, C
        pred_masks_fused = torch.einsum("bc,tbchw->bthw", pred_mask_embed, mask_features_video) #  B, T, H, W
        pred_masks_fused = pred_masks_fused.reshape(1, T, H, W) # B = 1

        # cN,B,T,H,W = pred_masks_fused.shape
        # mask_cls   = torch.cat(mask_cls)   # NUM_CLIP, cQ, K+1
        # mask_embed = torch.cat(mask_embed) # NUM_CLIP, cQ, C

        # labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # num_topk = self.test_topk_per_image

        # scores = F.softmax(mask_cls, dim=-1)[:, :, :-1] # NUM_CLIP, cQ, K
        # scores_per_video, _ = scores.max(dim=0)
        # scores_per_video, topk_indices = scores_per_video.flatten().topk(num_topk, sorted=False)

        # labels_per_video = labels[topk_indices]
        # topk_indices = torch.div(topk_indices, self.sem_seg_head.num_classes, rounding_mode='floor')

        # mask_embed = mask_embed[:, topk_indices]

        # masks_per_video = []
        # numerator   = torch.zeros(len(topk_indices), dtype=torch.float, device=self.device)
        # denominator = torch.zeros(len(topk_indices), dtype=torch.float, device=self.device)

        mask_pred = pred_masks_fused
        
        # upsample masks
        mask_pred = retry_if_cuda_oom(F.interpolate)(
            mask_pred,
            size=interim_size,
            mode="bilinear",
            align_corners=False,
        ) # L, T, H, W
        
        mask_pred = mask_pred[:, :, : image_size[0], : image_size[1]]
        mask_pred = F.interpolate(
            mask_pred, size=(out_height, out_width), mode="bilinear", align_corners=False
        ) > 0.
        
        
        # for i in range(math.ceil(num_frames/self.len_clip_window)):
        #     mask_pred = torch.einsum("qc,tchw->qthw", mask_embed[i], mask_features[i])

        #     # upsample masks
        #     mask_pred = retry_if_cuda_oom(F.interpolate)(
        #         mask_pred,
        #         size=interim_size,
        #         mode="bilinear",
        #         align_corners=False,
        #     ) # cQ, T, H, W

        #     mask_pred = mask_pred[:, :, : image_size[0], : image_size[1]]

        #     interim_mask_soft = mask_pred.sigmoid()
        #     interim_mask_hard = interim_mask_soft > 0.5

        #     numerator   += (interim_mask_soft.flatten(1) * interim_mask_hard.flatten(1)).sum(1)
        #     denominator += interim_mask_hard.flatten(1).sum(1)

        #     mask_pred = F.interpolate(
        #         mask_pred, size=(out_height, out_width), mode="bilinear", align_corners=False
        #     ) > 0.

            # masks_per_video.append(mask_pred.to(to_store))

        # masks_per_video   = torch.cat(masks_per_video, dim=1).cpu()
        # scores_per_video *= (numerator / (denominator + 1e-6))

        processed_results = {
            "image_size": (out_height, out_width),
            # "pred_scores": scores_per_video.tolist(),
            # "pred_labels": labels_per_video.tolist(),
            # "pred_masks": mask_pred.cpu(),
            "pred_masks": mask_pred,
        }

        return processed_results
    
class TextDecoder(nn.Module):
    def __init__(
        self, 
        nheads: int,
        dim_feedforward: int,
        enc_layers: int,
        dec_layers: int,
        hidden_dim: int, # text dim
        mask_dim: int,
        num_frames: int,
        ):
        """
        
        TextDecoder    
        
        """
        
        super().__init__()
        self.num_heads = nheads
        self.enc_layers = enc_layers
        self.num_layers = dec_layers
        # self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.src_embed = nn.Identity()
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.num_frames = num_frames
        
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
            # self.transformer_self_attention_layers.append(
            #     SelfAttentionLayer(
            #         d_model=hidden_dim,
            #         nhead=nheads,
            #         dropout=0.0,
            #         normalize_before=pre_norm,
            #     )
            # )

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

        # Not using window-based attention if self.window_size == 0.

        # return_shape = frame_query.shape        # cQ, LB, C


        for i in range(self.enc_layers):
            clip_q = self.enc_self_attn[i](clip_q)
            clip_q = self.enc_ffn[i](clip_q)

        # clip_q = frame_query.view(return_shape)
        return clip_q
        
    def forward(self, clip_q, text_q):
        # clip_q : cQcN, B, C
        # text_q : B, T, C 
        # if not self.training:
            # clip_q = clip_q[[-1]]
        
        cN, cQB, C = clip_q.shape
        B, T, C = text_q.shape
        cQ = cQB // B
        
        clip_q = self.encode_clip_query(clip_q) # cQcN, B, C
        src = self.src_embed(clip_q) # cQcN, B, C
        output = text_q[:,-1,:].reshape(1, B, C) # 1, B, C
        # decoder_outputs = []
        for i in range(self.num_layers):
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=None, query_pos=None
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            output = self.decoder_norm(output)
            
        pred_mask_embed = self.mask_embed(output)
        
        out = {
            'pred_mask_embed' : pred_mask_embed
        }
        
        
        return out, output
class GroundingDinoWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = load_model("/workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
        self._features = {}
        # layer = dict([*self.model.named_modules()])[layer_id]
        # layer.register_forward_hook(self.save_outputs_hook)
    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn
    
    def forward(self, images, texts):
        full_output = self.model(images, captions=texts)
        self._features['full_output'] = full_output
        return self._features