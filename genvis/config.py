# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_genvis_config(cfg):
    cfg.MODEL.GENVIS = CN()
    cfg.MODEL.GENVIS.LEN_CLIP_WINDOW = 5
    cfg.MODEL.GENVIS.TEXT_HIDDEN_DIM = 768
    cfg.MODEL.GENVIS.FREEZE_TEXT_ENCODER = True
    cfg.MODEL.GENVIS.TEXT_WEIGHT = 1.0