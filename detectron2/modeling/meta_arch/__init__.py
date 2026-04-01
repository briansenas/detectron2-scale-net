# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from .build import build_model
from .build import META_ARCH_REGISTRY
from .camcnn import ClassifierRCNN
from .dense_detector import DenseDetector
from .fcos import FCOS
from .panoptic_fpn import PanopticFPN
from .rcnn import GeneralizedRCNN
from .rcnn import ProposalNetwork
from .retinanet import RetinaNet
from .semantic_seg import build_sem_seg_head
from .semantic_seg import SEM_SEG_HEADS_REGISTRY
from .semantic_seg import SemanticSegmentor

# import all the meta_arch, so they will be registered


__all__ = list(globals().keys())
