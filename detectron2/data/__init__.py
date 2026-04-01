# Copyright (c) Facebook, Inc. and its affiliates.
from . import datasets
from . import samplers
from . import transforms  # isort:skip
from .build import build_batch_data_loader
from .build import build_detection_test_loader
from .build import build_detection_train_loader
from .build import get_detection_dataset_dicts
from .build import load_proposals_into_dataset
from .build import print_instances_class_histogram
from .catalog import DatasetCatalog
from .catalog import Metadata
from .catalog import MetadataCatalog
from .common import DatasetFromList
from .common import MapDataset
from .common import ToIterableDataset
from .dataset_mapper import CalibMapper
from .dataset_mapper import DatasetMapper

# ensure the builtin datasets are registered

__all__ = [k for k in globals().keys() if not k.startswith("_")]
