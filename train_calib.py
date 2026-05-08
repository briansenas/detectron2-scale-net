import torch

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.coco_scale import COCOScale2017, COCOScale2017Calib
from detectron2.data.datasets.pano360 import CalibDataset
from detectron2.engine import (
    CalibTrainer,
    COCOScaleTrainer,
    HybridScaleTrainer,
    default_argument_parser,
    default_setup,
    launch,
)

import os
from pathlib import Path

torch.autograd.set_detect_anomaly(True)
# torch.multiprocessing.set_sharing_strategy("file_system")
torch.multiprocessing.set_sharing_strategy("file_descriptor")

# Better CUDA memory usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True"
)

FILE_PATH = Path(__file__)
PANO_TRAIN_NAME = "Pano360_train"
PANO_VAL_NAME = "Pano360_val"
COCO_SCALE_DATASET_NAME = "COCOScale2017_train"
COCO_SCALE_CALIB_DATASET_NAME = "COCOScale2017Calib_train"


def register_datasets(keypoint_on: bool = False, debug: bool = True):
    calib_train = CalibDataset(
        train=True,
        json_name="datasets/pano360_crops_dataset_cvpr_myDistWider_train.json",
        logger=None,
        debug=debug,
    )
    calib_val = CalibDataset(
        train=False,
        json_name="datasets/pano360_crops_dataset_cvpr_myDistWider_train.json",
        logger=None,
        debug=debug,
    )
    DatasetCatalog.register(PANO_TRAIN_NAME, calib_train)
    DatasetCatalog.register(PANO_VAL_NAME, calib_val)

    # Base path for coco annotations.
    base_path = Path.cwd()
    coco_path = base_path / "data" / "coco"
    coco_annotations_path = coco_path / "annotations"
    coco_keypoints_path = coco_annotations_path / "person_keypoints_train2017.json"
    coco_scalenet_results_path = coco_path / "coco_results"
    coco_images_root_path = coco_path / "train2017"

    coco_scale_train = COCOScale2017(
        debug=debug,
        camera_parameters_file_path=coco_scalenet_results_path /
        "yannick_results_train2017_filtered",
        coco_json_file_path=coco_keypoints_path,
        coco_image_root_path=coco_images_root_path,
        coco_scale_pickle_path=coco_scalenet_results_path /
        "results_with_kps_20200208_morethan2_2-8" /
        "pickle",
    )

    coco_meta = _get_builtin_metadata("coco_person")
    DatasetCatalog.register(COCO_SCALE_DATASET_NAME, coco_scale_train)
    MetadataCatalog.get(COCO_SCALE_DATASET_NAME).set(
        json_file=coco_keypoints_path,
        image_root=coco_images_root_path,
        evaluator_type="coco",
        **coco_meta,
    )
    # This is need due to internal consistency checks of d2 for keypoints_on
    MetadataCatalog.get(PANO_TRAIN_NAME).set(
        json_file=coco_keypoints_path,
        image_root=coco_images_root_path,
        evaluator_type="coco",
        **coco_meta,
        thing_dataset_id_to_contiguous_id={1: 0},  # COCO ID 1 → internal ID 0
    )

    coco_scale_calib_dataset = COCOScale2017Calib(
        calib_train,
        get_detection_dataset_dicts(
            COCO_SCALE_DATASET_NAME, True, 2 if keypoint_on else 0, None, check_consistency=True
        ),
    )
    DatasetCatalog.register(COCO_SCALE_CALIB_DATASET_NAME, coco_scale_calib_dataset)
    MetadataCatalog.get(COCO_SCALE_CALIB_DATASET_NAME).set(
        json_file=coco_keypoints_path,
        image_root=coco_images_root_path,
        evaluator_type="coco",
        **coco_meta,
        thing_dataset_id_to_contiguous_id={1: 0},  # COCO ID 1 → internal ID 0
    )


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def build_args():

    parser = default_argument_parser()
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument("--experiment-name", type=str, default="debug-coco-scale")
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Load state dict from CFG and previous experiment filtering keys.",
    )
    parser.add_argument(
        "--resume-from-filter-name",
        type=str,
        help="Filter state dict for weights containing this key.",
    )
    return parser.parse_args()


def invoke_main():
    args = build_args()
    print("Command Line Args: ", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


def main(args):
    cfg = setup(args)
    register_datasets(cfg.MODEL.KEYPOINT_ON, args.debug)
    if len(cfg.DATASETS.TRAIN) > 1:
        raise ValueError("This is script is not intended for multiple datasets")
    if cfg.DATASETS.TRAIN[0] == PANO_TRAIN_NAME:
        trainer = CalibTrainer(cfg)
    elif cfg.DATASETS.TRAIN[0] == COCO_SCALE_DATASET_NAME:
        trainer = COCOScaleTrainer(cfg)
    elif cfg.DATASETS.TRAIN[0] == COCO_SCALE_CALIB_DATASET_NAME:
        trainer = HybridScaleTrainer(cfg)
    else:
        raise ValueError(
            "This file is not made for datasets outside %s and cfg is %s"
            % (
                {
                    PANO_TRAIN_NAME,
                    COCO_SCALE_CALIB_DATASET_NAME,
                    COCO_SCALE_CALIB_DATASET_NAME,
                },
                cfg.DATASETS.TRAIN,
            )
        )
    if args.resume_from:
        print(f"Resuming from previous experiment: {args.resume_from}")
        model = trainer.model

        def load_state_dict(exp_weights_path):
            checkpoint = torch.load(exp_weights_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            return state_dict

        calib_state_dict = load_state_dict(args.resume_from)
        coco_state_dict = load_state_dict(cfg.MODEL.WEIGHTS)
        if args.resume_from_filter_name:
            # Filter only camera_head weights
            calib_state_dict = {
                k: v
                for k, v in calib_state_dict.items()
                if args.resume_from_filter_name in k
            }
        # Merge both state dicts to have the full state dict to load. Make sure the argument is filtered.
        coco_state_dict.update(calib_state_dict)
        # Load into trainer.model
        missing, unexpected = model.load_state_dict(coco_state_dict, strict=False)

        print("Loaded camera_head weights into trainer.model")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
    else:
        trainer.resume_or_load(resume=args.resume)
    trainer.train()


if __name__ == "__main__":
    invoke_main()
