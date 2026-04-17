import numpy as np
import torch
from scipy.io import loadmat

from detectron2.data.datasets.coco import load_coco_json
from detectron2.structures import BoxMode

import logging
import pickle
import random
from pathlib import Path


class COCOScale2017:
    def __init__(
        self,
        split: str = "train",
        logger: logging.Logger | None = None,
        debug: bool = False,
        shuffle: bool = False,
        *,
        camera_parameters_file_path: str | Path,
        coco_json_file_path: str | Path,
        coco_image_root_path: str | Path,
        coco_scale_pickle_path: str | Path,
        debug_size: int = 1000,
    ):
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.coco_data = load_coco_json(coco_json_file_path, coco_image_root_path)
        self.is_train = split in ["train", "val"]

        # Estimated GT from coco files using a calibrated model?
        self.camera_parameters_files = sorted(Path(camera_parameters_file_path).glob("*.mat"), key=lambda x: str(x))
        random.shuffle(self.camera_parameters_files)
        if split == "train":
            self.camera_parameters_files = self.camera_parameters_files[: int(len(self.camera_parameters_files) * 0.8)]
            img_filenames = [
                str(camera_mat_file.name).split(".")[0]
                for camera_mat_file in self.camera_parameters_files
            ]

            self.img_files = [
                coco_image_root_path / (img_filename + ".jpg")
                for img_filename in img_filenames
            ]

            self.pickle_files = [
                coco_scale_pickle_path / (img_filename + ".data")
                for img_filename in img_filenames
            ]
        else:
            # No camera pre-information in val / test
            self.camera_parameters_files = []
            self.pickle_files = sorted(Path(coco_scale_pickle_path).glob("*.data"), key=lambda x: str(x))
            self.img_files = [
                coco_image_root_path / (str(pickle_file.stem()) + ".jpg")
                for pickle_file in self.pickle_files
            ]
        if debug:
            self.pickle_files = self.pickle_files[:debug_size]
            self.img_files = self.img_files[:debug_size]

        assert len(self.img_files) == len(self.pickle_files)
        if shuffle and self.is_train:
            list_zip = list(zip(self.img_files, self.pickle_files))
            random.shuffle(list_zip)
            self.img_files, self.pickle_files = zip(*list_zip)

    def __getitem__(self, k):
        with open(self.pickle_files[k], "rb") as fhdl:
            data = pickle.load(fhdl)
        im_path = self.img_files[k]
        bboxes = data["bboxes"].astype(np.float32)
        kps_gt = data["kps"].astype(int).tolist()
        pitch = vfov = roll = -1
        if self.is_train:
            camera_parameters = loadmat(self.camera_parameters_files[k])
            pitch = camera_parameters["pitch"][0][0].astype(np.float32)
            vfov = camera_parameters["vfov"][0][0].astype(np.float32)
            roll = camera_parameters["roll"][0][0].astype(np.float32)
            horizon = camera_parameters["horizon"][0][0].astype(np.float32)
        instances = []
        for bbox, kps in zip(bboxes, kps_gt):
            instances.append(dict(
                bbox=bbox.tolist(),
                bbox_mode=BoxMode.XYWH_ABS,
                category_id=0,
                keypoints=kps,
            ))
        return dict(
            source="coco",
            file_name=im_path,
            image_id=int(im_path.stem.lstrip("0")),
            pitch=pitch,
            vfov=vfov,
            roll=roll,
            horizon=horizon,
            annotations=instances
        )

    def get_all_items(self):
        for i, _ in enumerate(self.pickle_files):
            yield self[i]

    def __call__(self):
        return self

    def __len__(self):
        return len(self.pickle_files)


class COCOScale2017Calib(torch.utils.data.Dataset):
    def __init__(self, calib_dataset, coco_dataset, ratio: tuple[int, int] = (3, 1)):
        self.calib = calib_dataset
        self.coco = coco_dataset

        self.calib_len = len(calib_dataset)
        self.coco_len = len(coco_dataset)

        self.coco_samples_size, self.calib_sample_size = ratio

    def __len__(self):
        return self.coco_len

    def __getitem__(self, idx):
        coco_idx = [
            (idx * self.coco_samples_size + i) % self.coco_len for i in range(self.coco_samples_size)
        ]
        calib_idx = [
            (idx * self.calib_sample_size + i) % self.calib_len for i in range(self.calib_sample_size)
        ]

        coco_samples = [self.coco[i] for i in coco_idx]
        calib_samples = [self.calib[i] for i in calib_idx]

        return {
            "coco_data": coco_samples,
            "calib_data": calib_samples,
        }

    def __call__(self):
        return self
