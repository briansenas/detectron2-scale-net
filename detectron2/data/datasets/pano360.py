import numpy as np

import json
import logging
import random

from ..pano360_utils import pitch_bins, roll_bins, vfov_bins


class CalibDataset:
    def __init__(
        self,
        train: bool = True,
        logger: logging.Logger | None = None,
        json_name: str = "datasets/train_crops_dataset_cvpr_myDistWider.json",
        debug: bool = False,
        debug_train_size: int = 1000,
        debug_eval_size: int = 100,
    ):
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        with open(json_name) as fhdl:
            self.data = json.load(fhdl)

        max_load = -1 if not debug else debug_train_size
        self.data = self.data[:max_load]  # Only use 100 examples
        random.shuffle(self.data)
        train_load = -5000 if not debug else -debug_eval_size
        if train:
            self.data = self.data[:train_load]
        else:
            self.data = self.data[train_load:]

    def __getitem__(self, k):
        with open(self.data[k][:-4] + ".json") as fhdl:
            data = json.load(fhdl)
        im_path = self.data[k]
        data = data[0]
        pitch = data["pitch"]  # in radians
        roll = data["roll"]
        vfov = data["vfov"]
        pitch_idx = np.digitize(pitch, pitch_bins)
        roll_idx = np.digitize(roll, roll_bins)
        vfov_idx = np.digitize(vfov, vfov_bins)

        return dict(
            source="pano360",
            file_name=im_path,
            image_id=im_path,
            pitch=pitch,
            roll=roll,
            annotations=[],
            vfov=vfov,
            logits=dict(
                gt_pitch=pitch_idx,
                gt_roll=roll_idx,
                gt_vfov=vfov_idx,
            ),
        )

    def get_all_items(self):
        for i, _ in enumerate(self.data):
            yield self[i]

    def __call__(self):
        return self

    def __len__(self):
        return len(self.data)
