import numpy as np

import joblib
import json
import os
import random


class CalibDataset:
    def __init__(
        self,
        train: bool = True,
        json_name: str = "datasets/train_crops_dataset_cvpr_myDistWider.json",
        debug: bool = False,
        debug_train_size: int = 1000,
        debug_eval_size: int = 100,
    ):
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

        return dict(
            source="pano360",
            file_name=im_path,
            image_id=im_path,
            pitch=pitch,
            roll=roll,
            annotations=[],
            vfov=vfov,
        )

    def get_all_items(self):
        for i, _ in enumerate(self.data):
            yield self[i]

    def __call__(self):
        return self

    def __len__(self):
        return len(self.data)

# Based on the implementation of SPEC


class CameraRegressorDataset:
    def __init__(
            self,
            dataset='data/pano360/preprocessed_pano_dataset',
            is_train=True,
            loss_type='kl',
            num_images=-1,
            debug: bool = False,
            debug_train_size: int = 1000,
            debug_eval_size: int = 100,
    ):
        self.dataset_folder = dataset
        self.loss_type = loss_type

        if is_train:
            self.image_filenames = joblib.load(os.path.join(self.dataset_folder, 'train_images.pkl'))
            if debug:
                self.image_filenames = self.image_filenames[:debug_train_size]
        else:
            self.image_filenames = joblib.load(os.path.join(self.dataset_folder, 'val_images.pkl'))
            if debug:
                self.image_filenames = self.image_filenames[:debug_eval_size]
        if num_images > 0:
            self.image_filenames = np.random.choice(self.image_filenames, num_images)

    def __len__(self):
        return len(self.image_filenames)

    def get_all_items(self):
        for i, _ in enumerate(self.image_filenames):
            yield self[i]

    def __call__(self):
        return self

    def __getitem__(self, index):
        imgname = os.path.join(self.dataset_folder, 'images', self.image_filenames[index])
        data = json.load(open(imgname.replace('images', 'annotations').replace('.png', '.json')))

        pitch = data['pitch']  # in radians
        roll = data['roll']  # in radians
        vfov = np.radians(data['vfov'])  # in radians

        return dict(
            source="pano360",
            file_name=imgname,
            image_id=self.image_filenames[index],
            pitch=pitch,
            roll=roll,
            annotations=[],
            vfov=vfov,
        )
