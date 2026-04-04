import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.stats import norm

import json
import logging
import random


def showHorizonLine(
    image,
    vfov,
    pitch,
    roll,
    color=(0, 255, 0),
    width=5,
):
    """
    Angles should be in radians.
    """
    h, w, _ = image.shape
    if image.dtype in (np.float32, np.float64):
        image = (image * 255).astype("uint8")

    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)
    ctr = h * (0.5 - 0.5 * np.tan(pitch) / np.tan(vfov / 2))
    l = ctr - w * np.tan(roll) / 2
    r = ctr + w * np.tan(roll) / 2
    draw.line((0, l, w, r), fill=color, width=width)
    return np.array(im), ctr / h


def showHorizonLineFromHorizon(
    image,
    horizon,
    color=(0, 255, 0),
    width=5,
    debug=False,
    GT=False,
):
    """
    Angles should be in radians.
    """
    h, w, _ = image.shape
    if image.dtype in (np.float32, np.float64):
        image = (image * 255).astype("uint8")

    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)

    l = horizon * h
    r = horizon * h
    if debug:
        if not GT:
            draw.text((0, 12), f"v0:{horizon:.2f}", (255, 255, 255))
        else:
            draw.text((0, h - 24), f"GT: v0:{horizon:.2f}", (255, 255, 255))

    draw.line((0, l, w, r), fill=color, width=width)
    return np.array(im)


def getBins(minval, maxval, sigma, alpha, beta, kappa):
    """Remember, bin 0 = below value! last bin mean >= maxval"""
    x = np.linspace(minval, maxval, 255)

    rv = norm(0, sigma)
    pdf = rv.pdf(x)
    pdf /= pdf.max()
    pdf *= alpha
    pdf = pdf.max() * beta - pdf
    cumsum = np.cumsum(pdf)
    cumsum = cumsum / cumsum.max() * kappa
    cumsum -= cumsum[pdf.size // 2]

    return cumsum


pitch_bins_low = np.linspace(-np.pi / 2 + 1e-5, -5 * np.pi / 180.0, 31)
pitch_bins_high = np.linspace(5 * np.pi / 180.0, np.pi / 6, 31)
# crops_dataset_cvpr_myDistWider20200403:
pitch_bins = np.linspace(-0.6, 0.6, 255)
pitch_bins_centers = pitch_bins.copy()
pitch_bins_centers[:-1] += np.diff(pitch_bins_centers) / 2
pitch_bins_centers = np.append(pitch_bins_centers, pitch_bins[-1])

horizon_bins = np.linspace(-1.0, 0.95, 255)
horizon_bins_centers = horizon_bins.copy()
horizon_bins_centers[:-1] += np.diff(horizon_bins_centers) / 2
horizon_bins_centers = np.append(horizon_bins_centers, horizon_bins[-1])

roll_bins = getBins(-np.pi / 6, np.pi / 6, 0.5, 0.04, 1.1, np.pi)
roll_bins_centers = roll_bins.copy()
roll_bins_centers[:-1] += np.diff(roll_bins_centers) / 2
roll_bins_centers = np.append(roll_bins_centers, roll_bins[-1])

vfov_bins = np.linspace(0.2389, 1.6, 255)
vfov_bins_centers = vfov_bins.copy()
vfov_bins_centers[:-1] += np.diff(vfov_bins_centers) / 2
vfov_bins_centers = np.append(vfov_bins_centers, vfov_bins[-1])


def bins2horizon(bins):
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return horizon_bins_centers[idxes]


def bins2pitch(bins):
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return pitch_bins_centers[idxes]


def bins2roll(bins):
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return roll_bins_centers[idxes]


def bins2vfov(bins):
    idxes = np.argmax(bins, axis=bins.ndim - 1)
    return vfov_bins_centers[idxes]


def getHorizonLine(vfov, pitch):
    """
    Angles should be in radians.
    """
    ctr = 0.5 - 0.5 * np.tan(pitch) / np.tan(vfov / 2)
    return ctr


class CalibDataset:
    def __init__(
        self,
        train: bool = True,
        logger: logging.Logger | None = None,
        json_name: str = "datasets/train_crops_dataset_cvpr_myDistWider.json",
        debug: bool = False,
    ):
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        with open(json_name) as fhdl:
            self.data = json.load(fhdl)

        max_load = -1 if not debug else 100
        self.data = self.data[:max_load]  # Only use 100 examples
        random.shuffle(self.data)
        train_load = -2000 if not debug else -50
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
        focal_length_35mm_eq = data["focal_length_35mm_eq"]
        horizon = getHorizonLine(vfov, pitch)
        horizon_idx = np.digitize(horizon, horizon_bins)
        pitch_idx = np.digitize(pitch, pitch_bins)
        roll_idx = np.digitize(roll, roll_bins)
        vfov_idx = np.digitize(vfov, vfov_bins)
        horizon_gt = np.zeros((256,), dtype=np.float32)
        pitch_gt = np.zeros((256,), dtype=np.float32)
        roll_gt = np.zeros((256,), dtype=np.float32)
        vfov_gt = np.zeros((256,), dtype=np.float32)
        horizon_gt[horizon_idx] = 1.0
        pitch_gt[pitch_idx] = 1.0
        roll_gt[roll_idx] = 1.0
        vfov_gt[vfov_idx] = 1.0
        horizon_gt, pitch_gt, roll_gt, vfov_gt = map(
            torch.from_numpy,
            (horizon_gt, pitch_gt, roll_gt, vfov_gt),
        )
        return dict(
            source="pano360",
            file_name=im_path,
            image_id=im_path,
            pitch=pitch,
            roll=roll,
            annotations=[],
            horizon=horizon,
            vfov=vfov,
            focal_length_35mm_eq=focal_length_35mm_eq,
            logits=dict(
                gt_horizon=horizon_gt,
                gt_pitch=pitch_gt,
                gt_roll=roll_gt,
                gt_vfov=vfov_gt,
            ),
        )

    def get_all_items(self):
        for i, _ in enumerate(self.data):
            yield self[i]

    def __call__(self):
        return self

    def __len__(self):
        return len(self.data)
