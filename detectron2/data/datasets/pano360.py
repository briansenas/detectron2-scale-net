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


def make_bins_layers_list(x_bins_lowHigh_list):
    x_bins_layers_list = []
    for _, x_bins_lowHigh in enumerate(x_bins_lowHigh_list):
        x_bins = np.linspace(x_bins_lowHigh[0], x_bins_lowHigh[1], 255)
        x_bins_centers = x_bins.copy()
        x_bins_centers[:-1] += np.diff(x_bins_centers) / 2
        x_bins_centers = np.append(x_bins_centers, x_bins_centers[-1])  # 42 bins
        x_bins_layers_list.append(x_bins_centers)
    return x_bins_layers_list


pitch_bins_low = np.linspace(-np.pi / 2 + 1e-5, -5 * np.pi / 180.0, 31)
pitch_bins_high = np.linspace(5 * np.pi / 180.0, np.pi / 6, 31)
# crops_dataset_cvpr_myDistWider20200403:
pitch_bins = np.linspace(-0.6, 0.6, 255)
pitch_bins_centers = pitch_bins.copy()
pitch_bins_centers[:-1] += np.diff(pitch_bins_centers) / 2
pitch_bins_centers = np.append(pitch_bins_centers, pitch_bins[-1])

horizon_bins = np.linspace(-0.5, 1.5, 255)
horizon_bins_centers = horizon_bins.copy()
horizon_bins_centers[:-1] += np.diff(horizon_bins_centers) / 2
horizon_bins_centers = np.append(horizon_bins_centers, horizon_bins[-1])

roll_bins = getBins(-np.pi / 6, np.pi / 6, 0.5, 0.04, 1.1, np.pi)
roll_bins_centers = roll_bins.copy()
roll_bins_centers[:-1] += np.diff(roll_bins_centers) / 2
roll_bins_centers = np.append(roll_bins_centers, roll_bins[-1])

vfov_bins = np.linspace(0.2617, 2.1, 255)
vfov_bins_centers = vfov_bins.copy()
vfov_bins_centers[:-1] += np.diff(vfov_bins_centers) / 2
vfov_bins_centers = np.append(vfov_bins_centers, vfov_bins[-1])

yc_bins_lowHigh_list = [[0.5, 5.], [-0.3, 0.3], [-0.15, 0.15], [-0.3, 0.3], [-0.15, 0.15]]  # 'YcLargeBins'
yc_bins_layers_list = make_bins_layers_list(yc_bins_lowHigh_list)
yc_bins_centers = yc_bins_layers_list[0]


fmm_bins_lowHigh_list = [[0., 0.], [-0.2, 0.2], [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05]]  # percentage!!
fmm_bins_layers_list = make_bins_layers_list(fmm_bins_lowHigh_list)


v0_bins_lowHigh_list = [[0., 0.], [-0.15, 0.15], [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05]]  # 'SmallerBins'
v0_bins_layers_list = make_bins_layers_list(v0_bins_lowHigh_list)

# human_bins = np.linspace(1., 2., 256)
human_bins = np.linspace(1., 1.9, 256)  # 'SmallerPersonBins'
# human_bins = np.linspace(1., 2.5, 256) #  'V2PersonCenBins'
# human_bins = np.linspace(0.7, 1.9, 256) #  'V3PersonCenBins'
# human_bins_1 = np.linspace(-0.2, 0.2, 256)
human_bins_lowHigh_list = [[0., 0.], [-0.3, 0.15], [-0.10, 0.10], [-0.10, 0.10], [-0.05, 0.05]]  # 'SmallerBins'
human_bins_layers_list = make_bins_layers_list(human_bins_lowHigh_list)

car_bins = np.linspace(1.4, 1.70, 256)  # 'V2CarBins'
car_bins_lowHigh_list = [[0., 0.], [-0.10, 0.10], [-0.05, 0.05], [-0.10, 0.10], [-0.05, 0.05]]  # 'SmallerBins'
car_bins_layers_list = make_bins_layers_list(car_bins_lowHigh_list)

COCO_SCALE_STATS = [
    {"height_mean": 1.75, "height_std": 0.2, "id": 0, "name": "person",
        "bins": human_bins, "layer_list": human_bins_layers_list},
    {"height_mean": 1.59, "height_std": 0.2, "id": 2, "name": "car", "bins": car_bins, "layer_list": car_bins_layers_list},
]


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
        debug_train_size: int = 1000,
        debug_eval_size: int = 100,
        loss_criterion: str = "kl",
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
        self.loss_type = loss_criterion
        valid_criterions = ('kl', 'ce', 'softargmax_l2', 'softargmax_l2_biased')
        if loss_criterion not in valid_criterions:
            msg = f"The criterion is invalid. Got {loss_criterion} not in {valid_criterions}"
            raise ValueError(msg)

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
                gt_horizon=horizon_idx,
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


# Based on PARE: https://github.com/mkocabas/PARE/tree/master
def _softmax(tensor, temperature, dim=-1):
    return torch.nn.functional.softmax(tensor * temperature, dim=dim)


def softargmax1d(
        heatmaps,
        temperature=None,
        normalize_keypoints=True,
):
    dtype, device = heatmaps.dtype, heatmaps.device
    if temperature is None:
        temperature = torch.tensor(1.0, dtype=dtype, device=device)
    batch_size, num_channels, dim = heatmaps.shape
    points = torch.arange(0, dim, device=device, dtype=dtype).reshape(1, 1, dim).expand(batch_size, -1, -1)
    # y = torch.arange(0, height, device=device, dtype=dtype).reshape(1, 1, height, 1).expand(batch_size, -1, -1, width)
    # Should be Bx2xHxW

    # points = torch.cat([x, y], dim=1)
    normalized_heatmap = _softmax(
        heatmaps.reshape(batch_size, num_channels, -1),
        temperature=temperature.reshape(1, -1, 1),
        dim=-1)

    # Should be BxJx2
    keypoints = (normalized_heatmap.reshape(batch_size, -1, dim) * points).sum(dim=-1)

    if normalize_keypoints:
        # Normalize keypoints to [-1, 1]
        keypoints = (keypoints / (dim - 1) * 2 - 1)

    return keypoints, normalized_heatmap.reshape(
        batch_size, -1, dim)


def angle_to_soft_idx(angle, min, max):
    return 2 * ((angle - min) / (max - min)) - 1


def vfov2soft_idx(angle):
    return angle_to_soft_idx(angle, min=np.min(vfov_bins), max=np.max(vfov_bins))


def pitch2soft_idx(angle):
    return angle_to_soft_idx(angle, min=np.min(pitch_bins), max=np.max(pitch_bins))


def roll2soft_idx(angle):
    return angle_to_soft_idx(angle, min=-0.6, max=0.6)


def horizon2soft_idx(angle):
    return angle_to_soft_idx(angle, min=np.min(horizon_bins), max=np.max(horizon_bins))


def soft_idx_to_angle(soft_idx, min, max):
    return (max - min) * ((soft_idx + 1) / 2) + min


def get_softargmax(pred):
    pred = pred.unsqueeze(1)  # (N, 1, 256)
    pred_argmax, _ = softargmax1d(pred, normalize_keypoints=True)  # (N, 1, 1)
    pred_argmax = pred_argmax.reshape(-1)
    return pred_argmax


def convert_soft_idx_to_angle(vfov, pitch, roll):
    vfov_angle = soft_idx_to_angle(vfov, min=np.min(vfov_bins), max=np.max(vfov_bins))
    pitch_angle = soft_idx_to_angle(pitch, min=np.min(pitch_bins), max=np.max(pitch_bins))
    roll_angle = soft_idx_to_angle(roll, min=-0.6, max=0.6)
    return vfov_angle, pitch_angle, roll_angle


@torch.no_grad()
def convert_preds_to_angles(pred_vfov, pred_pitch, pred_roll, loss_type='kl', return_type='torch', legacy=False):
    if loss_type in ('kl', 'ce'):
        pred_vfov = bins2vfov(pred_vfov)
        pred_pitch = bins2pitch(pred_pitch)
        pred_roll = bins2roll(pred_roll)
    elif loss_type in ('softargmax_l2', 'softargmax_l2_biased'):
        pred_vfov = soft_idx_to_angle(get_softargmax(pred_vfov),
                                      min=np.min(vfov_bins), max=np.max(vfov_bins))
        pred_pitch = soft_idx_to_angle(get_softargmax(pred_pitch),
                                       min=np.min(pitch_bins), max=np.max(pitch_bins))
        if not legacy:
            pred_roll = soft_idx_to_angle(get_softargmax(pred_roll), min=-0.6, max=0.6)
        else:
            pred_roll = bins2roll(pred_roll)

    if return_type == 'np' and isinstance(pred_vfov, torch.Tensor):
        return pred_vfov.cpu().numpy(), \
            pred_pitch.cpu().numpy(), \
            pred_roll.cpu().numpy()

    if return_type == 'torch' and isinstance(pred_vfov, np.ndarray):
        return torch.from_numpy(pred_vfov), torch.from_numpy(pred_pitch), torch.from_numpy(pred_roll)

    return pred_vfov, pred_pitch, pred_roll
