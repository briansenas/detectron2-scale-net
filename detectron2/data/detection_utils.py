# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Common data processing utilities that are used in a
typical object detection data pipeline.
"""
import numpy as np
import pycocotools.mask as mask_util
import torch
from PIL import Image
from torch import nn

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)
from detectron2.utils.file_io import PathManager

import logging
from typing import Dict, List, Union

from . import transforms as T
from .catalog import MetadataCatalog

__all__ = [
    "SizeMismatchError",
    "convert_image_to_rgb",
    "check_image_size",
    "transform_proposals",
    "transform_instance_annotations",
    "annotations_to_instances",
    "annotations_to_instances_rotated",
    "build_augmentation",
    "build_transform_gen",
    "create_keypoint_hflip_indices",
    "filter_empty_instances",
    "read_image",
]


class SizeMismatchError(ValueError):
    """
    When loaded image has difference width/height compared with annotation.
    """


# https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]

# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag


def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image


def convert_image_to_rgb(image, format):
    """
    Convert an image from given format to RGB.

    Args:
        image (np.ndarray or Tensor): an HWC image
        format (str): the format of input image, also see `read_image`

    Returns:
        (np.ndarray): (H,W,3) RGB image in 0-255 range, can be either float or uint8
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if format == "BGR":
        image = image[:, :, [2, 1, 0]]
    elif format == "YUV-BT.601":
        image = np.dot(image, np.array(_M_YUV2RGB).T)
        image = image * 255.0
    else:
        if format == "L":
            image = image[:, :, 0]
        image = image.astype(np.uint8)
        image = np.asarray(Image.fromarray(image, mode=format).convert("RGB"))
    return image


def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

    Returns:
        image (np.ndarray):
            an HWC image in the given format, which is 0-255, uint8 for
            supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        image = _apply_exif_orientation(image)
        return convert_PIL_to_numpy(image, format)
    raise ValueError(f"Failed to read image at: {file_name}")


def check_image_size(dataset_dict, image):
    """
    Raise an error if the image does not match the size specified in the dict.
    """
    if "width" in dataset_dict or "height" in dataset_dict:
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (dataset_dict["width"], dataset_dict["height"])
        if not image_wh == expected_wh:
            raise SizeMismatchError(
                "Mismatched image shape{}, got {}, expect {}.".format(
                    (
                        " for image " + dataset_dict["file_name"]
                        if "file_name" in dataset_dict
                        else ""
                    ),
                    image_wh,
                    expected_wh,
                ) +
                " Please check the width/height in your annotation."
            )

    # To ensure bbox always remap to original image size
    if "width" not in dataset_dict:
        dataset_dict["width"] = image.shape[1]
    if "height" not in dataset_dict:
        dataset_dict["height"] = image.shape[0]


def transform_proposals(dataset_dict, image_shape, transforms, *, proposal_topk, min_box_size=0):
    """
    Apply transformations to the proposals in dataset_dict, if any.

    Args:
        dataset_dict (dict): a dict read from the dataset, possibly
            contains fields "proposal_boxes", "proposal_objectness_logits", "proposal_bbox_mode"
        image_shape (tuple): height, width
        transforms (TransformList):
        proposal_topk (int): only keep top-K scoring proposals
        min_box_size (int): proposals with either side smaller than this
            threshold are removed

    The input dict is modified in-place, with abovementioned keys removed. A new
    key "proposals" will be added. Its value is an `Instances`
    object which contains the transformed proposals in its field
    "proposal_boxes" and "objectness_logits".
    """
    if "proposal_boxes" in dataset_dict:
        # Transform proposal boxes
        boxes = transforms.apply_box(
            BoxMode.convert(
                dataset_dict.pop("proposal_boxes"),
                dataset_dict.pop("proposal_bbox_mode"),
                BoxMode.XYXY_ABS,
            )
        )
        boxes = Boxes(boxes)
        objectness_logits = torch.as_tensor(
            dataset_dict.pop("proposal_objectness_logits").astype("float32")
        )

        boxes.clip(image_shape)
        keep = boxes.nonempty(threshold=min_box_size)
        boxes = boxes[keep]
        objectness_logits = objectness_logits[keep]

        proposals = Instances(image_shape)
        proposals.proposal_boxes = boxes[:proposal_topk]
        proposals.objectness_logits = objectness_logits[:proposal_topk]
        dataset_dict["proposals"] = proposals


def get_bbox(annotation):
    """
    Get bbox from data
    Args:
        annotation (dict): dict of instance annotations for a single instance.
    Returns:
        bbox (ndarray): x1, y1, x2, y2 coordinates
    """
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    return bbox


def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    if "keypoints" in annotation:
        keypoints = transform_keypoint_annotations(
            annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
        )
        annotation["keypoints"] = keypoints

    return annotation


def transform_keypoint_annotations(keypoints, transforms, image_size, keypoint_hflip_indices=None):
    """
    Transform keypoint annotations of an image.
    If a keypoint is transformed out of image boundary, it will be marked "unlabeled" (visibility=0)

    Args:
        keypoints (list[float]): Nx3 float in Detectron2's Dataset format.
            Each point is represented by (x, y, visibility).
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.
            When `transforms` includes horizontal flip, will use the index
            mapping to flip keypoints.
    """
    # (N*3,) -> (N, 3)
    keypoints = np.asarray(keypoints, dtype="float64").reshape(-1, 3)
    keypoints_xy = transforms.apply_coords(keypoints[:, :2])

    # Set all out-of-boundary points to "unlabeled"
    inside = (keypoints_xy >= np.array([0, 0])) & (keypoints_xy <= np.array(image_size[::-1]))
    inside = inside.all(axis=1)
    keypoints[:, :2] = keypoints_xy
    keypoints[:, 2][~inside] = 0

    # This assumes that HorizFlipTransform is the only one that does flip
    do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1

    # Alternative way: check if probe points was horizontally flipped.
    # probe = np.asarray([[0.0, 0.0], [image_width, 0.0]])
    # probe_aug = transforms.apply_coords(probe.copy())
    # do_hflip = np.sign(probe[1][0] - probe[0][0]) != np.sign(probe_aug[1][0] - probe_aug[0][0])  # noqa

    # If flipped, swap each keypoint with its opposite-handed equivalent
    if do_hflip:
        if keypoint_hflip_indices is None:
            raise ValueError("Cannot flip keypoints without providing flip indices!")
        if len(keypoints) != len(keypoint_hflip_indices):
            raise ValueError(
                "Keypoint data has {} points, but metadata "
                "contains {} points!".format(len(keypoints), len(keypoint_hflip_indices))
            )
        keypoints = keypoints[np.asarray(keypoint_hflip_indices, dtype=np.int32), :]

    # Maintain COCO convention that if visibility == 0 (unlabeled), then x, y = 0
    keypoints[keypoints[:, 2] == 0] = 0
    return keypoints


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = (
        np.stack(
            [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
        )
        if len(annos)
        else np.zeros((0, 4))
    )
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            try:
                masks = PolygonMasks(segms)
            except ValueError as e:
                raise ValueError(
                    "Failed to use mask_format=='polygon' from the given annotations!"
                ) from e
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a binary segmentation mask "
                        " in a 2D numpy array of shape HxW.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    return target


def annotations_to_instances_rotated(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.
    Compared to `annotations_to_instances`, this function is for rotated boxes only

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            Containing fields "gt_boxes", "gt_classes",
            if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [obj["bbox"] for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = RotatedBoxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    return target


def filter_empty_instances(
    instances, by_box=True, by_mask=True, box_threshold=1e-5, return_mask=False
):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty
        return_mask (bool): whether to return boolean mask of filtered instances

    Returns:
        Instances: the filtered instances.
        tensor[bool], optional: boolean mask of filtered instances
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    if return_mask:
        return instances[m], m
    return instances[m]


def create_keypoint_hflip_indices(dataset_names: Union[str, List[str]]) -> List[int]:
    """
    Args:
        dataset_names: list of dataset names

    Returns:
        list[int]: a list of size=#keypoints, storing the
        horizontally-flipped keypoint indices.
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    check_metadata_consistency("keypoint_names", dataset_names)
    check_metadata_consistency("keypoint_flip_map", dataset_names)

    meta = MetadataCatalog.get(dataset_names[0])
    names = meta.keypoint_names
    # TODO flip -> hflip
    flip_map = dict(meta.keypoint_flip_map)
    flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in flip_map else flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return flip_indices


def get_fed_loss_cls_weights(dataset_names: Union[str, List[str]], freq_weight_power=1.0):
    """
    Get frequency weight for each class sorted by class id.
    We now calcualte freqency weight using image_count to the power freq_weight_power.

    Args:
        dataset_names: list of dataset names
        freq_weight_power: power value
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    check_metadata_consistency("class_image_count", dataset_names)

    meta = MetadataCatalog.get(dataset_names[0])
    class_freq_meta = meta.class_image_count
    class_freq = torch.tensor(
        [c["image_count"] for c in sorted(class_freq_meta, key=lambda x: x["id"])]
    )
    class_freq_weight = class_freq.float() ** freq_weight_power
    return class_freq_weight


def gen_crop_transform_with_instance(crop_size, image_size, instance):
    """
    Generate a CropTransform so that the cropping region contains
    the center of the given instance.

    Args:
        crop_size (tuple): h, w in pixels
        image_size (tuple): h, w
        instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
    """
    crop_size = np.asarray(crop_size, dtype=np.int32)
    bbox = BoxMode.convert(instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
    assert (
        image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
    ), "The annotation bounding box is outside of the image!"
    assert (
        image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
    ), "Crop size is larger than image size!"

    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)
    return T.CropTransform(x0, y0, crop_size[1], crop_size[0])


def check_metadata_consistency(key, dataset_names):
    """
    Check that the datasets have consistent metadata.

    Args:
        key (str): a metadata key
        dataset_names (list[str]): a list of dataset names

    Raises:
        AttributeError: if the key does not exist in the metadata
        ValueError: if the given datasets do not have the same metadata values defined by key
    """
    if len(dataset_names) == 0:
        return
    logger = logging.getLogger(__name__)
    entries_per_dataset = [getattr(MetadataCatalog.get(d), key) for d in dataset_names]
    for idx, entry in enumerate(entries_per_dataset):
        if entry != entries_per_dataset[0]:
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'".format(key, dataset_names[idx], str(entry))
            )
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'".format(
                    key, dataset_names[0], str(entries_per_dataset[0])
                )
            )
            raise ValueError("Datasets have different metadata '{}'!".format(key))


def build_augmentation(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    if is_train and cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )
    return augmentation


build_transform_gen = build_augmentation
"""
Alias for backward-compatibility.
"""


def softmax_with_bins(input, bins):
    # input: [N, D], bins: [D]
    # return: [N]
    return (nn.functional.softmax(input, dim=1) * bins).sum(dim=1)  # sum is reducing dims; [N]


def argmax_with_bins(input, bins):
    # input: [N, D], bins: [D]
    # return: [N]
    if bins.dim() == 1:
        idxx = torch.argmax(input, dim=1)
        est_batch = bins[idxx]
    else:
        # input: [N, D], bins: [N, D]
        # return: [N]
        N, D = input.shape

        # Get the index of the max logit for each row: [N]
        idxx = torch.argmax(input, dim=1)
        # Use advanced indexing to pick the bin at (row_i, idxx_i)
        # bins[0, idxx[0]], bins[1, idxx[1]], ...
        est_batch = bins[torch.arange(N), idxx]
    return est_batch


def prob_to_est(input, bins, is_training: bool = True):
    if is_training:
        return softmax_with_bins(input, bins)
    else:
        return argmax_with_bins(input, bins)


def get_straighten_ratio_from_kps(keypoints, kp_thresh=2):
    ratio_batch = []
    dataset_keypoints = MetadataCatalog.get("keypoints_coco_2017_train").keypoint_names
    for kps in keypoints:
        kps = kps.transpose(1, 0)
        mid_shoulder = (
            kps[:2, dataset_keypoints.index("right_shoulder")] +
            kps[:2, dataset_keypoints.index("left_shoulder")]
        ) / 2.0
        sc_mid_shoulder = torch.min(
            kps[2, dataset_keypoints.index("right_shoulder")],
            kps[2, dataset_keypoints.index("left_shoulder")],
        )
        mid_hip = (
            kps[:2, dataset_keypoints.index("right_hip")] +
            kps[:2, dataset_keypoints.index("left_hip")]
        ) / 2.0
        sc_mid_hip = torch.min(
            kps[2, dataset_keypoints.index("right_hip")],
            kps[2, dataset_keypoints.index("left_hip")],
        )
        dist_pred = 0.0
        dist_straighten = 0.0

        if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
            dist_pred += torch.abs(mid_shoulder[1] - mid_hip[1])
            dist_straighten += pts_dist(mid_shoulder, mid_hip)

        which_side_vis_list = []
        for which_side in ["left", "right"]:
            which_side_vis_list.append(
                kps[2, dataset_keypoints.index(which_side + "_hip")] > kp_thresh and
                kps[2, dataset_keypoints.index(which_side + "_ankle")] >
                kp_thresh and
                kps[2, dataset_keypoints.index(which_side + "_knee")] >
                kp_thresh,
            )
        which_side_vis = torch.as_tensor(which_side_vis_list).to(keypoints[0].device)
        sides_reweight = torch.sum(which_side_vis)
        if sides_reweight > 0:
            which_side_weight_array = (
                which_side_vis / sides_reweight
            )
        else:
            which_side_weight_array = [0.0, 0.0]

        for which_side, _, which_side_weight in zip(
            ["left", "right"],
            which_side_vis_list,
            which_side_weight_array,
        ):
            kps_hip = kps[:2, dataset_keypoints.index(which_side + "_hip")]
            kps_knee = kps[:2, dataset_keypoints.index(which_side + "_knee")]
            kps_ankle = kps[:2, dataset_keypoints.index(which_side + "_ankle")]
            dist_pred_side = torch.abs(kps_hip[1] - kps_ankle[1])
            dist_straighten_side = pts_dist(kps_hip, kps_knee) + pts_dist(
                kps_ankle,
                kps_knee,
            )
            dist_pred += dist_pred_side * which_side_weight
            dist_straighten += dist_straighten_side * which_side_weight

        if dist_straighten == 0.0 or dist_pred == 0.0:
            ratio = 1.0
        else:
            ratio = torch.clip(dist_pred / dist_straighten, 1e-5, 1.0)
            assert ratio > 0.0 and ratio <= 1.01, "ratio is %.2f!" % ratio
        ratio_batch.append(ratio)
    return ratio_batch


def pts_dist(pts1, pts2):
    return torch.sqrt((pts1[0] - pts2[0]) ** 2 + (pts1[1] - pts2[1]) ** 2)


def fit_camH(bbox, H, v0, vc, f_pixels, y_person):
    # bbox: [x, y, w, h]
    # H: image height in px
    # input and return both in [top H, bottom 0] space
    vt = H - bbox[1]
    vb = H - (bbox[1] + bbox[3])
    yc_single = y_person * (v0 - vb) / (vt - vb) / (1. + (vc - v0) * (vc - vt) / f_pixels**2)
    return yc_single


def fit_vt(yc_fit, vb, v0, vc, y_person_fit, f_pixels_est):
    inv_f2 = 1. / (f_pixels_est * f_pixels_est)
    # input and return both in [top H, bottom 0] space
    vt_camFit = (yc_fit * vb + y_person_fit * (v0 - vb) * (1. + inv_f2 * (vc - v0) * vc)) / \
        (yc_fit + y_person_fit * (v0 - vb) * inv_f2 * (vc - v0))
    return vt_camFit


def accu_model_batch(dataset_dict: dict):
    yc_est, vb, pred_height, v0, vc, f_pixels_est = (
        dataset_dict["yc_est"],
        dataset_dict["vb"],
        dataset_dict["pred_height"],
        dataset_dict["v0"],
        dataset_dict["vc"],
        dataset_dict["f_pixels_est"],
    )
    pitch_key = 'pitch'
    if pitch_key in dataset_dict:
        # Change the sign of the pitch
        theta_yannick = - dataset_dict[pitch_key]
    else:
        theta_yannick = torch.atan((vc - v0) / f_pixels_est)
    y_person = pred_height * torch.cos(theta_yannick)
    if "straighten_ratio" in dataset_dict:
        y_person = y_person * dataset_dict["straighten_ratio"]

    # Multiply the height by the pitch value
    z = - (f_pixels_est * yc_est) / (f_pixels_est *
                                     torch.sin(theta_yannick) - (vc - vb) * torch.cos(theta_yannick) + 1e-10)
    vt_camEst = ((f_pixels_est * torch.cos(theta_yannick) + vc * torch.sin(theta_yannick)) * y_person +
                 (-f_pixels_est * torch.sin(theta_yannick) + vc * torch.cos(theta_yannick)) * z +
                 -f_pixels_est * yc_est) \
        / (y_person * torch.sin(theta_yannick) + z * torch.cos(theta_yannick) + 1e-10)
    return vt_camEst, z


def _move_logits_to_device(batched_inputs: List[Dict[str, torch.Tensor]], device):
    # NOTE: check whether that is a better way to map this elsewhere.
    if "logits" in batched_inputs[0]:
        for i, _ in enumerate(batched_inputs):
            x = batched_inputs[i]["logits"].copy()
            batched_inputs[i]["logits"] = dict(
                gt_horizon=x["gt_horizon"].to(device),
                gt_pitch=x["gt_pitch"].to(device),
                gt_roll=x["gt_roll"].to(device),
                gt_vfov=x["gt_vfov"].to(device),
            )
    return batched_inputs


def _add_whole_image_as_proposal(images, device):
    # Set the whole image as a proposal region for camera head.
    proposals = []
    for image_size in images.image_sizes:
        h, w = image_size
        # one box covering the whole image
        full_box = torch.tensor([[0.0, 0.0, w, h]], device=device)
        inst = Instances(image_size)
        inst.proposal_boxes = Boxes(full_box)
        inst.objectness_logits = torch.ones(1, device=device)
        proposals.append(inst)
    return proposals


def pad_to_max(tensor_list, max_n, pad_value=0.0):
    padded = torch.zeros(
        (len(tensor_list), max_n, *tensor_list[0].shape[1:]), device=tensor_list[0].device, dtype=tensor_list[0].dtype)
    for i, t in enumerate(tensor_list):
        n = min(t.shape[0], max_n)
        padded[i][:n] = t[:n]
        padded[i][n:] = padded[i][n:] + pad_value
    return padded


def human_prior(tensor, height_mean, height_std, ):
    return 1. / np.sqrt(2. * np.pi * (height_std**2)) * torch.exp(-(tensor - height_mean)**2 / (2. * height_std**2))


def person_h_list_loss(all_person_hs, height_mean, height_std, num_instances):
    prob_all_person_hs = human_prior(all_person_hs, height_mean, height_std)
    prob_all_person_h_list = prob_all_person_hs.split(num_instances)
    return -torch.mean(
        torch.stack(
            [
                torch.mean(prob_all_person_h)
                for prob_all_person_h in prob_all_person_h_list
                # To deal with empty predictions in a batch
                if prob_all_person_h.numel() > 0
            ]
        )
    )


def person_h_list_loss_masked(all_person_hs, height_mean, height_std, mask):
    prob_all_person_hs = human_prior(all_person_hs, height_mean, height_std)
    return -((prob_all_person_hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(torch.finfo(prob_all_person_hs.dtype).eps)).mean()
