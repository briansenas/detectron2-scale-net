# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.datasets.pano360 import COCO_SCALE_STATS, human_bins
from detectron2.data.detection_utils import person_h_list_loss, prob_to_est
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, Keypoints, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

import inspect
import logging
from typing import Dict, List, Optional, Tuple

from ..backbone.resnet import BottleneckBlock, ResNet
from ..matcher import Matcher
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from ..sampling import subsample_labels
from .box_head import (
    FastRCNNConvFCHead,
    FastRCNNConvFCHeadCamera,
    FastRCNNConvFCHeadHeight,
    build_box_head,
)
from .fast_rcnn import FastRCNNOutputLayers
from .keypoint_head import build_keypoint_head, keypoint_rcnn_inference_no_heatmap
from .mask_head import build_mask_head

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def build_camera_head(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.CAMERA_HEAD.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(
    proposals: List[Instances],
    bg_label: int,
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


def select_proposals_with_visible_keypoints(
    proposals: List[Instances],
) -> List[Instances]:
    """
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.

    Returns:
        proposals: only contains proposals with at least one visible keypoint.

    Note that this is still slightly different from Detectron.
    In Detectron, proposals for training keypoint head are re-sampled from
    all the proposals with IOU>threshold & >=1 visible keypoint.

    Here, the proposals are first sampled from all proposals with
    IOU>threshold, then proposals with no visible keypoint are filtered out.
    This strategy seems to make no difference on Detectron and is easier to implement.
    """
    ret = []
    all_num_fg = []
    for proposals_per_image in proposals:
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(
            dim=1,
        )  # #fg x 1 x 4
        kp_in_box = (
            (xs >= proposal_boxes[:, :, 0]) &
            (xs <= proposal_boxes[:, :, 2]) &
            (ys >= proposal_boxes[:, :, 1]) &
            (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = nonzero_tuple(selection)[0]
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])

    storage = get_event_storage()
    storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg))
    return ret


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It typically contains logic to

    1. (in training only) match proposals with ground truth and sample them
    2. crop the regions and extract per-region features using proposals
    3. make per-region predictions with different heads

    It can have many variants, implemented as subclasses of this class.
    This base class contains the logic to match/sample proposals.
    But it is not necessary to inherit this class if the sampling logic is not needed.
    """

    @configurable
    def __init__(
        self,
        *,
        num_classes,
        batch_size_per_image,
        positive_fraction,
        proposal_matcher,
        proposal_append_gt=True,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            num_classes (int): number of foreground classes (i.e. background is not included)
            batch_size_per_image (int): number of proposals to sample for training
            positive_fraction (float): fraction of positive (foreground) proposals
                to sample for training.
            proposal_matcher (Matcher): matcher that matches proposals and ground truth
            proposal_append_gt (bool): whether to include ground truth as proposals as well
        """
        super().__init__()
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.num_classes = num_classes
        self.proposal_matcher = proposal_matcher
        self.proposal_append_gt = proposal_append_gt

    @classmethod
    def from_config(cls, cfg):
        return {
            "batch_size_per_image": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "proposal_append_gt": cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT,
            # Matcher to assign box proposals to gt boxes
            "proposal_matcher": Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            ),
        }

    def _sample_proposals(
        self,
        matched_idxs: torch.Tensor,
        matched_labels: torch.Tensor,
        gt_classes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_image,
            self.positive_fraction,
            self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self,
        proposals: List[Instances],
        targets: List[Instances],
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes,
                proposals_per_image.proposal_boxes,
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs,
                matched_labels,
                targets_per_image.gt_classes,
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for trg_name, trg_value in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name,
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            images (ImageList):
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            list[Instances]: length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.

            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    See :paper:`ResNet` Appendix A.
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        pooler: ROIPooler,
        res5: nn.Module,
        box_predictor: nn.Module,
        mask_head: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_features (list[str]): list of backbone feature map names to use for
                feature extraction
            pooler (ROIPooler): pooler to extra region features from backbone
            res5 (nn.Sequential): a CNN to compute per-region features, to be used by
                ``box_predictor`` and ``mask_head``. Typically this is a "res5"
                block from a ResNet.
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_head (nn.Module): transform features to make mask predictions
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.pooler = pooler
        if isinstance(res5, (list, tuple)):
            res5 = nn.Sequential(*res5)
        self.res5 = res5
        self.box_predictor = box_predictor
        self.mask_on = mask_head is not None
        if self.mask_on:
            self.mask_head = mask_head

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        ret = super().from_config(cfg)
        in_features = ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[in_features[0]].stride,)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        mask_on           = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(in_features) == 1

        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        # Compatbility with old moco code. Might be useful.
        # See notes in StandardROIHeads.from_config
        if not inspect.ismethod(cls._build_res5_block):
            logger.warning(
                "The behavior of _build_res5_block may change. "
                "Please do not depend on private methods.",
            )
            cls._build_res5_block = classmethod(cls._build_res5_block)

        ret["res5"], out_channels = cls._build_res5_block(cfg)
        ret["box_predictor"] = FastRCNNOutputLayers(
            cfg,
            ShapeSpec(channels=out_channels, height=1, width=1),
        )

        if mask_on:
            ret["mask_head"] = build_mask_head(
                cfg,
                ShapeSpec(
                    channels=out_channels,
                    width=pooler_resolution,
                    height=pooler_resolution,
                ),
            )
        return ret

    @classmethod
    def _build_res5_block(cls, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = ResNet.make_stage(
            BottleneckBlock,
            3,
            stride_per_block=[2, 1, 1],
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features: List[torch.Tensor], boxes: List[Boxes]):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features],
            proposal_boxes,
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals,
                    self.num_classes,
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self,
        features: Dict[str, torch.Tensor],
        instances: List[Instances],
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            feature_list = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(
                feature_list,
                [x.pred_boxes for x in instances],
            )
            return self.mask_head(x, instances)
        else:
            return instances


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head

        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )
        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels,
                width=pooler_resolution,
                height=pooler_resolution,
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["mask_head"] = build_mask_head(cfg, shape)
        return ret

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"keypoint_in_features": in_features}
        ret["keypoint_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels,
                width=pooler_resolution,
                height=pooler_resolution,
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["keypoint_head"] = build_keypoint_head(cfg, shape)
        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self,
        features: Dict[str, torch.Tensor],
        instances: List[Instances],
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            list[Instances]:
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
    ):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions,
                        proposals,
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals,
                        pred_boxes,
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def _forward_mask(
        self,
        features: Dict[str, torch.Tensor],
        instances: List[Instances],
    ):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [
                x.proposal_boxes if self.training else x.pred_boxes for x in instances
            ]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, instances)

    def _forward_keypoint(
        self,
        features: Dict[str, torch.Tensor],
        instances: List[Instances],
    ):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals with >=1 visible keypoints.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            instances = select_proposals_with_visible_keypoints(instances)

        if self.keypoint_pooler is not None:
            features = [features[f] for f in self.keypoint_in_features]
            boxes = [
                x.proposal_boxes if self.training else x.pred_boxes for x in instances
            ]
            features = self.keypoint_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.keypoint_in_features}
        return self.keypoint_head(features, instances)


@ROI_HEADS_REGISTRY.register()
class HeightStandardROIHeads(StandardROIHeads):
    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        padded_input_size: int = 10,
        height_predictor: Optional[nn.Module] = None,
        height_cls_score: Optional[nn.Module] = None,
        height_temperature: Optional[float] = 1.0,
        height_loss_weight: Optional[float] = None,
        height_mean: Optional[float] = None,
        height_std: Optional[float] = None,
        height_discount_from: Optional[str] = "GT",
        **kwargs,
    ):
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            mask_in_features=mask_in_features,
            mask_pooler=mask_pooler,
            mask_head=mask_head,
            keypoint_in_features=keypoint_in_features,
            keypoint_pooler=keypoint_pooler,
            keypoint_head=keypoint_head,
            train_on_pred_boxes=train_on_pred_boxes,
            **kwargs,
        )
        self.padded_input_size = padded_input_size
        self.height_loss_weight = height_loss_weight
        self.height_on = height_predictor is not None
        self.height_temperature = height_temperature
        self.height_discount_from = height_discount_from
        self.height_predictor = height_predictor
        self.height_cls_score = height_cls_score
        self.height_mean = height_mean
        self.height_std = height_std
        if height_mean:
            self.register_buffer("human_bins", torch.as_tensor(human_bins, dtype=torch.float16))

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"keypoint_in_features": in_features}
        ret["keypoint_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels,
                width=pooler_resolution,
                height=pooler_resolution,
            )
        else:
            shape = {f: input_shape[f] for f in in_features}

        ret["keypoint_head"] = build_keypoint_head(cfg, shape)
        ret["padded_input_size"] = cfg.MODEL.HEIGHT_HEAD.PADDED_INPUT
        # If we set the number of Conv3x to 0 and FC-2
        # We will have the same predictor as Jerry
        if cfg.MODEL.HEIGHT_ON:
            ret["height_loss_weight"] = cfg.MODEL.HEIGHT_HEAD.LOSS_WEIGHT
            ret["height_temperature"] = cfg.MODEL.HEIGHT_HEAD.TEMPERATURE
            ret["height_discount_from"] = cfg.MODEL.HEIGHT_HEAD.DISCOUNT_FROM
            height_predictor = FastRCNNConvFCHeadHeight(
                cfg,
                shape,
            )
            height_cls_score = nn.Linear(
                cfg.MODEL.HEIGHT_HEAD.FC_DIM,
                cfg.MODEL.HEIGHT_HEAD.NUM_CLASSES,
            )
            nn.init.normal_(height_cls_score.weight, std=0.01)
            nn.init.constant_(height_cls_score.bias, 0)
            ret["height_predictor"] = height_predictor
            ret["height_cls_score"] = height_cls_score
            ret["height_mean"] = cfg.MODEL.HEIGHT_MEAN
            ret["height_std"] = cfg.MODEL.HEIGHT_STD
        return ret

    def _add_gt_to_pred(
        self,
        pred_instances,
        gt_instances,
        iou_thresh=0.5,
    ):
        matcher = Matcher([iou_thresh], [0, 1], allow_low_quality_matches=False)
        results = []
        M = min(max([inst.pred_boxes.tensor.shape[0] for inst in pred_instances]), self.padded_input_size)
        for preds, gts in zip(pred_instances, gt_instances):
            device = preds.pred_boxes.tensor.device
            if len(preds) > 0:
                pred_boxes = preds.pred_boxes
                pred_classes = preds.pred_classes
                gt_boxes = gts.gt_boxes
                iou_matrix = pairwise_iou(gt_boxes, pred_boxes)
                matched_idxs, labels = matcher(iou_matrix)
                # Positive matches
                valid_mask = labels == 1
                # keep only valid matches
                matched_idxs = matched_idxs[valid_mask]
                pred_boxes.tensor = pred_boxes.tensor[valid_mask]
                gt_fields = {}
                for k, v in gts.get_fields().items():
                    if isinstance(v, torch.Tensor):
                        gt_fields[k] = v[matched_idxs]
                    else:
                        # e.g. Boxes, BitMasks, etc.
                        gt_fields[k] = v[matched_idxs]
                pred_boxes = pred_boxes.tensor
            else:
                pred_boxes = torch.zeros((0, 4), device=device)
                pred_classes = torch.zeros((0,), dtype=torch.long, device=device) - 1  # Set the class to background
                valid_mask = torch.zeros_like(pred_classes, dtype=torch.bool, device=device)
                gt_fields = {
                    k: v
                    for k, v in gts.get_fields().items()
                }

            valid_mask = torch.ones(gt_fields["gt_boxes"].tensor.shape[0], dtype=bool, device=device)
            valid_mask = pad_tensor(valid_mask, M, False)
            pred_boxes = pad_tensor(pred_boxes, M, 0)
            pred_classes = pad_tensor(pred_classes, M, -1)

            inst = Instances(
                image_size=preds.image_size if len(preds) > 0 else gts.image_size
            )

            inst.pred_boxes = Boxes(pred_boxes)
            inst.proposal_boxes = Boxes(pred_boxes)
            inst.pred_classes = pred_classes
            inst.valid_mask = valid_mask
            # Copy fields
            for k, v in gt_fields.items():
                if k == "gt_classes":
                    setattr(inst, k, pad_tensor(v, M, -1))
                elif isinstance(v, torch.Tensor):
                    setattr(
                        inst, k, pad_tensor(v, M, 0)
                    )
                else:
                    if isinstance(v, Boxes):
                        setattr(
                            inst, k, Boxes(pad_tensor(v.tensor, M, 0))
                        )
                    if isinstance(v, Keypoints):
                        setattr(
                            inst, k, Keypoints(pad_tensor(v.tensor, M, 0))
                        )
            results.append(inst)
        return results

    def _forward_keypoint(
        self,
        features: Dict[str, torch.Tensor],
        instances: List[Instances],
    ):
        if not self.keypoint_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals with >=1 visible keypoints.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            instances = select_proposals_with_visible_keypoints(instances)

        if self.keypoint_pooler is not None:
            features = [features[f] for f in self.keypoint_in_features]
            boxes = [
                x.proposal_boxes if self.training else x.pred_boxes for x in instances
            ]
            features = self.keypoint_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.keypoint_in_features}
        # This part is what we removed from normal keypoint_head forward_call:
        return self.keypoint_head(self.keypoint_head.layers(features), instances)

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            losses = {}
            if self.height_on:
                pred_instances, losses = self._forward_box_height(features, proposals)
                with torch.no_grad():
                    pred_instances = self._add_gt_to_pred(pred_instances, targets)
                del targets
            else:
                losses.update(self._forward_box(features, proposals))
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            # NOTE: for multi-cat I would've to edit this forward to always do keypoint_inference_proposals
            # So that for the instances that contain people I get a new field keypoints (hopefully w pointers).
            losses.update(self._forward_keypoint(features, proposals))
            if self.height_on:
                proposals, keypoint_losses = self._forward_keypoint_height(features, pred_instances)
                losses.update(keypoint_losses)
            return proposals, losses
        else:
            if self.height_on:
                pred_instances, _ = self._forward_box_height(features, proposals)
            else:
                pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances, _ = self._forward_keypoint_height(features, pred_instances)
            return pred_instances, {}

    def _forward_box_height(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
    ):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features
        losses = {}
        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions,
                        proposals,
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals,
                        pred_boxes,
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
        with torch.no_grad():
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
        return pred_instances, losses

    def _forward_keypoint_height(
        self,
        features: Dict[str, torch.Tensor],
        instances: List[Instances],
    ):
        losses = {}
        if self.training:
            # head is only trained on positive proposals with >=1 visible keypoints.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            if self.height_discount_from == "GT":
                instances = select_proposals_with_visible_keypoints(instances)

        if self.keypoint_pooler is not None:
            features = [features[f] for f in self.keypoint_in_features]
            boxes = [
                x.gt_boxes if self.training else x.pred_boxes for x in instances
            ]
            features = self.keypoint_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.keypoint_in_features}
        # Copy of keypoint_rcnn_inference that don't save the logits during training.
        if self.height_discount_from != "GT":
            keypoint_rcnn_inference_no_heatmap(self.keypoint_head.layers(features), instances)
        all_person_hs = prob_to_est(
            self.height_cls_score(self.height_predictor(features)) / self.height_temperature, self.human_bins
        )
        del features
        num_instances_per_image = [len(i) for i in instances]
        for height, pred_instances in zip(all_person_hs.split(num_instances_per_image), instances):
            # pred_instances.pred_height = torch.zeros_like(height, device=height.device) + 1.75
            pred_instances.pred_height = height
        if self.training:
            losses["height_loss"] = person_h_list_loss(
                all_person_hs,
                self.height_mean,
                self.height_std,
                num_instances_per_image,
            ) * self.height_loss_weight
        return instances, losses


@ROI_HEADS_REGISTRY.register()
class CameraHead(ROIHeads):
    @configurable
    def __init__(
        self,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        predictor: FastRCNNConvFCHead,
        cls_score: nn.Linear,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.predictor = predictor
        self.cls_score = cls_score

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        if inspect.ismethod(cls._init_classifier):
            ret.update(cls._init_classifier(cfg, input_shape))
        return ret

    @classmethod
    def _init_classifier(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.CAMERA_HEAD.IN_FEATURES
        pooler_resolution = cfg.MODEL.CAMERA_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.CAMERA_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.CAMERA_HEAD.POOLER_TYPE
        # fmt: on
        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        # If we set the number of Conv3x to 0 and FC-2
        # We will have the same predictor as Jerry
        predictor = FastRCNNConvFCHeadCamera(
            cfg,
            ShapeSpec(
                channels=in_channels,
                width=pooler_resolution,
                height=pooler_resolution,
            ),
        )
        cls_score = nn.Linear(
            cfg.MODEL.CAMERA_HEAD.FC_DIM,
            cfg.MODEL.CAMERA_HEAD.NUM_CLASSES,
        )
        nn.init.normal_(cls_score.weight, std=0.01)
        nn.init.constant_(cls_score.bias, 0)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "predictor": predictor,
            "cls_score": cls_score,
        }

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        x = self.predictor(box_features)
        del features
        class_logits = self.cls_score(x)
        del x
        return class_logits

    def compute_loss(self, logits, targets):
        assert targets, "'targets' argument is required during training"
        targets = torch.as_tensor(targets).to(device=logits.device, dtype=torch.long)
        targets = torch.nn.functional.one_hot(targets, num_classes=256).to(logits.dtype)
        return nn.functional.kl_div(
            nn.functional.log_softmax(logits, dim=1, dtype=logits.dtype),
            targets,
            reduction="batchmean",
        )


@ROI_HEADS_REGISTRY.register()
class CombinedCameraHeads(ROIHeads):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    @configurable
    def __init__(
        self,
        classifier_horizon: CameraHead,
        classifier_pitch: CameraHead,
        classifier_roll: CameraHead,
        classifier_vfov: CameraHead,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.classifier_horizon = classifier_horizon
        self.classifier_pitch = classifier_pitch
        self.classifier_vfov = classifier_vfov
        self.classifier_roll = classifier_roll

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret.update(
            dict(classifier_horizon=CameraHead(cfg=cfg, input_shape=input_shape)),
        )
        ret.update(
            dict(classifier_pitch=CameraHead(cfg=cfg, input_shape=input_shape)),
        )
        ret.update(
            dict(classifier_vfov=CameraHead(cfg=cfg, input_shape=input_shape)),
        )
        ret.update(
            dict(classifier_roll=CameraHead(cfg=cfg, input_shape=input_shape)),
        )
        return ret

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        losses = {}
        horizon_logits = self.classifier_horizon(
            features,
            proposals,
        )
        pitch_logits = self.classifier_pitch(
            features,
            proposals,
        )
        roll_logits = self.classifier_roll(
            features,
            proposals,
        )
        vfov_logits = self.classifier_vfov(
            features,
            proposals,
        )
        predictions = {
            "horizon_logits": horizon_logits,
            "pitch_logits": pitch_logits,
            "roll_logits": roll_logits,
            "vfov_logits": vfov_logits,
        }
        losses = {}
        if self.training:
            assert targets, "'targets' argument is required during training"
            # If the x value has gt_horizon, it also has all the other gt unless dataset mapper is changed (which is not)
            idxs = [i for i, x in enumerate(targets) if "gt_horizon" in x]
            assert len(idxs) > 0, "there are not gt camera parameters in targets"
            horizon_loss = self.classifier_horizon.compute_loss(
                horizon_logits[idxs],
                # Targets don't allow idx slicing but is aligned by using the same list comprehension
                [x["gt_horizon"] for x in targets if "gt_horizon" in x],
            )
            pitch_loss = self.classifier_pitch.compute_loss(
                pitch_logits[idxs],
                [x["gt_pitch"] for x in targets if "gt_pitch" in x],
            )
            roll_loss = self.classifier_roll.compute_loss(
                roll_logits[idxs],
                [x["gt_roll"] for x in targets if "gt_roll" in x],
            )
            vfov_loss = self.classifier_vfov.compute_loss(
                vfov_logits[idxs],
                [x["gt_vfov"] for x in targets if "gt_vfov" in x],
            )
            losses = {
                "horizon_loss": horizon_loss,
                "pitch_loss": pitch_loss,
                "roll_loss": roll_loss,
                "vfov_loss": vfov_loss,
            }
        return predictions, losses


def pad_tensor(x, target_len, pad_value=0):
    """
    Pads or truncates tensor `x` along dim=0 to `target_len`.
    """
    out = torch.zeros((target_len, *x.shape[1:]), dtype=x.dtype, device=x.device)
    cur_len = x.shape[0]
    n = min(cur_len, target_len)
    out[:n] = x[:n]
    out[n:] += pad_value
    return out
