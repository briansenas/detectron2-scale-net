import numpy as np
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.datasets.pano360 import bins2pitch, bins2roll, bins2vfov, showHorizonLine
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage

from typing import Dict, List, Optional, Tuple

from ..backbone import Backbone, build_backbone
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_camera_head, build_roi_heads
from .build import META_ARCH_REGISTRY
from .rcnn import GeneralizedRCNN

__all__ = ["CameraRCNN", "GeneralizedCamRCNN"]


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


@META_ARCH_REGISTRY.register()
class CameraRCNN(nn.Module):
    """"""

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        camera_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            camera_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.camera_heads = camera_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert (
                input_format is not None
            ), "input_format is required for visualization!"

        self.register_buffer(
            "pixel_mean",
            torch.tensor(pixel_mean).view(-1, 1, 1),
            False,
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @property
    def device(self):
        return self.pixel_mean.device

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(
                cfg,
                backbone.output_shape(),
            ),
            "camera_heads": build_camera_head(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        gt_instances = None

        features = self.backbone(images.tensor)
        proposals = _add_whole_image_as_proposal(images, self.device)
        proposal_losses = {}

        # Inyect GT manually to avoid other detectron2 previous logic
        if "logits" in batched_inputs[0]:
            gt_instances = [
                dict(
                    gt_horizon=x["logits"]["gt_horizon"].to(self.device),
                    gt_pitch=x["logits"]["gt_pitch"].to(self.device),
                    gt_roll=x["logits"]["gt_roll"].to(self.device),
                    gt_vfov=x["logits"]["gt_vfov"].to(self.device),
                )
                for x in batched_inputs
            ]
        predictions, detector_losses = self.camera_heads(
            images,
            features,
            proposals,
            gt_instances,
        )
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, predictions)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training
        batched_inputs = _move_logits_to_device(batched_inputs, self.device)
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals = _add_whole_image_as_proposal(images, self.device)
        results, _ = self.camera_heads(images, features, proposals, None)
        return results

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals.
        It shows the predicted camera parameters and the estimated horizon line.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        input = batched_inputs[0]
        pitch_logits = proposals["pitch_logits"][0].detach().cpu().numpy().squeeze()
        roll_logits = proposals["roll_logits"][0].detach().cpu().numpy().squeeze()
        vfov_logits = proposals["vfov_logits"][0].detach().cpu().numpy().squeeze()
        img = input["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        pitch = bins2pitch(pitch_logits)
        roll = bins2roll(roll_logits)
        vfov = bins2vfov(vfov_logits)
        gt_pitch = bins2pitch(input["logits"]["gt_pitch"])
        gt_roll = bins2roll(input["logits"]["gt_roll"])
        gt_vfov = bins2vfov(input["logits"]["gt_vfov"])
        anno_img, _ = showHorizonLine(img, gt_vfov, gt_pitch, gt_roll)
        prop_img, _ = showHorizonLine(img, vfov, pitch, roll)
        vis_img = np.concatenate((anno_img, prop_img), axis=1)
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = "Left: GT Horizon;  Right: Predicted Horizon"
        storage.put_image(vis_name, vis_img)


@META_ARCH_REGISTRY.register()
class GeneralizedCamRCNN(GeneralizedRCNN):
    """"""

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        camera_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            camera_heads: A ROI head that performs camera parameter estimation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__(
            backbone=backbone,
            proposal_generator=proposal_generator,
            roi_heads=roi_heads,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            input_format=input_format,
            vis_period=vis_period,
        )
        self.camera_heads = camera_heads

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(
                cfg,
                backbone.output_shape(),
            ),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "camera_heads": build_camera_head(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    def _forward_generalized_rcnn(self, batched_inputs: List[Dict[str, torch.Tensor]], images, features):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        # Check if tensors share memory
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def _forward_camrcnn(self, batched_inputs: List[Dict[str, torch.Tensor]], images, features):
        proposals = _add_whole_image_as_proposal(images, self.device)

        if "logits" in batched_inputs[0]:
            gt_instances = [
                dict(
                    gt_horizon=x["logits"]["gt_horizon"].to(self.device),
                    gt_pitch=x["logits"]["gt_pitch"].to(self.device),
                    gt_roll=x["logits"]["gt_roll"].to(self.device),
                    gt_vfov=x["logits"]["gt_vfov"].to(self.device),
                )
                for x in batched_inputs
            ]
        predictions, detector_losses = self.camera_heads(
            images,
            features,
            proposals,
            gt_instances,
        )
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training_camrcnn(batched_inputs, predictions)

        losses = {}
        losses.update(detector_losses)
        return losses

    def _parse_dt_cam_inputs(self, batched_inputs):
        dt_inputs = []
        cam_inputs = []
        for input in batched_inputs:
            dt_inputs += input["coco_data"]
            cam_inputs += input["calib_data"]
        cam_inputs = _move_logits_to_device(cam_inputs, self.device)
        return dt_inputs, cam_inputs

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        if not self.training:
            return self.inference(batched_inputs)
        dt_inputs, cam_inputs = self._parse_dt_cam_inputs(batched_inputs)
        all_inputs = dt_inputs + cam_inputs
        images = self.preprocess_image(all_inputs)
        # NOTE: Doing all together means I need more memory
        # but it's slightly faster than doing it in parts
        features = self.backbone(images.tensor)
        dt_losses = {}
        cam_losses = {}
        if dt_inputs:
            dt_losses = self._forward_generalized_rcnn(
                dt_inputs,
                ImageList(
                    tensor=images.tensor[: len(dt_inputs)],
                    image_sizes=images.image_sizes[: len(dt_inputs)],
                ),
                {k: v[: len(dt_inputs)] for k, v in features.items()},
            )
        if cam_inputs:
            cam_losses = self._forward_camrcnn(
                cam_inputs,
                ImageList(
                    tensor=images.tensor[len(dt_inputs):],
                    image_sizes=images.image_sizes[len(dt_inputs):],
                ),
                {k: v[len(dt_inputs):] for k, v in features.items()},
            )
        return {**dt_losses, **cam_losses}

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training
        if "coco_data" in batched_inputs[0]:
            dt_inputs, cam_inputs = self._parse_dt_cam_inputs(batched_inputs)
            batched_inputs = dt_inputs + cam_inputs
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            results = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

        proposals = _add_whole_image_as_proposal(images, self.device)
        cam_results, _ = self.camera_heads(images, features, proposals, None)
        return {**results, **cam_results}

    def visualize_training_camrcnn(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals.
        It shows the predicted camera parameters and the estimated horizon line.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        input = batched_inputs[0]
        if "annotation" in input:
            # Visualize
            super().visualize_training(batched_inputs, proposals)
        else:
            # Visualize camera parameter estimation
            pitch_logits = proposals["pitch_logits"][0].detach().cpu().numpy().squeeze()
            roll_logits = proposals["roll_logits"][0].detach().cpu().numpy().squeeze()
            vfov_logits = proposals["vfov_logits"][0].detach().cpu().numpy().squeeze()
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            pitch = bins2pitch(pitch_logits)
            roll = bins2roll(roll_logits)
            vfov = bins2vfov(vfov_logits)
            gt_pitch = bins2pitch(input["logits"]["gt_pitch"].detach().cpu().numpy().squeeze())
            gt_roll = bins2roll(input["logits"]["gt_roll"].detach().cpu().numpy().squeeze())
            gt_vfov = bins2vfov(input["logits"]["gt_vfov"].detach().cpu().numpy().squeeze())
            anno_img, _ = showHorizonLine(img, gt_vfov, gt_pitch, gt_roll)
            prop_img, _ = showHorizonLine(img, vfov, pitch, roll)
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT Horizon;  Right: Predicted Horizon"
            storage.put_image(vis_name, vis_img)
