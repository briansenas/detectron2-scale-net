import numpy as np
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn

from detectron2.config import configurable
from detectron2.data.datasets.pano360 import (
    bins2pitch,
    bins2roll,
    bins2vfov,
    getHorizonLine,
    horizon_bins_centers,
    pitch_bins_centers,
    showHorizonLine,
    vfov_bins_centers,
)
from detectron2.data.detection_utils import (
    _add_whole_image_as_proposal,
    _move_logits_to_device,
    accu_model_batch,
    convert_image_to_rgb,
    get_straighten_ratio_from_kps,
    prob_to_est,
)
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
        horizon_bins_center: np.ndarray,
        pitch_bins_center: np.ndarray,
        vfov_bins_center: np.ndarray,
        reduce_method: str = "softmax",
        smooth_l1_beta: float = 0.0,
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
        self.register_buffer("horizon_bins_center", horizon_bins_center)
        self.register_buffer("pitch_bins_center", pitch_bins_center)
        self.register_buffer("vfov_bins_center", vfov_bins_center)
        self.reduce_method = reduce_method
        self.smooth_l1_beta = smooth_l1_beta

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
            "horizon_bins_center": torch.as_tensor(horizon_bins_centers),
            "pitch_bins_center": torch.as_tensor(pitch_bins_centers),
            "vfov_bins_center": torch.as_tensor(vfov_bins_centers),
            "reduce_method": cfg.MODEL.HEIGHT_HEAD.REDUCE_METHOD,
            "smooth_l1_beta": cfg.MODEL.HEIGHT_HEAD.SMOOTH_L1_BETA,
        }

    def _camrcnn_predictions(self, images, features, predicted_proposals: List[Instances]):
        self.camera_heads.eval()
        with torch.no_grad():
            cls_logits, _ = self.camera_heads(features, _add_whole_image_as_proposal(images, self.device), None)
        self.camera_heads.train()
        vt_loss_sample_list = []
        vfov_est = prob_to_est(
            cls_logits["vfov_logits"], self.vfov_bins_center, self.reduce_method
        )
        horizon_est = prob_to_est(
            cls_logits["horizon_logits"], self.horizon_bins_center, self.reduce_method
        )
        pitch_est = prob_to_est(
            cls_logits["pitch_logits"], self.pitch_bins_center, self.reduce_method
        )
        # NOTE: Should optimize and do this batchwise
        for i, instance in enumerate(predicted_proposals):
            # Here we will attempt to measure the vt_loss or somehow correlate the camera parameters with the height estimation
            # There is a missing link on how the model estimates heights at the moment
            if instance.pred_height.numel() < 1:
                continue
            H, _ = instance.image_size
            gt_bboxes = instance.gt_boxes.tensor
            f_estim = H / torch.tan(vfov_est[i] / 2) / 2
            v0_pred = (
                H - horizon_est[i] * H
            )  # (H = top of the image, 0 = bottom of the image)
            straighten_ratio = torch.as_tensor(
                get_straighten_ratio_from_kps(instance.pred_keypoints),
                device=self.device
            )
            h_human_s = instance.pred_height * straighten_ratio
            vb_batch = H - (gt_bboxes[:, 1] + gt_bboxes[:, 3])  # [top H bottom 0]
            vt_batch = H - gt_bboxes[:, 1]  # [top H bottom 0]
            vc = H / 2.0
            # NOTE: yc_est Should be the result of CamHPointNet
            yc_est = (
                instance.pred_height *
                (v0_pred - vb_batch) /
                (vt_batch - vb_batch) /
                (1.0 + (vc - v0_pred) * (vc - vt_batch) / f_estim**2)
            )
            geo_model_input_dict = {
                "yc_est": yc_est,
                "vb": vb_batch,
                "y_person": h_human_s * torch.cos(pitch_est[i]),
                "v0": v0_pred,
                "vc": vc,
                "f_pixels_est": f_estim,
                "pitch_est": pitch_est[i],
            }
            vt_camEst_batch, _, _ = accu_model_batch(
                geo_model_input_dict
            )
            vt_loss_ori_batch = (
                smooth_l1_loss(vt_batch, vt_camEst_batch, beta=self.smooth_l1_beta) / gt_bboxes[:, 3]
            )
            vt_loss_ori_batch = torch.where(
                torch.isnan(vt_loss_ori_batch),
                torch.zeros_like(vt_loss_ori_batch),
                vt_loss_ori_batch,
            )
            vt_loss_batch = torch.clamp(vt_loss_ori_batch, 0.0, 2)
            vt_loss_sample = torch.mean(vt_loss_batch)
            vt_loss_sample_list.append(vt_loss_sample)
        losses = {}
        if vt_loss_sample_list:
            vt_loss = torch.mean(torch.stack(vt_loss_sample_list))
            losses.update({"vt_loss": vt_loss})
        return {}, losses

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

        predicted_proposals, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return predicted_proposals, losses

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
        return predictions, losses

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
        losses = {}
        if dt_inputs:
            dt_image_list_slice = ImageList(
                tensor=images.tensor[: len(dt_inputs)],
                image_sizes=images.image_sizes[: len(dt_inputs)],
            )
            dt_features_slice = {k: v[: len(dt_inputs)] for k, v in features.items()}
            predicted_proposals, dt_losses = self._forward_generalized_rcnn(
                dt_inputs,
                dt_image_list_slice,
                dt_features_slice,
            )
            losses.update(dt_losses)
            predictions, vt_loss = self._camrcnn_predictions(
                dt_image_list_slice,
                dt_features_slice,
                predicted_proposals,
            )
            losses.update(vt_loss)
        if cam_inputs:
            _, cam_losses = self._forward_camrcnn(
                cam_inputs,
                ImageList(
                    tensor=images.tensor[len(dt_inputs):],
                    image_sizes=images.image_sizes[len(dt_inputs):],
                ),
                {k: v[len(dt_inputs):] for k, v in features.items()},
            )
            losses.update(cam_losses)
        return losses

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

        cam_results, _ = self.camera_heads(images, features, _add_whole_image_as_proposal(images, self.device), None)
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
