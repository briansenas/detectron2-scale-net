import numpy as np
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.pano360 import (
    bins2pitch,
    bins2roll,
    bins2vfov,
    horizon_bins_centers,
    pitch_bins_centers,
    roll_bins_centers,
    showHorizonLine,
    vfov_bins_centers,
    yc_bins_centers,
)
from detectron2.data.detection_utils import (
    _add_whole_image_as_proposal,
    _move_logits_to_device,
    accu_model_batch,
    convert_image_to_rgb,
    pad_to_max,
    prob_to_est,
)
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage

from typing import Dict, List, Optional, Tuple

from ..backbone import Backbone, build_backbone
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_camera_head, build_roi_heads
from .build import META_ARCH_REGISTRY
from .pointnet.pointnet_cls import CamHPointNet
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
        horizon_bins_center: torch.tensor,
        pitch_bins_center: torch.tensor,
        vfov_bins_center: torch.tensor,
        yc_bins_center: torch.tensor,
        roll_bins_center: torch.tensor,
        point_net: nn.Module,
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
        self.point_net = point_net
        self.register_buffer("horizon_bins_center", horizon_bins_center)
        self.register_buffer("pitch_bins_center", pitch_bins_center)
        self.register_buffer("vfov_bins_center", vfov_bins_center)
        self.register_buffer("roll_bins_center", roll_bins_center)
        self.register_buffer("yc_bins_center", yc_bins_center)
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
            "roll_bins_center": torch.as_tensor(roll_bins_centers),
            "yc_bins_center": torch.as_tensor(yc_bins_centers),
            "point_net": CamHPointNet(in_channels=9, out_channels=cfg.MODEL.HEIGHT_HEAD.NUM_CLASSES, with_bn=False),
            "reduce_method": cfg.MODEL.HEIGHT_HEAD.REDUCE_METHOD,
            "smooth_l1_beta": cfg.MODEL.HEIGHT_HEAD.SMOOTH_L1_BETA,
        }

    def _get_camera_values(self, cls_logits):
        vfov = prob_to_est(
            cls_logits["vfov_logits"], self.vfov_bins_center, self.reduce_method
        )
        pitch = prob_to_est(
            cls_logits["pitch_logits"], self.pitch_bins_center, self.reduce_method
        )
        roll = prob_to_est(
            cls_logits["roll_logits"], self.roll_bins_center, self.reduce_method
        )
        horizon = prob_to_est(
            cls_logits["horizon_logits"], self.horizon_bins_center, self.reduce_method
        )
        return vfov, pitch, roll, horizon

    def _camrcnn_predictions(self, predicted_proposals: List[Instances], cls_logits):
        valid_mask = [inst for inst in predicted_proposals if inst.pred_height.numel() > 0]
        if len(valid_mask) == 0:
            return {}, {}
        vfov_est, pitch_est, _, _ = self._get_camera_values(cls_logits)
        H_batch = torch.tensor(
            [inst.image_size[0] for inst in predicted_proposals],
            device=self.device
        )  # [B]
        gt_boxes_list = [inst.gt_boxes.tensor for inst in predicted_proposals]
        pred_height_list = [inst.pred_height for inst in predicted_proposals]
        pred_straighten_ratio_list = [inst.pred_straighten_ratio for inst in predicted_proposals]
        MAX_N = 100
        # Padding
        gt_boxes_pad, mask = pad_to_max(gt_boxes_list, self.device, MAX_N)
        pred_height_pad, _ = pad_to_max(pred_height_list, self.device, MAX_N)
        pred_straighten_ratio_pad, _ = pad_to_max(pred_straighten_ratio_list, self.device, MAX_N)

        H = H_batch.unsqueeze(1)

        vfov = vfov_est.unsqueeze(1)
        pitch = pitch_est.unsqueeze(1)

        f_estim = H / torch.tan(vfov / 2.0) / 2.0
        horizon = 0.5 - 0.5 * torch.tan(pitch) / torch.tan(vfov / 2)
        v0_pred = H - horizon * H
        vc = H / 2.0

        y1 = gt_boxes_pad[:, :, 1]
        y2 = gt_boxes_pad[:, :, 3]
        vb = H - (y1 + y2)
        vt = H - y1

        # Discount from predicted keypoints.
        h_human_s = pred_height_pad * pred_straighten_ratio_pad

        # Attempt to predict camera height using PointNet
        bbox_y1y2_offset = gt_boxes_pad[:, :, [1, 3]] - (H - v0_pred).unsqueeze(-1)  # [top 0 , bottom H]
        bbox_concat = torch.cat((gt_boxes_pad, bbox_y1y2_offset), 2) / H.unsqueeze(-1) - 0.05
        vt_01_est = ((H - v0_pred) / H).view(-1, 1, 1).repeat(1, MAX_N, 1)
        input_list = [
            vt_01_est,
            bbox_concat.view(gt_boxes_pad.shape[0], MAX_N, -1),
            (pred_height_pad / self.roi_heads.keypoint_head.height_mean - 1).view(gt_boxes_pad.shape[0], MAX_N, -1),
            (h_human_s / self.roi_heads.keypoint_head.height_mean - 1).view(gt_boxes_pad.shape[0], MAX_N, -1),
        ]
        points = torch.cat(input_list, 2).permute(0, 2, 1).float().to(self.device)
        camH_cls_logits = self.point_net({'points': points})['cls_logit']
        yc_est = prob_to_est(camH_cls_logits, self.yc_bins_center, self.reduce_method)
        geo_model_input_dict = {
            "yc_est": yc_est.unsqueeze(1),
            "vb": vb,
            "y_person": h_human_s * torch.cos(-pitch),
            "v0": v0_pred,
            "vc": vc,
            "f_pixels_est": f_estim,
            "pitch_est": -pitch,
        }
        vt_camEst_batch, _, _ = accu_model_batch(geo_model_input_dict)

        loss = smooth_l1_loss(
            vt,
            vt_camEst_batch,
            beta=self.smooth_l1_beta,
            reduction="none"
        )
        eps = 1e-6
        # Normalize by bbox height
        loss = loss / gt_boxes_pad[:, :, 3].clamp(min=eps)
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        loss = torch.clamp(loss, 0.0, 2.0)
        # Apply mask
        loss = loss * mask
        # Per-instance mean
        vt_loss = (loss.sum(dim=1) / (mask.sum(dim=1) + eps)).mean()
        return {}, {"vt_loss": vt_loss}

    def visualize_training(self, batched_inputs, proposals, cls_logits):
        """Basically the same as the GeneralizedRCNN method"""
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 10
        metadata = MetadataCatalog.get("COCOScale2017Calib_train")

        vfov_est, pitch_est, roll_est, _ = self._get_camera_values(cls_logits)
        for i, (input, prop) in enumerate(zip(batched_inputs, proposals)):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, metadata=metadata)
            v_gt = v_gt.overlay_instances(
                boxes=input["instances"].gt_boxes,
                keypoints=input["instances"].gt_keypoints,
            )
            gt_pitch, gt_vfov, gt_roll = input["pitch"], input["vfov"], input["roll"]
            anno_img = v_gt.get_image()
            anno_img, _ = showHorizonLine(anno_img, gt_vfov, gt_pitch, gt_roll)
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, metadata=None)  # Connections rules is noisy with box_size>1
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy(),
                keypoints=prop.pred_keypoints[0:box_size].detach().cpu().numpy(),
                labels=(prop.pred_height[0: box_size] * prop.pred_straighten_ratio[0: box_size]).detach().cpu().numpy(),
            )
            prop_img = v_pred.get_image()
            prop_img, _ = showHorizonLine(
                prop_img,
                vfov_est[i].detach().cpu().numpy(),
                pitch_est[i].detach().cpu().numpy(),
                roll_est[i].detach().cpu().numpy(),
            )
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT data;  Right: Predict data"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

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
        self.camera_heads.eval()
        with torch.no_grad():
            cls_logits, _ = self.camera_heads(features, _add_whole_image_as_proposal(images, self.device), None)
        self.camera_heads.train()
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, predicted_proposals, cls_logits)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return predicted_proposals, cls_logits, losses

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
        # NOTE: If if filter for valid indices in camrcnn to measure the loss
        # We could do only one forward pass and get the camera predictions since we will need it.
        # NOTE: Doing backbone all together means I need more memory
        # but it's slightly faster than doing it in parts
        features = self.backbone(images.tensor)
        losses = {}
        if dt_inputs:
            dt_image_list_slice = ImageList(
                tensor=images.tensor[: len(dt_inputs)],
                image_sizes=images.image_sizes[: len(dt_inputs)],
            )
            dt_features_slice = {k: v[: len(dt_inputs)] for k, v in features.items()}
            predicted_proposals, cls_logits, dt_losses = self._forward_generalized_rcnn(
                dt_inputs,
                dt_image_list_slice,
                dt_features_slice,
            )
            losses.update(dt_losses)
            _, vt_loss = self._camrcnn_predictions(
                predicted_proposals,
                cls_logits,
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
