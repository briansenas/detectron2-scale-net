import numpy as np
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.pano360 import (
    bins2horizon,
    bins2pitch,
    bins2roll,
    bins2vfov,
    horizon_bins_centers,
    human_bins_layers_list,
    pitch_bins_centers,
    roll_bins_centers,
    showHorizonLine,
    vfov_bins_centers,
    yc_bins_layers_list,
)
from detectron2.data.detection_utils import (
    _add_whole_image_as_proposal,
    _move_logits_to_device,
    accu_model_batch,
    convert_image_to_rgb,
    get_straighten_ratio_from_kps,
    pad_to_max,
    person_h_list_loss,
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
from .pointnet.pointnet_part_seg import CamHPersonHPointNet
from .rcnn import GeneralizedRCNN

__all__ = ["CameraRCNN", "GeneralizedCamRCNN"]

_CAMRCNN_SKIPPED = 0


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
        roll_bins_center: torch.tensor,
        yc_bins_centers_list: Optional[List[torch.tensor]] = None,
        human_height_centers_list: Optional[List[torch.tensor]] = None,
        point_net: Optional[nn.Module] = None,
        point_net_temperature: float = 1.0,
        point_net_detach: bool = True,
        point_net_refine: Optional[nn.Module] = None,
        point_net_refine_layers: Optional[int] = None,
        point_net_refine_temperature: float = 1.0,
        height_loss_weight: Optional[float] = None,
        reduce_method: str = "softmax",
        smooth_l1_beta: float = 0.0,
        input_format: Optional[str] = None,
        padded_input_size: int = 100,
        discount_from: str = "GT",
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
        self.point_net_temperature = point_net_temperature
        self.point_net_detach = point_net_detach
        self.point_net_refine = point_net_refine
        self.point_net_refine_layers = point_net_refine_layers
        self.point_net_refine_temperature = point_net_refine_temperature
        self.height_loss_weight = height_loss_weight
        self.reduce_method = reduce_method
        self.reduce_method = reduce_method
        self.smooth_l1_beta = smooth_l1_beta
        self.padded_input_size = padded_input_size
        self.discount_from = discount_from
        self.register_buffer("horizon_bins_center", horizon_bins_center)
        self.register_buffer("pitch_bins_center", pitch_bins_center)
        self.register_buffer("vfov_bins_center", vfov_bins_center)
        self.register_buffer("roll_bins_center", roll_bins_center)
        self.height_on = self.point_net is not None
        self.height_refine_on = point_net_refine is not None
        if self.height_on:
            self.register_buffer("yc_bins_centers_list", yc_bins_centers_list)
            self.register_buffer("human_height_centers_list", human_height_centers_list)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        ret = {
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
            "reduce_method": cfg.MODEL.HEIGHT_HEAD.REDUCE_METHOD,
            "smooth_l1_beta": cfg.MODEL.HEIGHT_HEAD.SMOOTH_L1_BETA,
        }
        if cfg.MODEL.HEIGHT_ON:
            ret["padded_input_size"] = cfg.MODEL.HEIGHT_HEAD.PADDED_INPUT
            ret["height_loss_weight"] = cfg.MODEL.HEIGHT_HEAD.LOSS_WEIGHT
            ret["yc_bins_centers_list"] = torch.stack(
                [torch.from_numpy(yc_bins_layer).float() for yc_bins_layer in yc_bins_layers_list])
            ret["human_height_centers_list"] = torch.stack(
                [torch.from_numpy(human_bins_layer).float() for human_bins_layer in human_bins_layers_list])
            ret["point_net"] = CamHPointNet(
                in_channels=9,
                out_channels=cfg.MODEL.HEIGHT_HEAD.NUM_CLASSES,
                with_bn=cfg.MODEL.POINT_NET.BN,
                with_transform=cfg.MODEL.POINT_NET.TRANSFORM,
                with_pooling=cfg.MODEL.POINT_NET.POOLING,
            )
            ret["point_net_temperature"] = cfg.MODEL.POINT_NET.TEMPERATURE
            ret["point_net_detach"] = cfg.MODEL.POINT_NET.DETACH
            ret["discount_from"] = cfg.MODEL.HEIGHT_HEAD.DISCOUNT_FROM
            if cfg.MODEL.HEIGHT_REFINE_ON:
                point_net_refine = nn.ModuleDict([])
                point_net_refine_layers = cfg.MODEL.POINT_NET.REFINE_LAYERS
                ret["point_net_refine_layers"] = point_net_refine_layers
                ret["point_net_refine_temperature"] = cfg.MODEL.POINT_NET.REFINE_TEMPERATURE
                for layer_idx in range(point_net_refine_layers):
                    point_net_refine.update(
                        {
                            "point_net_refine_cls_layer_%d"
                            % (layer_idx + 1): CamHPointNet(
                                in_channels=10,
                                out_channels=cfg.MODEL.HEIGHT_HEAD.NUM_CLASSES,
                                with_bn=cfg.MODEL.POINT_NET.BN,
                                with_transform=cfg.MODEL.POINT_NET.TRANSFORM,
                                with_pooling=cfg.MODEL.POINT_NET.POOLING,
                            )
                        }
                    )
                    point_net_refine.update(
                        {
                            "point_net_refine_seg_layer_%d"
                            % (layer_idx + 1): CamHPersonHPointNet(
                                in_channels=10,
                                num_classes_camH=cfg.MODEL.HEIGHT_HEAD.NUM_CLASSES,
                                num_seg_classes=cfg.MODEL.HEIGHT_HEAD.NUM_CLASSES,
                                with_bn=cfg.MODEL.POINT_NET.BN,
                                with_transform=cfg.MODEL.POINT_NET.TRANSFORM,
                                with_pooling=cfg.MODEL.POINT_NET.POOLING,
                                if_cls=False,
                            )
                        }
                    )

                ret["point_net_refine"] = point_net_refine
        return ret

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

    def _camrcnn_predictions(self, predicted_proposals: List[Instances], camrcnn_data):
        valid_mask = torch.as_tensor(
            [inst.pred_height.numel() > 0 for inst in predicted_proposals], dtype=bool, device=self.device)
        if sum(valid_mask) == 0:
            global _CAMRCNN_SKIPPED
            _CAMRCNN_SKIPPED += 1
            storage = get_event_storage()
            storage.put_scalar("camrcnn_num_skipped_batches", _CAMRCNN_SKIPPED, smoothing_hint=False)
            return {}, {}
        camrcnn_data = camrcnn_data or {}
        vfov_est, pitch_est, roll_est, horizon_est = camrcnn_data["vfov_est"], camrcnn_data[
            "pitch_est"], camrcnn_data["roll_est"], camrcnn_data["horizon_est"]
        H_batch = torch.tensor(
            [inst.image_size[0] for inst in predicted_proposals],
            device=self.device
        )  # [B]
        gt_boxes_list = [inst.gt_boxes.tensor if self.training else inst.pred_boxes.tensor for inst in predicted_proposals]
        pred_height_list = [inst.pred_height for inst in predicted_proposals]
        # Padding
        pad_to_size = self.padded_input_size if self.training else max([x.shape[0] for x in gt_boxes_list])
        gt_boxes_pad, _ = pad_to_max(gt_boxes_list, self.device, pad_to_size)
        gt_mask_list = [inst.valid_mask if self.training else torch.ones(
            (len(predicted_proposals)), device=self.device) for inst in predicted_proposals]
        gt_mask_pad, _ = pad_to_max(gt_mask_list, self.device, pad_to_size)
        mask = gt_mask_pad
        pred_height_pad, _ = pad_to_max(pred_height_list, self.device, pad_to_size)
        if self.discount_from == "GT" and self.training:
            # NOTE: for multi-cat I would've to filter for only class 0 (person)
            straighten_ratio_kps_list = [inst.gt_keypoints.tensor for inst in predicted_proposals]
        else:
            # NOTE: for multi-cat I would have to check "if inst.has(...)" and fill the straighten ratios with 1.0
            # But filling would be position dependent so I would need a mask vector.
            straighten_ratio_kps_list = [inst.pred_keypoints for inst in predicted_proposals]

        straighten_discount_ratio, _ = pad_to_max(
            [
                torch.as_tensor(
                    get_straighten_ratio_from_kps(keypoints),
                    device=self.device,
                )
                for keypoints in straighten_ratio_kps_list
            ],
            self.device,
            pad_to_size,
            1.0,
        )

        H = H_batch.unsqueeze(1)

        vfov = vfov_est.unsqueeze(1)
        pitch = pitch_est.unsqueeze(1)

        f_estim = (H / torch.tan(vfov / 2.0) / 2.0)
        horizon = (0.5 - 0.5 * torch.tan(pitch) / torch.tan(vfov / 2))
        v0_pred = (H - horizon * H)
        vc = (H / 2.0)

        y1 = gt_boxes_pad[:, :, 1]
        y2 = gt_boxes_pad[:, :, 3]
        vb = (H - (y1 + y2)) * mask
        vt = (H - y1) * mask

        # Discount from predicted keypoints.
        h_human_s = (pred_height_pad * straighten_discount_ratio) * mask
        eps = 1e-6

        bbox_y1y2_offset = gt_boxes_pad[:, :, [1, 3]] - (H - v0_pred).unsqueeze(-1)  # [top 0 , bottom H]
        bboxes_offset_norm = (
            torch.cat((gt_boxes_pad, bbox_y1y2_offset), 2) / H.unsqueeze(-1) - 0.5
        ).view(gt_boxes_pad.shape[0], pad_to_size, -1) * mask.unsqueeze(2)
        if "yc_est" in camrcnn_data:
            yc_est = camrcnn_data["yc_est"]
        else:
            vt_01_est = ((H - v0_pred) / H).view(-1, 1, 1).repeat(1, pad_to_size, 1)
            person_h_norm = (pred_height_pad / self.roi_heads.height_mean -
                             1).view(gt_boxes_pad.shape[0], pad_to_size, -1)
            person_h_discount_norm = (h_human_s / self.roi_heads.height_mean -
                                      1).view(gt_boxes_pad.shape[0], pad_to_size, -1)
            input_list = [
                vt_01_est,
                bboxes_offset_norm,
                person_h_norm,
                person_h_discount_norm,
            ]
            # NOTE: Start at i>0 if we want to exactly match SVMIW input_list shape and values
            # By multiplying by the mask, we set to 0 values that don't correspond to having predictions
            # Due to the fact that we pad the input to have a rectangular tensor.
            input_list = [x * mask.unsqueeze(2) for _, x in enumerate(input_list)]
            if self.point_net_detach:
                input_list = [x.detach() for x in input_list]
            points = torch.cat(input_list, 2).permute(0, 2, 1).float().to(self.device)
            camH_cls_logits = self.point_net({'points': points, "mask": mask.unsqueeze(1)})['cls_logit']
            yc_est = prob_to_est(camH_cls_logits / self.point_net_temperature,
                                 self.yc_bins_centers_list[0], self.reduce_method).unsqueeze(1)
        geo_model_input_dict = {
            "yc_est": yc_est,
            "vb": vb,
            "y_person": h_human_s * torch.cos(-pitch),
            "v0": v0_pred,
            "vc": vc,
            "f_pixels_est": f_estim,
            "pitch_est": -pitch,
        }
        vt_camEst_batch, _ = accu_model_batch(geo_model_input_dict)
        vt_camEst_batch *= mask
        loss = smooth_l1_loss(
            vt,
            vt_camEst_batch,
            beta=self.smooth_l1_beta,
            reduction="none"
        )
        # Normalize by bbox height
        loss = loss / gt_boxes_pad[:, :, 3].clamp(min=eps)
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        vt_camEst_N = torch.clamp(loss, -2.0, 2.0)
        loss = torch.clamp(loss, 0.0, 2.0)
        # Apply mask
        loss = loss * mask
        vt_camEst_N = vt_camEst_N * mask
        # Per-instance mean
        vt_loss = (loss.sum(dim=1) / (mask.sum(dim=1) + eps)).mean()
        losses = {"vt_loss": vt_loss}
        # with torch.no_grad():
        #     denominator = (vt - vb) * (1.0 + (vc - v0_pred) * (vc - vt) / f_estim ** 2)
        #     yc_implied = h_human_s * (v0_pred - vb) / (denominator + eps)
        # loss_consistency = torch.mean((yc_est.detach() - yc_implied)**2 * mask)
        # losses.update({"consistency_loss": loss_consistency})
        camrcnn_data = {
            "valid_mask": valid_mask,
            "mask": mask,
            "H": H,
            "yc_est": yc_est,
            "yc_est_delta": vt_camEst_batch,
            "v0_pred": v0_pred,
            "vt_camEst_N": vt_camEst_N,
            "person_h": pred_height_pad,
            "straighten_ratio": straighten_discount_ratio,
            "bboxes_offset_norm": bboxes_offset_norm,
            "vfov_est": vfov_est,
            "pitch_est": pitch_est,
            "horizon_est": horizon_est,
            "roll_est": roll_est,
        }
        return camrcnn_data, losses

    def _draw_labels(self, visualizer, texts):
        x = 10  # left padding
        line_height = 15
        padding_bottom = 30
        items = sorted(texts.items())
        n = len(items)

        # Start above left bottom corner
        start_y = visualizer.img.shape[0] - padding_bottom - line_height * (n - 1)
        for i, (k, v) in enumerate(items):
            visualizer.draw_text(
                f"{k}: {v}",
                (x, start_y + i * line_height),
                color="white",
                horizontal_alignment="left",
                font_size=10
            )

        return visualizer

    def visualize_prediction(self, batched_inputs, instances, camrcnn_data: dict):
        from detectron2.utils.visualizer import Visualizer
        max_vis_prop = 10
        vfov_est, pitch_est = camrcnn_data["vfov_est"], camrcnn_data["pitch_est"]
        roll_est, horizon_est = camrcnn_data["roll_est"], camrcnn_data["horizon_est"]
        images = []
        for i, (input, prop) in enumerate(zip(batched_inputs, instances)):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            box_size = min(len(prop.pred_boxes), max_vis_prop)
            v_pred = Visualizer(img, metadata=None)
            v_pred.overlay_instances(
                boxes=prop.pred_boxes[0:box_size].tensor.cpu().numpy(),
                keypoints=prop.pred_keypoints[0:box_size].detach().cpu().numpy() if self.height_on else None,
                labels=(prop.pred_height[0: box_size] * camrcnn_data["straighten_ratio"][i][0:box_size]
                        ).detach().cpu().numpy() if self.height_on else None,
            )
            texts = {}
            texts["vfov"] = vfov_est[i]
            texts["pitch"] = pitch_est[i]
            texts["roll"] = roll_est[i]
            texts["horizon"] = horizon_est[i]
            if self.height_on:
                texts["yc_estCam"] = camrcnn_data["yc_est"][i][0]
            v_pred = self._draw_labels(
                v_pred,
                texts,
            )
            prop_img, _ = showHorizonLine(
                v_pred.get_output().get_image(),
                vfov_est[i].detach().cpu().numpy(),
                -pitch_est[i].detach().cpu().numpy(),
                roll_est[i].detach().cpu().numpy(),
            )
            images.append(prop_img)
        return images

    def visualize_training(self, batched_inputs, proposals, camrcnn_data: Optional[dict] = None):
        """Basically the same as the GeneralizedRCNN method"""
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 10
        metadata = MetadataCatalog.get("COCOScale2017Calib_train")
        if not camrcnn_data:
            return
        vfov_est, pitch_est = camrcnn_data["vfov_est"], camrcnn_data["pitch_est"]
        roll_est, horizon_est = camrcnn_data["roll_est"], camrcnn_data["horizon_est"]
        for i, (input, prop) in enumerate(zip(batched_inputs, proposals)):
            if (
                (not prop.has("pred_boxes") or len(prop.pred_boxes) <= 1) or
                (self.height_on and (not prop.has("pred_keypoints") or prop.pred_keypoints.shape[0] <= 1))
            ):
                continue
            box_size = min(len(prop.pred_boxes), max_vis_prop)
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, metadata=metadata)
            v_gt.overlay_instances(
                boxes=input["instances"].gt_boxes,
                keypoints=input["instances"].gt_keypoints if self.height_on else None,
            )
            gt_pitch, gt_vfov, gt_roll = input["pitch"], input["vfov"], input["roll"]
            v_pred = Visualizer(img, metadata=None)
            v_pred.overlay_instances(
                boxes=prop.pred_boxes[0:box_size].tensor.cpu().numpy(),
                keypoints=prop.pred_keypoints[0:box_size].detach().cpu().numpy() if self.height_on else None,
                labels=(prop.pred_height[0: box_size] * camrcnn_data["straighten_ratio"][i][0:box_size]
                        ).detach().cpu().numpy() if self.height_on else None,
            )
            if camrcnn_data:
                texts = {}
                for col in ["vfov", "pitch", "roll", "horizon"]:
                    texts[col] = input[col]
                if self.height_on:
                    texts["yc_estCam"] = input["yc_estCam"]
                v_gt = self._draw_labels(v_gt, texts)
                texts["vfov"] = vfov_est[i]
                texts["pitch"] = pitch_est[i]
                texts["roll"] = roll_est[i]
                texts["horizon"] = horizon_est[i]
                if self.height_on:
                    texts["yc_estCam"] = camrcnn_data["yc_est"][i][0]
                v_pred = self._draw_labels(
                    v_pred,
                    texts,
                )
            anno_img, _ = showHorizonLine(
                v_gt.get_output().get_image(), gt_vfov, gt_pitch, gt_roll
            )
            prop_img, _ = showHorizonLine(
                v_pred.get_output().get_image(),
                vfov_est[i].detach().cpu().numpy(),
                -pitch_est[i].detach().cpu().numpy(),
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

        proposals, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        del images
        if self.height_on:
            detector_losses["height_loss"] *= self.height_loss_weight

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return proposals, losses

    def _forward_camrcnn(self, batched_inputs: List[Dict[str, torch.Tensor]], features, proposals):
        gt_instances = [
            dict(
                gt_horizon=x["logits"]["gt_horizon"].to(self.device),
                gt_pitch=x["logits"]["gt_pitch"].to(self.device),
                gt_roll=x["logits"]["gt_roll"].to(self.device),
                gt_vfov=x["logits"]["gt_vfov"].to(self.device),
            )
            if "logits" in x else {}
            for x in batched_inputs
        ]
        predictions, detector_losses = self.camera_heads(
            features,
            proposals,
            gt_instances,
        )
        del features
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
        if not isinstance(batched_inputs, List):  # For AspectRatioGroupedMultipleDataset
            batched_inputs = [batched_inputs]
        for input in batched_inputs:
            dt_inputs += input["coco_data"]
            cam_inputs += input["calib_data"]
        cam_inputs = _move_logits_to_device(cam_inputs, self.device)
        return dt_inputs, cam_inputs

    def _camrcnn_refine(self, camrcnn_data):
        mask = camrcnn_data["mask"]
        eps = 1e-6
        losses = []
        v0_01_batch_est = ((camrcnn_data['H'] - camrcnn_data['v0_pred']) /
                           camrcnn_data['H']).view(-1, 1, 1).repeat(1, camrcnn_data["person_h"].shape[1], 1)
        # NOTE: In the original SVMIW he saves intermediate states before each refine to have the vt_loss (due to new yc)
        # as well as, the person_h at every layer
        for layer_idx in range(self.point_net_refine_layers):
            h_human_s = camrcnn_data["person_h"] * camrcnn_data["straighten_ratio"]
            person_h_norm = (camrcnn_data["person_h"] / self.roi_heads.height_mean -
                             1).view(*camrcnn_data["person_h"].shape[:2], -1)
            person_h_discount_norm = (h_human_s / self.roi_heads.height_mean -
                                      1).view(*camrcnn_data["person_h"].shape[:2], -1)
            input_list = [v0_01_batch_est, camrcnn_data["bboxes_offset_norm"],
                          camrcnn_data["yc_est_delta"].unsqueeze(-1), person_h_norm, person_h_discount_norm]
            input_list = [x * mask.unsqueeze(2) for x in input_list]
            if self.point_net_detach:
                input_list = [x.detach() for x in input_list]
            points = torch.cat(input_list, 2).permute(0, 2, 1).float().to(self.device)
            points = points * mask.unsqueeze(1)
            cls_layer_name = "point_net_refine_cls_layer_%d" % (layer_idx + 1)
            point_net_refine_cls_layer_output = self.point_net_refine[cls_layer_name](
                {"points": points, "mask": mask.unsqueeze(1)}
            )
            camH_cls_logits_delta = point_net_refine_cls_layer_output["cls_logit"]
            yc_est_batch_delta = prob_to_est(
                camH_cls_logits_delta / self.point_net_refine_temperature,
                self.yc_bins_centers_list[layer_idx + 1],
                self.reduce_method,
            )
            # Refining our latest prediction of camera height
            camrcnn_data["yc_est"] = camrcnn_data["yc_est"] + yc_est_batch_delta.unsqueeze(1)
            # NOTE: Here we would have to measure the vt_loss once again to have the refine_layer vt_loss
            # But we would have to create a custom AMP / SimpleTrainer that don't sum them to the total loss
            # Unless we don't care about the total loss since we can .detach() it.
            # Refine our personH
            seg_layer_name = "point_net_refine_seg_layer_%d" % (layer_idx + 1)
            point_net_refine_seg_layer_output = self.point_net_refine[seg_layer_name](
                {"points": points, "mask": mask.unsqueeze(1)}
            )
            personH_cls_logits_delta = point_net_refine_seg_layer_output['seg_logit'].permute(0, 2, 1)
            all_person_hs_delta = prob_to_est(
                personH_cls_logits_delta.reshape(
                    -1, personH_cls_logits_delta.shape[-1]
                ) / self.point_net_refine_temperature,
                self.human_height_centers_list[layer_idx + 1],
                self.reduce_method,
            )
            all_person_hs_delta = all_person_hs_delta.reshape(camrcnn_data["person_h"].shape)
            camrcnn_data["person_h"] = camrcnn_data["person_h"] + all_person_hs_delta * mask
            # NOTE: We would have to do the same as before for the person_h layer level loss
            height_loss = (
                person_h_list_loss(
                    camrcnn_data["person_h"],
                    self.roi_heads.height_mean,
                    self.roi_heads.height_std,
                    camrcnn_data["person_h"].shape[1]
                ) *
                mask *
                self.height_loss_weight
            )
            height_loss = (height_loss.sum(dim=1) / (mask.sum(dim=1) + eps)).mean()
            losses.append(height_loss)
        return camrcnn_data, {"height_loss": sum(losses) / len(losses)}

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
        losses = {}
        images = self.preprocess_image(all_inputs)
        features = self.backbone(images.tensor)
        proposals, dt_losses = self._forward_generalized_rcnn(
            dt_inputs,
            ImageList(
                tensor=images.tensor[: len(dt_inputs)],
                image_sizes=images.image_sizes[: len(dt_inputs)],
            ),
            {k: v[: len(dt_inputs)] for k, v in features.items()}
        )
        cam_proposals = _add_whole_image_as_proposal(images, self.device)
        del images
        losses.update(dt_losses)
        cls_logits, cam_losses = self._forward_camrcnn(
            all_inputs,
            features,
            cam_proposals
        )
        del features, cam_proposals
        losses.update(cam_losses)
        dt_logits = {k: v[:len(dt_inputs)].detach() for k, v in cls_logits.items()}
        vfov_est, pitch_est, roll_est, horizon_est = self._get_camera_values(dt_logits)
        camrcnn_data = {"vfov_est": vfov_est, "pitch_est": pitch_est, "roll_est": roll_est, "horizon_est": horizon_est}
        if self.height_on:
            camrcnn_data, vt_loss = self._camrcnn_predictions(
                proposals,
                camrcnn_data,
            )
            losses.update(vt_loss)
            if self.height_refine_on and camrcnn_data:
                camrcnn_data, refine_loss = self._camrcnn_refine(
                    camrcnn_data,
                )
                losses["height_loss"] = (losses["height_loss"] + refine_loss["height_loss"]) / 2.0
                camrcnn_data, vt_loss = self._camrcnn_predictions(
                    proposals,
                    camrcnn_data,
                )
                losses["vt_loss"] = (losses["vt_loss"] + vt_loss["vt_loss"]) / 2.0

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(dt_inputs, proposals, camrcnn_data)
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
            results = [x["instances"] for x in results]
        cam_logits, _ = self.camera_heads(features, _add_whole_image_as_proposal(images, self.device), None)
        vfov_est, pitch_est, roll_est, horizon_est = self._get_camera_values(cam_logits)
        camrcnn_data = {"vfov_est": vfov_est, "pitch_est": pitch_est, "roll_est": roll_est, "horizon_est": horizon_est}
        if self.height_on:
            camrcnn_data, _ = self._camrcnn_predictions(
                results,
                camrcnn_data,
            )
            if self.height_refine_on and camrcnn_data:
                camrcnn_data, _ = self._camrcnn_refine(
                    camrcnn_data,
                )
                camrcnn_data, _ = self._camrcnn_predictions(
                    results,
                    camrcnn_data,
                )
        return results, camrcnn_data

    def visualize_training_camrcnn(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals.
        It shows the predicted camera parameters and the estimated horizon line.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        idx = [i for i, x in enumerate(batched_inputs) if "logits" in x][0]
        input = batched_inputs[idx]
        pitch_logits = proposals["pitch_logits"][idx].detach().cpu().numpy().squeeze()
        roll_logits = proposals["roll_logits"][idx].detach().cpu().numpy().squeeze()
        vfov_logits = proposals["vfov_logits"][idx].detach().cpu().numpy().squeeze()
        horizon_logits = proposals["horizon_logits"][idx].detach().cpu().numpy().squeeze()
        img = input["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        pitch = bins2pitch(pitch_logits)
        roll = bins2roll(roll_logits)
        vfov = bins2vfov(vfov_logits)
        horizon = bins2horizon(horizon_logits)
        gt_pitch = bins2pitch(input["logits"]["gt_pitch"].detach().cpu().numpy().squeeze())
        gt_roll = bins2roll(input["logits"]["gt_roll"].detach().cpu().numpy().squeeze())
        gt_vfov = bins2vfov(input["logits"]["gt_vfov"].detach().cpu().numpy().squeeze())
        gt_horizon = bins2horizon(input["logits"]["gt_horizon"].detach().cpu().numpy().squeeze())
        anno_img, _ = showHorizonLine(img, gt_vfov, gt_pitch, gt_roll)
        prop_img, _ = showHorizonLine(img, vfov, pitch, roll)
        v_gt = Visualizer(anno_img, None)
        v_pred = Visualizer(prop_img, None)
        texts = {}
        texts["pitch"] = gt_pitch
        texts["roll"] = gt_roll
        texts["vfov"] = gt_pitch
        texts["horizon"] = gt_horizon
        v_gt = self._draw_labels(v_gt, texts)
        texts["pitch"] = pitch
        texts["roll"] = roll
        texts["vfov"] = pitch
        texts["horizon"] = horizon
        v_pred = self._draw_labels(v_pred, texts)
        anno_img = v_gt.get_output().get_image()
        prop_img = v_pred.get_output().get_image()
        vis_img = np.concatenate((anno_img, prop_img), axis=1)
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = "Left: GT Horizon;  Right: Predicted Horizon"
        storage.put_image(vis_name, vis_img)
