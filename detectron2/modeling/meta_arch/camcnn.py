import numpy as np
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn

from detectron2.config import configurable
from detectron2.data.datasets.pano360 import (
    COCO_SCALE_STATS,
    convert_preds_to_angles,
    get_softargmax,
    pitch2soft_idx,
    pitch_bins,
    pitch_bins_centers,
    roll2soft_idx,
    roll_bins_centers,
    showHorizonLine,
    soft_idx_to_angle,
    vfov2soft_idx,
    vfov_bins,
    vfov_bins_centers,
    yc_bins_layers_list,
)
from detectron2.data.detection_utils import (
    _add_whole_image_as_proposal,
    accu_model_batch,
    convert_image_to_rgb,
    person_h_list_loss_masked,
    prob_to_est,
)
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage

from typing import Dict, Iterable, List, Optional, Tuple

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
        self.loss_criterion = self.camera_heads.loss_criterion

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
            gt_instances = prepare_camrcnn_gt_instances(batched_inputs, criterion=self.camera_heads.loss_criterion)
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
        # batched_inputs = _move_logits_to_device(batched_inputs, self.device)
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals = _add_whole_image_as_proposal(images, self.device)
        results, _ = self.camera_heads(features, proposals, None)
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
        from detectron2.utils.visualizer import Visualizer
        storage = get_event_storage()
        input = batched_inputs[0]
        pitch_logits = proposals["pitch_logits"][:1].detach().float().cpu()
        roll_logits = proposals["roll_logits"][:1].detach().float().cpu()
        vfov_logits = proposals["vfov_logits"][:1].detach().float().cpu()
        vfov, pitch, roll = convert_preds_to_angles(
            vfov_logits,
            pitch_logits,
            roll_logits,
            loss_type=self.loss_criterion,
            return_type="np",
        )
        img = input["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        gt_vfov, gt_pitch, gt_roll = input["vfov"], input["pitch"], input["roll"]
        anno_img, _ = showHorizonLine(img, gt_vfov, gt_pitch, gt_roll)
        prop_img, _ = showHorizonLine(img, vfov, pitch, roll)
        v_gt = Visualizer(anno_img, None)
        v_pred = Visualizer(prop_img, None)
        texts = {}
        texts["pitch"] = gt_pitch
        texts["roll"] = gt_roll
        texts["vfov"] = gt_vfov
        v_gt = draw_labels(v_gt, texts)
        texts["pitch"] = pitch
        texts["roll"] = roll
        texts["vfov"] = vfov
        v_pred = draw_labels(v_pred, texts)
        anno_img = v_gt.get_output().get_image()
        prop_img = v_pred.get_output().get_image()
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
        pitch_bins_center: torch.tensor,
        vfov_bins_center: torch.tensor,
        roll_bins_center: torch.tensor,
        yc_bins_centers_list: Optional[List[torch.tensor]] = None,
        class_ids_to_idx: Optional[Dict] = None,
        class_height_centers_list: Optional[List[torch.tensor]] = None,
        height_on: Optional[bool] = False,
        point_net: Optional[nn.Module] = None,
        point_net_temperature: float = 1.0,
        point_net_detach: bool = True,
        point_net_refine: Optional[nn.Module] = None,
        point_net_refine_layers: Optional[int] = None,
        point_net_refine_temperature: float = 1.0,
        smooth_l1_beta: float = 0.0,
        input_format: Optional[str] = None,
        padded_input_size: int = 100,
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
        self.smooth_l1_beta = smooth_l1_beta
        self.padded_input_size = padded_input_size
        self.register_buffer("pitch_bins_center", pitch_bins_center)
        self.register_buffer("vfov_bins_center", vfov_bins_center)
        self.register_buffer("roll_bins_center", roll_bins_center)
        self.height_on = height_on
        self.point_net_on = self.point_net is not None
        self.height_refine_on = point_net_refine is not None
        if self.height_on:
            self.register_buffer("yc_bins_centers_list", yc_bins_centers_list)
            self.register_buffer("class_height_centers_list", class_height_centers_list)
            self.class_ids_to_idx = class_ids_to_idx

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
            "pitch_bins_center": torch.as_tensor(pitch_bins_centers, dtype=torch.float16),
            "vfov_bins_center": torch.as_tensor(vfov_bins_centers, dtype=torch.float16),
            "roll_bins_center": torch.as_tensor(roll_bins_centers, dtype=torch.float16),
            "smooth_l1_beta": cfg.MODEL.HEIGHT_HEAD.SMOOTH_L1_BETA,
        }
        ret["height_on"] = cfg.MODEL.HEIGHT_ON
        if cfg.MODEL.HEIGHT_ON:
            ret["padded_input_size"] = cfg.MODEL.HEIGHT_HEAD.PADDED_INPUT
            ret["yc_bins_centers_list"] = torch.stack(
                [torch.from_numpy(yc_bins_layer).to(torch.float16) for yc_bins_layer in yc_bins_layers_list])
            all_class_bins = []
            ids_to_idx = {}
            for i, entry in enumerate(COCO_SCALE_STATS):
                ids_to_idx[entry["id"]] = i
                class_layer_stack = torch.stack([torch.as_tensor(l, dtype=torch.float16) for l in entry["layer_list"]])
                all_class_bins.append(class_layer_stack)
            ids_to_idx[-1] = -1
            ret["class_ids_to_idx"] = ids_to_idx
            ret["class_height_centers_list"] = torch.stack(all_class_bins)
            if cfg.MODEL.POINT_NET_ON:
                ret["point_net"] = CamHPointNet(
                    in_channels=8,
                    out_channels=cfg.MODEL.HEIGHT_HEAD.NUM_CLASSES,
                    with_bn=cfg.MODEL.POINT_NET.BN,
                    with_transform=cfg.MODEL.POINT_NET.TRANSFORM,
                    with_pooling=cfg.MODEL.POINT_NET.POOLING,
                )
                ret["point_net_temperature"] = cfg.MODEL.POINT_NET.TEMPERATURE
                ret["point_net_detach"] = cfg.MODEL.POINT_NET.DETACH
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
                                in_channels=9,
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
                                in_channels=9,
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
        if self.camera_heads.loss_criterion in ('kl', 'ce'):
            vfov = prob_to_est(
                cls_logits["vfov_logits"], self.vfov_bins_center, self.training,
            )
            pitch = prob_to_est(
                cls_logits["pitch_logits"], self.pitch_bins_center, self.training,
            )
            roll = prob_to_est(
                cls_logits["roll_logits"], self.roll_bins_center, self.training,
            )
        else:
            vfov = soft_idx_to_angle(get_softargmax(cls_logits["vfov_logits"]),
                                     min=np.min(vfov_bins), max=np.max(vfov_bins))
            pitch = soft_idx_to_angle(get_softargmax(cls_logits["pitch_logits"]),
                                      min=np.min(pitch_bins), max=np.max(pitch_bins))
            roll = soft_idx_to_angle(get_softargmax(cls_logits["roll_logits"]), min=-0.6, max=0.6)
        return vfov, pitch, roll

    def _call_accu_model(self, camrcnn_data: dict, grad_detach: bool = False):
        """
        Modifies camrcnn_data to add the reprojection loss vt_camEst_N
        Expects a dict with all the necessary keys, read code below haha
        The accu_model_batch expects the pitch without the - sign
        """
        # NOTE: This function should not belong to this class but rather be a function call such as fast_rcnn_inference
        # Calculate some variables:
        if not grad_detach:
            # When this function is called, it should be after _camrcnn which detaches pitch, v0 and f_p
            camrcnn_data["pitch"] = camrcnn_data["pitch_est"]
            camrcnn_data["f_pixels_est"] = camrcnn_data["f_estim"]
            camrcnn_data["v0"] = camrcnn_data["v0_pred"]
        # Read some variables
        vt = camrcnn_data["vt"]
        mask = camrcnn_data["mask"]
        gt_boxes_pad = camrcnn_data["gt_boxes_pad"]
        eps = camrcnn_data["eps"]
        # call accu model
        vt_camEst_batch, _ = accu_model_batch(camrcnn_data)
        # To avoid NaNs / inf in huber loss
        vt_camEst_batch = torch.where(torch.isnan(vt_camEst_batch), torch.zeros_like(vt_camEst_batch), vt_camEst_batch)
        vt_camEst_batch = vt_camEst_batch * mask
        loss = smooth_l1_loss(
            vt,
            vt_camEst_batch,
            beta=self.smooth_l1_beta,
            reduction="none"
        )
        # Normalize by bbox height
        loss = loss * mask
        loss = loss / gt_boxes_pad[:, :, 3].clamp(min=eps)
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        vt_camEst_N = torch.clamp(loss, -2.0, 2.0)
        loss = torch.clamp(loss, 0.0, 2.0)
        # Per-instance masked mean
        vt_loss = (loss.sum(dim=1) / (mask.sum(dim=1) + eps)).mean()
        # Store the projection loss for refine heads.
        camrcnn_data["vt_camEst_N"] = vt_camEst_N
        camrcnn_data["vt_loss"] = vt_loss
        return vt_loss

    def _pad_to_max_camrcnn(self, predicted_proposals):
        B = len(predicted_proposals)
        if self.training:
            # Use at most padded_input_sizes predictions
            M = min(max([inst.gt_boxes.tensor.shape[0] for inst in predicted_proposals]), self.padded_input_size)
            box_shape = predicted_proposals[0].gt_boxes.tensor.shape[1:]
            dtype_ = predicted_proposals[0].gt_boxes.tensor.dtype
        else:
            M = max([inst.pred_boxes.tensor.shape[0] for inst in predicted_proposals])
            box_shape = predicted_proposals[0].pred_boxes.tensor.shape[1:]
            dtype_ = predicted_proposals[0].pred_boxes.tensor.dtype
        # infer shapes
        gt_boxes_pad = torch.zeros(
            (B, M, *box_shape),
            device=self.device,
            dtype=dtype_,
        )
        gt_mask_pad = torch.zeros(
            (B, M),
            device=self.device,
            dtype=torch.bool,
        )
        gt_classes_pad = torch.zeros(
            (B, M),
            device=self.device,
            dtype=torch.int,
        )
        pred_height_pad = torch.zeros(
            (B, M),
            device=self.device,
            dtype=dtype_,
        )
        H_batch = torch.zeros(
            (B),
            device=self.device,
            dtype=dtype_,
        )
        for i, inst in enumerate(predicted_proposals):
            if self.training:
                boxes = inst.gt_boxes.tensor
                n = min(len(boxes), M)
                gt_classes_pad[i, :n] = inst.gt_classes[:n]
            else:
                boxes = inst.pred_boxes.tensor
                n = min(len(boxes), M)
                gt_classes_pad[i, :n] = inst.pred_classes[:n]
            gt_mask_pad[i, :n] = True
            gt_boxes_pad[i, :n] = boxes[:n]
            gt_classes_pad[i, n:] += -1
            pred_height_pad[i, :n] = inst.pred_height[:n]
            H_batch[i] = inst.image_size[0]
        gt_classes_mapped_idx = torch.tensor(
            [self.class_ids_to_idx[int(v)] for c in gt_classes_pad for v in c],
            device=gt_classes_pad.device,
            dtype=torch.int,
        ).view(gt_classes_pad.shape)
        return H_batch, gt_boxes_pad, gt_mask_pad, gt_classes_pad, gt_classes_mapped_idx, pred_height_pad

    def _camrcnn_predictions(self, predicted_proposals: List[Instances], camrcnn_data, grad_detach: bool = False):
        vfov_est, pitch_est, roll_est = camrcnn_data["vfov_est"], camrcnn_data[
            "pitch_est"], camrcnn_data["roll_est"]
        H_batch, gt_boxes_pad, mask, gt_classes_pad, gt_classes_mapped_idx, pred_height_pad = self._pad_to_max_camrcnn(
            predicted_proposals)
        pad_to_size = mask.shape[-1]
        # NOTE: No pose discount in multiple classes
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
        eps = torch.finfo(pred_height_pad.dtype).eps
        bbox_y1y2_offset = gt_boxes_pad[:, :, [1, 3]] - (H - v0_pred).unsqueeze(-1)  # [top 0 , bottom H]
        bboxes_offset_norm = (
            torch.cat((gt_boxes_pad, bbox_y1y2_offset), 2) / H.unsqueeze(-1) - 0.5
        ).view(gt_boxes_pad.shape[0], pad_to_size, -1) * mask.unsqueeze(2)
        v0_01_est = ((H - v0_pred) / H).view(-1, 1, 1).repeat(1, pad_to_size, 1)
        if self.point_net_on:
            person_h_norm = (pred_height_pad / self.roi_heads.class_means[gt_classes_mapped_idx] -
                             1).view(gt_boxes_pad.shape[0], pad_to_size, -1)
            input_list = [
                v0_01_est,
                bboxes_offset_norm,
                person_h_norm,
            ]
            # NOTE: Start at i>0 if we want to exactly match SVMIW input_list shape and values
            # By multiplying by the mask, we set to 0 values that don't correspond to having predictions
            # Due to the fact that we pad the input to have a rectangular tensor.
            if self.point_net_detach:
                input_list = [x.detach() for x in input_list]
            points = (torch.cat(input_list, 2) * mask.unsqueeze(-1)).permute(0, 2, 1)
            camH_cls_logits = self.point_net({'points': points, "mask": mask.unsqueeze(1)})['cls_logit']
            yc_est = prob_to_est(camH_cls_logits / self.point_net_temperature,
                                 self.yc_bins_centers_list[0], self.training).unsqueeze(1)
        else:
            yc_est = camrcnn_data["yc_est"]
        camrcnn_data = {
            "mask": mask,
            "H": H,
            # Store point net
            "v0_01_est": v0_01_est,
            "pred_height": pred_height_pad,
            "bboxes_offset_norm": bboxes_offset_norm,
            # Store gt_data for losses
            "gt_classes_pad": gt_classes_pad,
            "gt_classes_mapped_idx": gt_classes_mapped_idx,
            "gt_boxes_pad": gt_boxes_pad,
            # Store accu model variables
            "yc_est": yc_est,
            "v0_pred": v0_pred,
            "vt": vt,
            "vb": vb,
            "vc": vc,
            "f_estim": f_estim,
            "horizon": horizon,
            # Store logits
            "vfov_est": vfov,
            "pitch_est": pitch,
            "pitch": pitch.clone().detach() if grad_detach else pitch,
            "v0": v0_pred.clone().detach() if grad_detach else v0_pred,
            "f_pixels_est": f_estim.clone().detach() if grad_detach else f_estim,
            # Copy so that we can detach sometimes
            "roll_est": roll_est,
            "eps": eps,
        }
        losses = {}
        losses["vt_loss"] = self._call_accu_model(camrcnn_data, grad_detach)
        # fit_derek (fit_camH) from Rui Zhu code
        # with torch.no_grad():
        #     denominator = (vt - vb) * (1.0 + (vc - v0_pred) * (vc - vt) / f_estim ** 2)
        #     yc_implied = pred_height_pad * (v0_pred - vb) / (denominator + eps)
        # loss_consistency = torch.mean((yc_est.detach() - yc_implied)**2 * mask)
        # losses.update({"consistency_loss": loss_consistency})
        return camrcnn_data, losses

    def visualize_prediction(self, batched_inputs, instances, camrcnn_data: dict):
        from detectron2.utils.visualizer import Visualizer
        max_vis_prop = 10
        vfov_est, pitch_est = camrcnn_data["vfov_est"], camrcnn_data["pitch_est"]
        roll_est = camrcnn_data["roll_est"]
        images = []
        for i, (input, prop) in enumerate(zip(batched_inputs, instances)):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            box_size = min(len(prop.pred_boxes), max_vis_prop)
            v_pred = Visualizer(img, metadata=None)
            v_pred.overlay_instances(
                boxes=prop.pred_boxes[0:box_size].tensor.cpu().numpy(),
                labels=(list(map("{:.4f}".format, prop.pred_height[0: box_size])) if self.height_on else None),
            )
            texts = {}
            texts["vfov"] = vfov_est[i].detach().cpu().numpy()[0]
            texts["pitch"] = pitch_est[i].detach().cpu().numpy()[0]
            texts["roll"] = roll_est[i].detach().cpu().numpy()
            if self.height_on:
                texts["yc_estCam"] = camrcnn_data["yc_est"][i][0].detach().cpu().numpy()
            v_pred = draw_labels(
                v_pred,
                texts,
            )
            prop_img, _ = showHorizonLine(
                v_pred.get_output().get_image(),
                texts["vfov"],
                texts["pitch"],
                texts["roll"],
            )
            images.append(prop_img)
        return images

    def visualize_training(self, batched_inputs, proposals, target_proposals, camrcnn_data: Optional[dict] = None):
        """Basically the same as the GeneralizedRCNN method"""
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 10
        super().visualize_training(batched_inputs, proposals)
        if not camrcnn_data:
            return
        vfov_est, pitch_est = camrcnn_data["vfov_est"], camrcnn_data["pitch_est"]
        roll_est = camrcnn_data["roll_est"]
        for i, (input, prop) in enumerate(zip(batched_inputs, target_proposals)):
            box_size = min(len(prop.gt_boxes), max_vis_prop)
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, metadata=None)
            v_gt.overlay_instances(
                boxes=input["instances"].gt_boxes,
            )
            if "pitch" in input:
                texts = {}
                for col in ["vfov", "pitch", "roll"]:
                    texts[col] = input[col]
                if self.height_on:
                    texts["yc_estCam"] = input["yc_estCam"]
                v_gt = draw_labels(v_gt, texts)
                gt_pitch, gt_vfov, gt_roll = input["pitch"], input["vfov"], input["roll"]
                anno_img, _ = showHorizonLine(
                    v_gt.get_output().get_image(), gt_vfov, gt_pitch, gt_roll
                )
            else:
                anno_img = v_gt.get_output().get_image()
            v_pred = Visualizer(img, metadata=None)
            v_pred.overlay_instances(
                boxes=prop.gt_boxes[0:box_size].tensor.cpu().numpy(),
                labels=(list(map("{:.4f}".format, prop.pred_height[0: box_size])) if self.height_on else None),
            )
            if camrcnn_data:
                texts = {}
                texts["vfov"] = vfov_est[i].detach().cpu().numpy()[0]
                texts["pitch"] = pitch_est[i].detach().cpu().numpy()[0]
                texts["roll"] = roll_est[i].detach().cpu().numpy()
                if self.height_on:
                    texts["yc_estCam"] = camrcnn_data["yc_est"][i][0].detach().cpu().numpy()
                v_pred = draw_labels(
                    v_pred,
                    texts,
                )
            prop_img, _ = showHorizonLine(
                v_pred.get_output().get_image(),
                texts["vfov"],
                texts["pitch"],
                texts["roll"],
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

        proposals, target_proposals, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        del images

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return proposals, target_proposals, detector_losses

    def _forward_camrcnn(self, batched_inputs: List[Dict[str, torch.Tensor]], features, proposals):
        gt_instances = prepare_camrcnn_gt_instances(batched_inputs, criterion=self.camera_heads.loss_criterion)
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
            dt_inputs += input["scale_data"]
            cam_inputs += input["calib_data"]
        # cam_inputs = _move_logits_to_device(cam_inputs, self.device)
        return dt_inputs, cam_inputs

    def _camrcnn_refine(self, camrcnn_data):
        mask = camrcnn_data["mask"]
        gt_classes_mapped_idx = camrcnn_data["gt_classes_mapped_idx"]
        h_losses = []
        vt_losses = []
        # NOTE: In the original SVMIW he saves intermediate states before each refine to have the vt_loss (due to new yc)
        # as well as, the pred_height at every layer
        for layer_idx in range(self.point_net_refine_layers):
            is_last_layer = layer_idx + 1 == self.point_net_refine_layers
            pred_height_norm = (camrcnn_data["pred_height"] / self.roi_heads.class_means[gt_classes_mapped_idx] -
                                1).view(*camrcnn_data["pred_height"].shape[:2], -1)
            input_list = [camrcnn_data["v0_01_est"], camrcnn_data["bboxes_offset_norm"],
                          camrcnn_data["vt_camEst_N"].unsqueeze(-1), pred_height_norm]
            if self.point_net_detach:
                input_list = [x.detach() for x in input_list]
            points = (torch.cat(input_list, 2) * mask.unsqueeze(-1)).permute(0, 2, 1)
            cls_layer_name = "point_net_refine_cls_layer_%d" % (layer_idx + 1)
            point_net_refine_cls_layer_output = self.point_net_refine[cls_layer_name](
                {"points": points, "mask": mask.unsqueeze(1)}
            )
            camH_cls_logits_delta = point_net_refine_cls_layer_output["cls_logit"]
            yc_est_batch_delta = prob_to_est(
                camH_cls_logits_delta / self.point_net_refine_temperature,
                self.yc_bins_centers_list[layer_idx + 1],
                self.training,
            )
            # Refining our latest prediction of camera height
            camrcnn_data["yc_est"] = camrcnn_data["yc_est"] + yc_est_batch_delta.unsqueeze(1)
            # Refine our personH
            seg_layer_name = "point_net_refine_seg_layer_%d" % (layer_idx + 1)
            point_net_refine_seg_layer_output = self.point_net_refine[seg_layer_name](
                {"points": points, "mask": mask.unsqueeze(1)}
            )
            personH_cls_logits_delta = point_net_refine_seg_layer_output['seg_logit'].permute(0, 2, 1)
            all_pred_heights_delta = prob_to_est(
                personH_cls_logits_delta.reshape(
                    -1, personH_cls_logits_delta.shape[-1]
                ) / self.point_net_refine_temperature,
                self.class_height_centers_list[gt_classes_mapped_idx, layer_idx +
                                               1].reshape(-1, personH_cls_logits_delta.shape[-1]),
                self.training,
            )
            all_pred_heights_delta = all_pred_heights_delta.reshape(camrcnn_data["pred_height"].shape)
            camrcnn_data["pred_height"] = camrcnn_data["pred_height"] + all_pred_heights_delta * mask
            # New reprojection loss for next layer
            vt_loss = self._call_accu_model(
                # Camrcnn_data is modified in place
                camrcnn_data,
                grad_detach=not is_last_layer,
            )
            vt_losses.append(vt_loss)
            # NOTE: We would have to do the same as before for the person_h layer level loss
            height_loss = (
                person_h_list_loss_masked(
                    camrcnn_data["pred_height"],
                    self.roi_heads.class_means[gt_classes_mapped_idx],
                    self.roi_heads.class_stds[gt_classes_mapped_idx],
                    mask,
                ) *
                self.roi_heads.height_loss_weight
            )
            h_losses.append(height_loss)
        # Since d2 sum all losses by default in DefaultTrainer we return the average layer loss.
        # We could also return only the last layer loss or implement custom loss management
        return camrcnn_data, {"height_losses": h_losses, "vt_losses": vt_losses}

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
        proposals, target_proposals, dt_losses = self._forward_generalized_rcnn(
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
        dt_logits = {k: v[:len(dt_inputs)] for k, v in cls_logits.items()}
        vfov_est, pitch_est, roll_est = self._get_camera_values(dt_logits)
        camrcnn_data = {"vfov_est": vfov_est, "pitch_est": pitch_est, "roll_est": roll_est}
        if not self.point_net_on and self.height_on:
            camera_height_key = "camera_height"
            camrcnn_data["yc_est"] = torch.as_tensor(
                [x[camera_height_key] for x in dt_inputs], dtype=vfov_est.dtype, device=vfov_est.device).unsqueeze(1)
        if self.height_on:
            camrcnn_data, vt_loss = self._camrcnn_predictions(
                target_proposals,
                camrcnn_data,
                self.height_refine_on,
            )
            losses.update(vt_loss)
            if self.height_refine_on and camrcnn_data:
                camrcnn_data, refine_loss = self._camrcnn_refine(
                    camrcnn_data,
                )
                losses["height_loss"] = (losses["height_loss"] + sum(refine_loss["height_losses"])
                                         ) / (len(refine_loss["height_losses"]) + 1)
                losses["vt_loss"] = (losses["vt_loss"] + sum(refine_loss["vt_losses"])) / \
                    (len(refine_loss["vt_losses"]) + 1)
                # Either assign new values in Instances object or use camrcnn_data downstream
                for m, h, prop in zip(camrcnn_data["mask"], camrcnn_data["pred_height"], target_proposals):
                    prop.pred_height = h[:m.sum()]

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(dt_inputs, proposals, target_proposals, camrcnn_data)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
        camera_data: bool = False,
    ):
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        if all([x["source"] == "pano360" for x in batched_inputs]):
            cam_logits, _ = self.camera_heads(features, _add_whole_image_as_proposal(images, self.device), None)
            return [], cam_logits

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
        # NOTE: Should only skip the problematic
        empty_set = [x for x in results if len(x) <= 0]
        non_empty_set = [x for x in results if len(x) >= 1]
        camrcnn_data = {}
        if self.height_on or camera_data:
            cam_logits, _ = self.camera_heads(features, _add_whole_image_as_proposal(images, self.device), None)
            vfov_est, pitch_est, roll_est = self._get_camera_values(cam_logits)
            camrcnn_data = {"vfov_est": vfov_est, "pitch_est": pitch_est, "roll_est": roll_est, **cam_logits}
            if non_empty_set and self.height_on:
                if not self.point_net_on:
                    camera_height_key = "camera_height"
                    camrcnn_data["yc_est"] = torch.as_tensor(
                        [x[camera_height_key] for x in batched_inputs], dtype=vfov_est.dtype, device=vfov_est.device).unsqueeze(1)
                camrcnn_data, _ = self._camrcnn_predictions(
                    non_empty_set,
                    camrcnn_data,
                )
                if self.height_refine_on and camrcnn_data:
                    camrcnn_data, _ = self._camrcnn_refine(
                        camrcnn_data,
                    )
                    # Either assign new values in Instances object or use camrcnn_data downstream
                    for m, h, prop in zip(camrcnn_data["mask"], camrcnn_data["pred_height"], results):
                        prop.pred_height = h[:m.sum()]
        return empty_set + non_empty_set, camrcnn_data

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
        pitch_logits = proposals["pitch_logits"][:1].detach().float().cpu()
        roll_logits = proposals["roll_logits"][:1].detach().float().cpu()
        vfov_logits = proposals["vfov_logits"][:1].detach().float().cpu()
        vfov, pitch, roll = convert_preds_to_angles(
            vfov_logits,
            pitch_logits,
            roll_logits,
            loss_type=self.camera_heads.loss_criterion,
            return_type="np",
        )
        img = input["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        gt_vfov, gt_pitch, gt_roll = input["vfov"], input["pitch"], input["roll"]
        anno_img, _ = showHorizonLine(img, gt_vfov, gt_pitch, gt_roll)
        prop_img, _ = showHorizonLine(img, vfov, pitch, roll)
        v_gt = Visualizer(anno_img, None)
        v_pred = Visualizer(prop_img, None)
        texts = {}
        texts["pitch"] = gt_pitch
        texts["roll"] = gt_roll
        texts["vfov"] = gt_vfov
        v_gt = draw_labels(v_gt, texts)
        texts["pitch"] = pitch
        texts["roll"] = roll
        texts["vfov"] = vfov
        v_pred = draw_labels(v_pred, texts)
        anno_img = v_gt.get_output().get_image()
        prop_img = v_pred.get_output().get_image()
        vis_img = np.concatenate((anno_img, prop_img), axis=1)
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = "Left: GT Horizon;  Right: Predicted Horizon"
        storage.put_image(vis_name, vis_img)


def draw_labels(visualizer, texts):
    x = 10  # left padding
    line_height = 15
    padding_bottom = 30
    items = sorted(texts.items())
    n = len(items)

    # Start above left bottom corner
    start_y = visualizer.img.shape[0] - padding_bottom - line_height * (n - 1)
    for i, (k, v) in enumerate(items):
        if isinstance(v, Iterable):
            v = v[0]
        visualizer.draw_text(
            f"{k}: {v:.4f}",
            (x, start_y + i * line_height),
            color="white",
            horizontal_alignment="left",
            font_size=10
        )

    return visualizer


def prepare_camrcnn_gt_instances(batched_inputs: List[Dict[str, torch.Tensor]], criterion: str = "kl"):
    if criterion in ("kl", "ce"):
        gt_instances = [
            dict(
                gt_pitch=x["logits"]["gt_pitch"],
                gt_roll=x["logits"]["gt_roll"],
                gt_vfov=x["logits"]["gt_vfov"],
            )
            if "logits" in x else {}
            for x in batched_inputs
        ]
    else:
        gt_instances = [
            dict(
                gt_pitch=pitch2soft_idx(x["pitch"]),
                gt_roll=roll2soft_idx(x["roll"]),
                gt_vfov=vfov2soft_idx(x["vfov"]),
            )
            if "pitch" in x else {}
            for x in batched_inputs
        ]
    return gt_instances
