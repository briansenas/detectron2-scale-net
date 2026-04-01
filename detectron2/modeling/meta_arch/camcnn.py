from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw
from torch import nn

from ..backbone import Backbone
from ..backbone import build_backbone
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
from detectron2.config import configurable
from detectron2.layers import move_device_like
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage

__all__ = ["ClassifierRCNN"]


@META_ARCH_REGISTRY.register()
class ClassifierRCNN(nn.Module):
    """"""

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
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
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

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
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
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
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(
                images,
                features,
                gt_instances,
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

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
        predictions, detector_losses = self.roi_heads(
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

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        results, _ = self.roi_heads(images, features, None)
        return results

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

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals.
        It shows the predicted camera parameters and the estimated horizon line.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        raise NotImplementedError
        # from detectron2.utils.visualizer import Visualizer

        # storage = get_event_storage()

        # for input, prop in zip(batched_inputs, proposals):
        #     img = input["image"]
        #     img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        # NOTE: Do something to transform the stored bins into values to the function.
        # He uses the bins2<pred> and a vfov operation
        # pitch = prop["pitch_logits"]
        # roll = prop["roll_logits"]
        # vfov = prop["vfov_logits"]
        # img, _ = self.showHorizonLine(img, vfov, pitch, roll)
        # vis_img = np.concatenate((anno_img, prop_img), axis=1)
        # vis_img = vis_img.transpose(2, 0, 1)
        # vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
        # storage.put_image(vis_name, vis_img)
        # break  # only visualize one image in a batch
