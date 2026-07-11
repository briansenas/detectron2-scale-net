# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch

from detectron2.evaluation import DatasetEvaluator
from detectron2.modeling.matcher import Matcher
from detectron2.structures import Instances, pairwise_iou
from detectron2.utils import comm


class KittyEvaluator(DatasetEvaluator):
    def __init__(self, iou_thresh: float = 0.5):
        self.iou_thresh = iou_thresh
        self.matcher = Matcher(
            [self.iou_thresh],
            [0, 1],
            allow_low_quality_matches=False,
        )

    def reset(self):
        self.height_error_sum = 0.0
        self.num_matches = 0
        self.num_samples = 0
        self.num_empty_samples = 0
        self.num_invalid_samples = 0
        self.vt_loss = 0

    def process(self, inputs, outputs):
        outputs, camrcnn_data = outputs
        for inp, out in zip(inputs, outputs):
            gt = inp["instances"].to("cpu")
            pred = out.to("cpu")
            # We need to resize the gt boxes using the different in image size
            # Following the implementation of detector_postprocess
            # Due to .inference(...do_postprocess=True)
            gt_height, gt_width = gt.image_size
            pred_height, pred_width = pred.image_size
            scale_x, scale_y = (
                pred_width / gt_width,
                pred_height / gt_height,
            )
            new_size = (pred_height, pred_width)
            results = Instances(new_size, **gt.get_fields())
            gt_boxes = results.gt_boxes
            gt_boxes.scale(scale_x, scale_y)
            gt_boxes.clip(results.image_size)

            if len(results) == 0 or len(pred) == 0:
                self.num_empty_samples += 1
                continue

            iou_matrix = pairwise_iou(
                results.gt_boxes,
                pred.pred_boxes,
            )

            matched_idxs, labels = self.matcher(iou_matrix)

            valid_mask = labels == 1
            if valid_mask.sum() == 0:
                self.num_invalid_samples += 1
                continue

            gt_heights = results.object_height[
                matched_idxs[valid_mask]
            ]

            pred_heights = pred.pred_height[
                valid_mask
            ]

            errors = torch.abs(
                pred_heights.float() -
                gt_heights.float()
            )

            self.height_error_sum += errors.sum().item()
            self.num_matches += len(errors)
        if "vt_losses" in camrcnn_data:
            vt_losses = torch.as_tensor(camrcnn_data["vt_losses"])
            self.vt_loss += vt_losses[-1].item()
            self.num_samples += 1

    def evaluate(self):
        stats = {
            "height_error_sum": self.height_error_sum,
            "num_matches": self.num_matches,
            "vt_loss": self.vt_loss,
            "num_samples": self.num_samples,
            "num_invalid_samples": self.num_invalid_samples,
            "num_empty_samples": self.num_empty_samples,
        }

        all_stats = comm.gather(stats, dst=0)

        if not comm.is_main_process():
            return {}

        total_error = sum(
            x["height_error_sum"]
            for x in all_stats
        )

        total_matches = sum(
            x["num_matches"]
            for x in all_stats
        )
        total_samples = sum(
            x["num_samples"]
            for x in all_stats
        )
        total_vt = sum(
            x["vt_loss"]
            for x in all_stats
        )
        total_invalid_samples = sum(
            x["num_invalid_samples"]
            for x in all_stats
        )
        total_empty_samples = sum(
            x["num_empty_samples"]
            for x in all_stats
        )

        if total_matches == 0:
            return {
                "kitty_height_mae": float("nan"),
                "kitty_num_matches": 0,
            }

        mae = total_error / total_matches

        return {
            "kitty_height_mae": mae,
            "kitty_num_matches": total_matches,
            "kitty_num_samples": total_samples,
            "kitty_vt_loss_mean": total_vt / total_samples,
            "kitty_num_invalid_samples": total_invalid_samples,
            "kitty_num_empty_samples": total_empty_samples,
        }
