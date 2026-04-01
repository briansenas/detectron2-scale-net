# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn

from .evaluator import DatasetEvaluator
from detectron2.utils import comm


class Pano360Evaluator(DatasetEvaluator):
    def reset(self):
        self.horizon_loss = 0
        self.pitch_loss = 0
        self.roll_loss = 0
        self.vfov_loss = 0

    def process(self, inputs, outputs):
        self.horizon_loss = nn.functional.kl_div(
            nn.functional.log_softmax(outputs["horizon_logits"], dim=1),
            torch.stack([x["logits"]["gt_horizon"] for x in inputs]),
            reduction="batchmean",
        )
        self.pitch_loss = nn.functional.kl_div(
            nn.functional.log_softmax(outputs["pitch_logits"], dim=1),
            torch.stack([x["logits"]["gt_pitch"] for x in inputs]),
            reduction="batchmean",
        )
        self.roll_loss = nn.functional.kl_div(
            nn.functional.log_softmax(outputs["roll_logits"], dim=1),
            torch.stack([x["logits"]["gt_roll"] for x in inputs]),
            reduction="batchmean",
        )
        self.vfov_loss = nn.functional.kl_div(
            nn.functional.log_softmax(outputs["vfov_logits"], dim=1),
            torch.stack([x["logits"]["gt_vfov"] for x in inputs]),
            reduction="batchmean",
        )

    def evaluate(self):
        all_loss = comm.all_gather(
            [self.horizon_loss, self.pitch_loss, self.roll_loss, self.vfov_loss],
        )
        self.horizon_loss = sum(x[0] for x in all_loss)
        self.pitch_loss = sum(x[1] for x in all_loss)
        self.roll_loss = sum(x[2] for x in all_loss)
        self.vfov_loss = sum(x[3] for x in all_loss)
        total_loss = sum(sum(x) for x in all_loss)
        return {
            "total_loss": total_loss,
            "horizon_loss": self.horizon_loss,
            "pitch_loss": self.pitch_loss,
            "roll_loss": self.roll_loss,
            "vfov_loss": self.vfov_loss,
        }
