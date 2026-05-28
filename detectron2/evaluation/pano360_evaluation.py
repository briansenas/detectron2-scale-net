# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn

from detectron2.utils import comm

from .evaluator import DatasetEvaluator


class Pano360Evaluator(DatasetEvaluator):
    def reset(self):
        self.horizon_loss = 0
        self.pitch_loss = 0
        self.roll_loss = 0
        self.vfov_loss = 0

    def process(self, inputs, outputs):
        for key in ["horizon", "pitch", "roll", "vfov"]:
            logits = outputs[f"{key}_logits"]
            targets = torch.as_tensor([x["logits"][f"gt_{key}"] for x in inputs]).to(device=logits.device)
            targets = torch.nn.functional.one_hot(targets, num_classes=logits.shape[1]).to(logits.dtype)
            setattr(
                self,
                f"{key}_loss",
                nn.functional.kl_div(
                    nn.functional.log_softmax(logits, dim=1, dtype=logits.dtype),
                    targets,
                    reduction="batchmean",
                ),
            )

    def evaluate(self):
        all_loss = comm.all_gather(
            [self.horizon_loss, self.pitch_loss, self.roll_loss, self.vfov_loss],
        )
        self.horizon_loss = sum(x[0] for x in all_loss)
        self.pitch_loss = sum(x[1] for x in all_loss)
        self.roll_loss = sum(x[2] for x in all_loss)
        self.vfov_loss = sum(x[3] for x in all_loss)
        self.total_loss = sum(sum(x) for x in all_loss)
        return {
            "batchmean_total_loss": self.total_loss,
            "batchmean_horizon_loss": self.horizon_loss,
            "batchmean_pitch_loss": self.pitch_loss,
            "batchmean_roll_loss": self.roll_loss,
            "batchmean_vfov_loss": self.vfov_loss,
        }
