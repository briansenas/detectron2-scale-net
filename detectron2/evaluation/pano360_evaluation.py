# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn

from detectron2.data.datasets.pano360 import convert_preds_to_angles
from detectron2.utils import comm

from .evaluator import DatasetEvaluator


class Pano360Evaluator(DatasetEvaluator):
    def reset(self):
        self.pitch_loss_sum = 0.0
        self.roll_loss_sum = 0.0
        self.vfov_loss_sum = 0.0
        self.num_samples = 0

    def process(self, inputs, outputs):
        batch_size = len(inputs)
        if isinstance(outputs, tuple):
            preds, outputs = outputs

        for key in ["pitch", "roll", "vfov"]:
            logits = outputs[f"{key}_logits"]

            targets = torch.as_tensor(
                [x["logits"][f"gt_{key}"] for x in inputs],
                device=logits.device,
            )

            targets = torch.nn.functional.one_hot(
                targets,
                num_classes=logits.shape[1],
            ).to(logits.dtype)

            loss = nn.functional.kl_div(
                nn.functional.log_softmax(
                    logits,
                    dim=1,
                    dtype=logits.dtype,
                ),
                targets,
                reduction="batchmean",
            )

            getattr(self, f"{key}_loss_sum")
            setattr(
                self,
                f"{key}_loss_sum",
                getattr(self, f"{key}_loss_sum") +
                loss.item() * batch_size,
            )

        self.num_samples += batch_size

    def evaluate(self):
        stats = {
            "pitch_loss_sum": self.pitch_loss_sum,
            "roll_loss_sum": self.roll_loss_sum,
            "vfov_loss_sum": self.vfov_loss_sum,
            "num_samples": self.num_samples,
        }

        all_stats = comm.gather(stats, dst=0)

        if not comm.is_main_process():
            return {}

        pitch_loss_sum = sum(x["pitch_loss_sum"] for x in all_stats)
        roll_loss_sum = sum(x["roll_loss_sum"] for x in all_stats)
        vfov_loss_sum = sum(x["vfov_loss_sum"] for x in all_stats)
        num_samples = sum(x["num_samples"] for x in all_stats)

        pitch_loss = pitch_loss_sum / num_samples
        roll_loss = roll_loss_sum / num_samples
        vfov_loss = vfov_loss_sum / num_samples

        total_loss = pitch_loss + roll_loss + vfov_loss

        return {
            "calib_total_loss": total_loss,
            "calib_pitch_loss": pitch_loss,
            "calib_roll_loss": roll_loss,
            "calib_vfov_loss": vfov_loss,
        }


class Pano360EvaluatorME(DatasetEvaluator):
    def __init__(self, loss_criterion="kl"):
        self.loss_criterion = loss_criterion

    def reset(self):
        self.vfov_error_sum = 0.0
        self.pitch_error_sum = 0.0
        self.roll_error_sum = 0.0
        self.num_samples = 0

    def process(self, inputs, outputs):
        pred_vfov_logits = outputs["vfov_logits"].cpu()
        pred_pitch_logits = outputs["pitch_logits"].cpu()
        pred_roll_logits = outputs["roll_logits"].cpu()

        gt_vfov = np.asarray([x["vfov"] for x in inputs])
        gt_pitch = np.asarray([x["pitch"] for x in inputs])
        gt_roll = np.asarray([x["roll"] for x in inputs])

        pred_vfov, pred_pitch, pred_roll = convert_preds_to_angles(
            pred_vfov_logits,
            pred_pitch_logits,
            pred_roll_logits,
            loss_type=self.loss_criterion,
            return_type="np",
        )

        vfov_err = np.abs(
            np.degrees(pred_vfov) - np.degrees(gt_vfov)
        ).sum()

        pitch_err = np.abs(
            np.degrees(pred_pitch) - np.degrees(gt_pitch)
        ).sum()

        roll_err = np.abs(
            np.degrees(pred_roll) - np.degrees(gt_roll)
        ).sum()

        batch_size = len(inputs)

        self.vfov_error_sum += vfov_err
        self.pitch_error_sum += pitch_err
        self.roll_error_sum += roll_err
        self.num_samples += batch_size

    def evaluate(self):
        stats = {
            "vfov_error_sum": self.vfov_error_sum,
            "pitch_error_sum": self.pitch_error_sum,
            "roll_error_sum": self.roll_error_sum,
            "num_samples": self.num_samples,
        }

        all_stats = comm.gather(stats, dst=0)

        if not comm.is_main_process():
            return {}

        vfov_error_sum = sum(x["vfov_error_sum"] for x in all_stats)
        pitch_error_sum = sum(x["pitch_error_sum"] for x in all_stats)
        roll_error_sum = sum(x["roll_error_sum"] for x in all_stats)
        num_samples = sum(x["num_samples"] for x in all_stats)

        vfov_mae = vfov_error_sum / num_samples
        pitch_mae = pitch_error_sum / num_samples
        roll_mae = roll_error_sum / num_samples

        total_mae = vfov_mae + pitch_mae + roll_mae

        return {
            "calib_mae_total": total_mae,
            "calib_mae_vfov": vfov_mae,
            "calib_mae_pitch": pitch_mae,
            "calib_mae_roll": roll_mae,
        }
