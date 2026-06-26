# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.utils import comm

from .coco_evaluation import COCOEvaluator
from .evaluator import DatasetEvaluator


class COCOScaleEvaluator(COCOEvaluator):
    def process(self, inputs, outputs):
        preds, _ = outputs
        return super().process(inputs, [{"instances": x for x in preds}])


class COCOScaleEvaluatorVT(DatasetEvaluator):
    def reset(self):
        self.vt_loss_sum = 0.0
        self.num_samples = 0

    def process(self, inputs, outputs):
        batch_size = len(inputs)

        # NOTE: Now I could record each layer vt_loss if needed
        vt_loss = outputs[1]["vt_losses"][-1]

        if hasattr(vt_loss, "item"):
            vt_loss = vt_loss.item()

        self.vt_loss_sum += vt_loss * batch_size
        self.num_samples += batch_size

    def evaluate(self):
        stats = {
            "vt_loss_sum": self.vt_loss_sum,
            "num_samples": self.num_samples,
        }

        all_stats = comm.gather(stats, dst=0)

        if not comm.is_main_process():
            return {}

        vt_loss_sum = sum(x["vt_loss_sum"] for x in all_stats)
        num_samples = sum(x["num_samples"] for x in all_stats)

        vt_loss = vt_loss_sum / num_samples

        return {
            "vt_loss_val": vt_loss,
        }


class COCOScaleEvaluatorAndVT(COCOEvaluator):
    def reset(self):
        super().reset()
        self.vt_loss_sum = 0.0
        self.num_samples = 0

    def process(self, inputs, outputs):
        preds, camrcnn_data = outputs
        super().process(inputs, [{"instances": x for x in preds}])
        batch_size = len(inputs)

        # NOTE: Now I could record each layer vt_loss if needed
        vt_loss = camrcnn_data["vt_losses"][-1]

        if hasattr(vt_loss, "item"):
            vt_loss = vt_loss.item()

        self.vt_loss_sum += vt_loss * batch_size
        self.num_samples += batch_size

    def evaluate(self):
        # Do normal COCO Evaluation to see impact on detection
        results = super().evaluate()
        # Get vt reprojection loss on eval for comparison with original implementation
        stats = {
            "vt_loss_sum": self.vt_loss_sum,
            "num_samples": self.num_samples,
        }

        all_stats = comm.gather(stats, dst=0)

        if not comm.is_main_process():
            return {}

        vt_loss_sum = sum(x["vt_loss_sum"] for x in all_stats)
        num_samples = sum(x["num_samples"] for x in all_stats)

        vt_loss = vt_loss_sum / num_samples
        results["vt_loss_val"] = vt_loss
        return results
