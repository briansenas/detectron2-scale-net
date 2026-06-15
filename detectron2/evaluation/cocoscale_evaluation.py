# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.utils import comm

from .evaluator import DatasetEvaluator


class COCOScaleEvaluator(DatasetEvaluator):
    def reset(self):
        self.vt_loss = 0

    def process(self, inputs, outputs):
        self.vt_loss = outputs[1]["vt_loss"]

    def evaluate(self):
        all_loss = comm.all_gather([self.vt_loss])
        return {
            "vt_loss_val": [x[0] for x in all_loss],
        }
