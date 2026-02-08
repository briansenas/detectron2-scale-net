# Some basic setup:
# Setup detectron2 logger
import argparse

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

# import some common libraries
# import some common detectron2 utilities


parser = argparse.ArgumentParser(
    description="Rui's Scale Estimation Network Training",
)
# Training
parser.add_argument(
    "--config-file-name",
    "-cfg",
    type=str,
    default="COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml",
    help="The config file for the model",
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="models/detectron2-R50-KeypointDetection.pkl",
    help="The path for .pkl model",
)
parser.add_argument(
    "--output-name",
    "-o",
    type=str,
    default="model.pth",
    help="The name for .pth model",
)
args = parser.parse_args()
setup_logger()
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(
    model_zoo.get_config_file(
        args.config_file_name,
    ),
)
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    args.config_file_name,
)
model = build_model(cfg)  # returns a torch.nn.Module
DetectionCheckpointer(model).load(
    args.model,
)  # load a file, usually from cfg.MODEL.WEIGHTS

checkpointer = DetectionCheckpointer(model, save_dir="output")
checkpointer.save(args.output_name)  # save to output/model_999.pth
