import torch
torch.cuda.empty_cache()
import cv2
import os
import shutil
import yaml
import wget

# from detectron2.utils.logger import setup_logger
# setup_logger()
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.projects import point_rend
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo
from detectron2.modeling.roi_heads import fast_rcnn
from detectron2.checkpoint import PeriodicCheckpointer, DetectionCheckpointer
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import DatasetEvaluator
from detectron2.data.datasets import register_coco_instances

train = True
resume_training = False
unzip = False

# COCO Dataset Loading
with open("./config/parameters.yml", "r") as file:
    parameters = yaml.safe_load(file)
name = parameters["directory"]["dataset_directory_name"]
semantics = "obj_sem"
pr = "pr"
model = "pointrend_rcnn_R_50_FPN_3x_coco"
events_dir = f"./detectron2/events/{name}"+f"_{model}_{semantics}_{pr}"
try: os.mkdir(events_dir)
except: print("Directory already exists!")
try: os.mkdir("./demo/train/config")
except: print("Train directory already exists!")

class_dict = {1: "device",
             2: "cable"}

register_coco_instances("train", {}, f"./datasets/{name}/train/train.json", f"./datasets/{name}/train/")
register_coco_instances("val", {}, f"./datasets/{name}/val/val.json", f"./datasets/{name}/val/")

url_config = f'https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/configs/InstanceSegmentation/{model}.yaml?raw=true'
base_config = 'https://github.com/facebookresearch/detectron2/blob/main/configs/Base-RCNN-FPN.yaml?raw=true'
url_model = f'https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/{model}/164955410/model_final_edd263.pkl?raw=true'

wget.download(url_config, f'./demo/train/config/{model}.yaml')
wget.download(url_model, f'./demo/train/config/{model}.pkl')
wget.download(base_config, f'./demo/train/config/Base-PointRend-RCNN-FPN.yaml')

cfg = get_cfg()
point_rend.add_pointrend_config(cfg)
cfg.merge_from_file(f"./demo/train/config/{model}.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_dict.keys())
cfg.MODEL.POINT_HEAD.NUM_CLASSES = len(class_dict.keys())
cfg.OUTPUT_DIR = events_dir
weights_dir = f'./demo/train/config/{model}.pkl'
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
# Disabling all parallelism further increases reproducibility.
cfg.SEED = -1

resume = False
if train:
  cfg.DATASETS.TRAIN = ("train",)
  cfg.MODEL.MASK_ON = True
  cfg.DATASETS.TEST = ("val",)
  cfg.DATASETS.TEST = ()
  cfg.INPUT.MASK_FORMAT = "bitmask"
  cfg.DATALOADER.NUM_WORKERS = 1
  cfg.MODEL.WEIGHTS = weights_dir  # Let training initialize from model zoo
  cfg.SOLVER.BASE_LR = 0.00025
  cfg.SOLVER.MAX_ITER = 10
  cfg.SOLVER.CHECKPOINT_PERIOD = 5000 # limit this number- AstrobeeBumble only has 15GB of storage available and each checkpoint takes up ~0.5GB
  # cfg.SOLVER.STEPS = (20,100,500)
  # cfg.SOLVER.STEPS = (20, 10000, 20000)
  # cfg.SOLVER.gamma = 0.5
  cfg.SOLVER.IMS_PER_BATCH = 8
  # cfg.TEST.EVAL_PERIOD = 1000 # No. of iterations after which the Validation Set is evaluated. 
  cfg.TEST.EVAL_PERIOD = 100
  cfg.SOLVER.STEPS = []
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
  # cfg.TEST.DETECTIONS_PER_IMAGE = 20
  trainer = DefaultTrainer(cfg)
  # Load from last iteration
  trainer.resume_or_load(resume=resume_training)
  trainer.train()

if train:
  os.mkdir("./demo/train/trained-models/")
  shutil.move(events_dir + "/model_final.pth", "./demo/train/trained-models/" + f"{name}_{model}_{semantics}_{pr}.pth")

cfg.MODEL.WEIGHTS = f"./demo/train/trained-models/{name}_{model}_{semantics}_{pr}.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # testing threshold
predictor = DefaultPredictor(cfg)
test_directory = f"./demo/train/opencv/"
prediction_directory = f"./demo/train/opencv/{name}_{model}_{semantics}_{pr}"
try: os.mkdir(test_directory)
except: print("Test directory already exists!")
try: os.mkdir(prediction_directory)
except: print("Prediction directory already exists!")

dataset_dicts = DatasetCatalog.get("val")
metadata = MetadataCatalog.get("val")

for idx, d in enumerate(dataset_dicts[:10]):  
  im = cv2.imread(d["file_name"])
  outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
  instances = outputs["instances"]
  masks = instances.pred_masks
  v = Visualizer(im, metadata=metadata, instance_mode=ColorMode.IMAGE)
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  save_directory = os.path.join(prediction_directory,os.path.basename(d["file_name"]))
  cv2.imwrite(save_directory,out.get_image())

evaluator = COCOEvaluator("val", cfg, False)
val_loader = build_detection_test_loader(cfg, "val")
inference_on_dataset(predictor.model, val_loader, evaluator)