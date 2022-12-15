import os
import shutil

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.projects import point_rend
from detectron2.data.datasets import register_coco_instances

import wget
import torch

torch.cuda.empty_cache()


class Train:
    """
    A class to train an object detection model from the detectron2
    model zoo. The model is trained to predict category id,
    bounding box, and instance segmentation maks given a dataset
    in COCO format.
    ...

    Attributes
    ----------
    parameters : dict
        contains all parameters used to generate the dataset

    Methods
    -------
    make_new_dirs():
        Make new directories where model configuration and checkpoint
        files are saved based on the name of the dataset
    register_dataset():
        Register the dataset given a folder of images and their
        corresponding annotations in COCO format
    train():
        Train an object detection model given a training configuration and
        save model checkpoints
    """

    def __init__(
        self,
        parameters: dict,
    ) -> None:
        # Initializing parameters
        self.parameters = parameters
        self.train_detectron2 = True
        self.resume_training = True
        self.unzip = False

        # COCO Dataset Loading
        self.name = parameters["directory"]["dataset_directory_name"]
        self.model = "pointrend_rcnn_R_50_FPN_3x_coco"
        self.class_dict = {1: "device", 2: "wire"}

        # Training directory names
        self.train_dir = "./train/"
        self.config_dir = "./train/config"
        self.events_dir = "./train/events/"
        self.model_dir = f"./train/events/{self.name}_{self.model}"
        self.trained_models_dir = "./train/trained-models"

    def make_new_dirs(self):
        """
        Make new directories where model configuration and checkpoint
        files are saved based on the name of the dataset
        """
        try:
            os.mkdir(self.train_dir)
            os.mkdir(self.config_dir)
            os.mkdir(self.events_dir)
            os.mkdir(self.model_dir)
            url_config = f"https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/configs/InstanceSegmentation/{self.model}.yaml?raw=true"
            base_config = "https://github.com/facebookresearch/detectron2/blob/main/configs/Base-RCNN-FPN.yaml?raw=true"
            url_model = f"https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/{self.model}/164955410/model_final_edd263.pkl?raw=true"

            wget.download(url_config, f"{self.config_dir}/{self.model}.yaml")
            wget.download(url_model, f"{self.config_dir}/{self.model}.pkl")
            wget.download(
                base_config, f"{self.config_dir}/Base-PointRend-RCNN-FPN.yaml"
            )
        except FileExistsError:
            print("Train directory already exists!")

        if self.train_detectron2:
            try:
                os.mkdir(self.trained_models_dir)
            except FileExistsError:
                print("Trained models directory already exists!")

    def register_dataset(self):
        """
        Register the dataset given a folder of images and their
        corresponding annotations in COCO format
        """
        try:
            register_coco_instances(
                "train",
                {},
                f"./datasets/{self.name}/train/train.json",
                f"./datasets/{self.name}/train/",
            )
            register_coco_instances(
                "val",
                {},
                f"./datasets/{self.name}/val/val.json",
                f"./datasets/{self.name}/val/",
            )
        except KeyError:
            print("train and val datasets already registered!")

    def train(self):
        """
        Train an object detection model given a training configuration and
        save model checkpoints
        """
        cfg = get_cfg()
        point_rend.add_pointrend_config(cfg)
        cfg.merge_from_file(f"./train/config/{self.model}.yaml")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.class_dict.keys())
        cfg.MODEL.POINT_HEAD.NUM_CLASSES = len(self.class_dict.keys())
        cfg.OUTPUT_DIR = self.model_dir
        weights_dir = f"./train/config/{self.model}.pkl"
        # Set seed to negative to fully randomize everything.
        # Set seed to positive to use a fixed seed.
        # Note: a fixed seed increases reproducibility but does not guarantee
        # deterministic behavior. Disabling all parallelism further
        # increases reproducibility.
        cfg.SEED = -1
        if not self.resume_training:
            self.make_new_dirs()
        else:
            pass

        if self.train_detectron2:
            cfg.DATASETS.TRAIN = ("train",)
            cfg.MODEL.MASK_ON = True
            cfg.DATASETS.TEST = ("val",)
            cfg.DATASETS.TEST = ()
            cfg.INPUT.MASK_FORMAT = "bitmask"
            cfg.DATALOADER.NUM_WORKERS = 1
            # Initialize training from model zoo:
            cfg.MODEL.WEIGHTS = weights_dir
            cfg.SOLVER.BASE_LR = 0.00025
            cfg.SOLVER.MAX_ITER = 100000
            cfg.SOLVER.CHECKPOINT_PERIOD = 5000
            # cfg.SOLVER.STEPS = (20,100,500)
            # cfg.SOLVER.STEPS = (20, 10000, 20000)
            # cfg.SOLVER.gamma = 0.5
            cfg.SOLVER.IMS_PER_BATCH = 8
            # No. of iterations after which the Validation Set is evaluated:
            # cfg.TEST.EVAL_PERIOD = 1000
            cfg.TEST.EVAL_PERIOD = 100
            cfg.SOLVER.STEPS = []
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
            # cfg.TEST.DETECTIONS_PER_IMAGE = 20
            trainer = DefaultTrainer(cfg)
            # Load from last iteration
            trainer.resume_or_load(resume=self.resume_training)
            trainer.train()

        if self.train_detectron2:
            shutil.move(
                self.events_dir + f"{self.name}_{self.model}/model_final.pth",
                f"./train/trained-models/{self.name}_{self.model}.pth",
            )
            print(
                f"Training complete! Model saved in ./train/trained-models/{self.name}_{self.model}.pth!"
            )
