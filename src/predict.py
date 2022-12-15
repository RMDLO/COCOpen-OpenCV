import cv2
import os

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.projects import point_rend
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode


# Class for the predict object
class Predict:
    def __init__(
        self,
        parameters: dict,
    ) -> None:

        self.parameters = parameters

        # COCO Dataset Loading
        self.name = parameters["directory"]["dataset_directory_name"]
        self.pr = "pr"
        self.model = "pointrend_rcnn_R_50_FPN_3x_coco"
        self.class_dict = {1: "device", 2: "wire"}

    def register_dataset(self):
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
        except FileExistsError:
            print("train and val datasets already registered!")

    def make_new_dirs(self):
        try:
            test_directory = "./train/opencv/"
            os.mkdir(test_directory)
        except FileExistsError:
            print("Test directory already exists!")
        try:
            prediction_directory = f"./train/opencv/{self.name}_{self.model}_{self.pr}"
            os.mkdir(prediction_directory)
        except FileExistsError:
            print("Prediction directory already exists!")

    def predict(self):

        cfg = get_cfg()
        point_rend.add_pointrend_config(cfg)
        cfg.merge_from_file(f"./train/config/{self.model}.yaml")

        cfg.MODEL.WEIGHTS = (
            f"./train/trained-models/{self.name}_{self.model}_{self.pr}.pth"
        )
        cfg.MODEL.DEVICE = "cuda:0"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        predictor = DefaultPredictor(cfg)

        dicts = DatasetCatalog.get("val")
        metadata = MetadataCatalog.get("val")
        print(f"./train/opencv/{self.name}_{self.model}_{self.pr}")
        for i, d in enumerate(
            dicts[: self.parameters["dataset_prediction"]["number_of_images"]]
        ):
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)
            v = Visualizer(im, metadata, instance_mode=ColorMode.IMAGE)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            save_directory = os.path.join(
                f"./train/opencv/{self.name}_{self.model}_{self.pr}",
                os.path.basename(d["file_name"]))
            cv2.imwrite(save_directory, out.get_image())
