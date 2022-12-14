# Import libraries
import pycocotools.mask as pycocomask
import cv2
import os

from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

# Class for the demo object
class Demo:
    # Constructor
    def __init__(
        self,
        parameters: dict,
    ) -> None:
        # Initializing parameters
        self.parameters = parameters

        # Initializing root and destination directory
        self.root_dir = self.parameters["directory"]["root_dir"]
        self.dataset_directory_name = self.parameters["directory"][
            "dataset_directory_name"
        ]

        # Saving all directory names
        self.dataset_dir = self.root_dir + f"/datasets/{self.dataset_directory_name}"
        self.train = self.dataset_dir + "/train"
        self.val = self.dataset_dir + "/val"

        self.demo_dir = self.root_dir + "/demo"
        self.demo_dataset_dir = self.demo_dir + f"/{self.dataset_directory_name}"
        self.visualization_dir = self.demo_dataset_dir + "/visualization"
        self.mask_dir = self.demo_dataset_dir + "/masks"

    def make_new_dirs(self):
        try:
            os.mkdir(self.demo_dir)
        except:
            print("Demo directory already exists!")
        try:
            os.mkdir(self.demo_dataset_dir)
        except:
            print("Demo dataset directory already exists!")
        try:
            os.mkdir(self.visualization_dir)
        except:
            print("Visualization directory already exists!")
        try:
            os.mkdir(self.mask_dir)
        except:
            print("Masks directory already exists!")

    def demo(self):

        if self.parameters["dataset_verification"]["which_dataset"] == "train":
            register_coco_instances(
                "train", {}, f"{self.train}/train.json", f"{self.train}/"
            )
            dicts = DatasetCatalog.get("train")
            metadata = MetadataCatalog.get("train")
        elif self.parameters["dataset_verification"]["which_dataset"] == "val":
            register_coco_instances("val", {}, f"{self.val}/val.json", f"{self.val}/")
            dicts = DatasetCatalog.get("val")
            metadata = MetadataCatalog.get("val")

        for i, d in enumerate(
            dicts[: self.parameters["dataset_verification"]["number_of_images"] - 1]
        ):
            img = cv2.imread(d["file_name"])
            annos = d["annotations"]
            mask_list = []
            for anno in annos:
                encoded = anno["segmentation"]
                instance_annotation = pycocomask.decode(encoded) * 255
                instance_img = cv2.cvtColor(instance_annotation, cv2.COLOR_GRAY2BGR)
                x, y, w, h = anno["bbox"]
                mask = cv2.rectangle(
                    instance_img, (x, y), (x + w, y + h), (255, 255, 255), 2
                )
                mask_list.append(mask)

            # horizontally concatenate visualization of object instance segmentation masks
            concatenated_masks = cv2.hconcat(mask_list)

            # visualize object instance segmentation on cocopen-generated data
            visualizer = Visualizer(
                img, metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE
            )
            out = visualizer.draw_dataset_dict(d)

            # save masks and visualizations
            cv2.imwrite(
                os.path.join(self.visualization_dir, str(i) + ".png"), out.get_image()
            )
            cv2.imwrite(
                os.path.join(self.mask_dir, str(i) + ".png"), concatenated_masks
            )
