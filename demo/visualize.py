# Import libraries
import pycocotools.mask as pycocomask
import cv2
import os
import yaml

import detectron2
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
        self.visualization_dir = self.root_dir + "/demo/visualization"
        self.mask_dir = self.root_dir + "/demo/masks"

        # Initializing supercategories dictionary
        self.categories = []

        # Saving all directory names
        self.dataset_dir = self.root_dir + f"/datasets/{self.dataset_directory_name}"
        self.train = self.dataset_dir + "/train"
        self.val = self.dataset_dir + "/val"

        # Initialize height and width
        self.height = parameters["shape"]["height"]
        self.width = parameters["shape"]["width"]

    # Generate super categories, used when an object super category (like "wire") may contain subcategories (like "ethernet")
    def generate_supercategories(self):
        """
        Generate dictionary for super categories based on parameters .yaml file
        """
        for key in self.parameters["categories"]:
            supercategory_dict = {
                "supercategory": key,
                "id": self.parameters["categories"][key],
                "name": key,
            }
            self.categories.append(supercategory_dict)
        print("Generated Categories Dictionary from Parameters")

    def demo(self):

        register_coco_instances(
            "train", {}, f"{self.train}/train.json", f"{self.train}/"
        )
        dicts = DatasetCatalog.get("train")
        metadata = MetadataCatalog.get("train")
        for i, d in enumerate(dicts[:10]):
            img = cv2.imread(d["file_name"])
            annos = d["annotations"]
            mask_list = []
            for anno in annos:
                encoded = anno["segmentation"]
                instance_annotation = pycocomask.decode(encoded) * 255
                instance_img = cv2.cvtColor(instance_annotation, cv2.COLOR_GRAY2BGR)
                x, y, w, h = anno["bbox"]
                mask = cv2.rectangle(instance_img, (x,y), (x+w, y+h), (255,255,255), 2)
                mask_list.append(mask)

            # horizontally concatenate visualization of object instance segmentation masks
            concatenated_masks = cv2.hconcat(mask_list)
            
            # visualize object instance segmentation on cocopen-generated data
            visualizer = Visualizer(
                img, metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE
            )
            out = visualizer.draw_dataset_dict(d)

            # save masks and visualizations
            cv2.imwrite(os.path.join(self.visualization_dir, str(i) + ".png"), out.get_image())
            cv2.imwrite(os.path.join(self.mask_dir, str(i) + ".png"), concatenated_masks)

if __name__ == "__main__":
    # Load cocopen parameters
    try:
        os.mkdir("./demo/visualization")
    except:
        print("datasets directory already exists!")

    with open("./config/parameters.yml", "r") as file:
        parameters = yaml.safe_load(file)
    demo = Demo(
        parameters=parameters,
    )
    demo.demo()
