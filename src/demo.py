import pycocotools.mask as pycocomask
import cv2
import os

from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode


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
        self.data_dir_name = self.parameters["directory"]["dataset_dir_name"]

        # Saving all directory names
        self.dataset_dir = self.root_dir + f"/datasets/{self.data_dir_name}"
        self.demo_dir = self.root_dir + "/demo"
        self.demo_dataset_dir = self.demo_dir + f"/{self.data_dir_name}"
        self.visualization_dir = self.demo_dataset_dir + "/visualization"
        self.mask_dir = self.demo_dataset_dir + "/masks"

    def make_new_dirs(self):
        try:
            os.mkdir(self.demo_dir)
        except FileExistsError:
            print("Demo directory already exists!")
        try:
            os.mkdir(self.demo_dataset_dir)
        except FileExistsError:
            print("Demo dataset directory already exists!")
        try:
            os.mkdir(self.visualization_dir)
        except FileExistsError:
            print("Visualization directory already exists!")
        try:
            os.mkdir(self.mask_dir)
        except FileExistsError:
            print("Masks directory already exists!")

    def demo(self):

        data = self.parameters["dataset_verification"]["which_dataset"]
        register_coco_instances(
            data,
            {},
            f"{self.dataset_dir}/{data}/{data}.json",
            f"{self.dataset_dir}/{data}/",
        )
        dicts = DatasetCatalog.get(f"{data}")
        metadata = MetadataCatalog.get(f"{data}")

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

            # horizontally concatenate visualization of object instance
            # segmentation masks
            cat_masks = cv2.hconcat(mask_list)

            # visualize object instance segmentation on cocopen-generated data
            visualizer = Visualizer(
                img, metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE
            )
            out = visualizer.draw_dataset_dict(d)

            # save masks and visualizations
            cv2.imwrite(
                os.path.join(self.visualization_dir, str(i) + ".png"), out.get_image()
            )
            cv2.imwrite(os.path.join(self.mask_dir, str(i) + ".png"), cat_masks)
