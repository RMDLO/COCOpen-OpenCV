"""
Basic run script for generating and demonstrating a dataset. It also
trains a model on the generated dataset and performs inference if these
flags are set in config/parameters.yaml
"""

import random
import yaml
from cocopen import COCOpen
from demo import Demo
from train import Train
from predict import Predict


def main(params):
    """
    This function performs all functions necessary to generate
    a dataset with the COCOpen class, visualize annotations and
    images in this auto-generated dataset, train an instance
    segmentation model using the detectron2 library, and perform
    prediction with the trained model.
    """

    # initialize random seed
    random.seed(random.randint(1, 1000))
    # random.seed(2)

    # Generate a new COCO-formatted dataset
    if parameters["generate_dataset"]:

        # initialize cocopen object
        cocopen = COCOpen(
            param=params,
        )

        # Create categories dictionary from parameters
        cocopen.generate_supercategories()

        # Make new directories
        cocopen.make_new_dirs()

        # Initializing Azure connection
        cocopen.init_azure()

        # Creating foreground and background image list
        cocopen.create_image_list()

        # Generate training data
        cocopen.generate_train_data()

        # Generate val data
        cocopen.generate_val_data()

        # Zip all files
        cocopen.zip(
            base_name=f"./datasets/zip/{cocopen.dataset_directory_name}",
            format="zip",
            root_dir=f"./datasets/{cocopen.dataset_directory_name}",
        )

    # Run the demo
    if params["demo_dataset"]:
        example = Demo(parameters=params)
        example.make_new_dirs()
        example.demo()

    # Train a new detectron2 model
    if params["train_detectron2"]:
        trainer = Train(parameters=params)
        trainer.register_dataset()
        trainer.train()

    if params["predict"]:
        predictor = Predict(parameters=params)
        predictor.register_dataset()
        predictor.make_new_dirs()
        predictor.predict()


if __name__ == "__main__":
    # Load cocopen parameters
    with open("./config/parameters.yaml", "r") as file:
        parameters = yaml.safe_load(file)
    main(params=parameters)
