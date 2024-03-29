"""
Basic run script for generating and demonstrating a dataset. It also
trains a model on the generated dataset and performs inference if these
flags are set in config/parameters.yaml
"""

import random
import yaml
from cocopen import COCOpen
from demo import Demo
from train import Train, Predict


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

        # Initializing Box connection
        cocopen.init_box()

        # Make new directories
        cocopen.make_new_dirs()

        # Creating foreground and background image list
        cocopen.create_image_list()

        # Generate training data
        cocopen.generate_train_data()

        # Generate val data
        cocopen.generate_val_data()

    # Run the demo
    if params["demo_dataset"]:
        example = Demo(parameters=params)
        example.make_new_dirs()
        example.demo()

    # Train a new detectron2 model
    if params["train_detectron2"]:
        trainer = Train(parameters=params)
        trainer.make_new_dirs()
        trainer.download_models()
        trainer.register_dataset()
        trainer.train()

    # Perform inference using a trained model
    if params["predict_dataset"]:
        predictor = Predict(parameters=params)
        predictor.make_new_dirs()
        predictor.predict()


if __name__ == "__main__":
    # Load cocopen parameters
    with open("./config/parameters.yaml", mode="r", encoding="utf-8") as file:
        parameters = yaml.safe_load(file)
    main(params=parameters)
