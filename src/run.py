# Import libraries
import random
import yaml
from cocopen import Cocopen


def main():

    # initialize random seed
    random.seed(random.randint(1, 1000))

    # Load cocopen parameters
    with open("./config/parameters.yml", "r") as file:
        parameters = yaml.safe_load(file)

    # initialize cocopen object
    cocopen = Cocopen(
        parameters=parameters,
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


if __name__ == "__main__":
    main()
