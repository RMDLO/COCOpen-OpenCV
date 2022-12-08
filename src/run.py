# Import libraries
import random
import yaml
from dotenv import load_dotenv

from cocopen import Cocopen

# loading environment variables
load_dotenv()


def main():

    # initialize random seed
    random.seed(random.randint(1, 1000))

    # Root director
    root_dir = "."

    # Load dataset parameters
    with open("./config/parameters.yml", 'r') as file:
        parameters = yaml.safe_load(file)

    # initialize cocopen object
    cocopen = Cocopen(
        root_dir=root_dir,
        parameters=parameters
    )

    # Create categories dictionary from parameters
    cocopen.generate_supercategories()

    # Make new directories
    cocopen.make_new_dirs(root_dir=root_dir)

    # Creating hand labeled validation dataset
    # cocopen.create_val_dataset()

    # Initializing Azure connection
    foreground_image_containers = (

    )
    foreground_image_wire_container = (
        "wire"  # name of the foreground image container on Azure for wire images
    )
    foreground_image_device_container = (
        "device"  # name of the foreground image container on Azure for device images
    )
    background_image_container = (
        "background"  # name of the background image container on Azure
    )
    cocopen.init_azure(
        foreground_image_wire_container=foreground_image_wire_container,
        foreground_image_device_container=foreground_image_device_container,
        background_image_container=background_image_container,
    )

    # Creating foreground image list
    cocopen.create_foreground_image_list()

    # Creating background image list
    cocopen.create_background_image_list()

    # Generate training data
    cocopen.generate_train_data()

    # Generate val data
    cocopen.generate_val_data()

    # Zip all files
    cocopen.zip(
        base_name=f"./datasets/zip/{dataset_directory_name}",
        format="zip",
        root_dir=f"./datasets/{dataset_directory_name}",
    )

if __name__ == "__main__":
    main()
