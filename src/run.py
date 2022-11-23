# Import libraries
import random
from dotenv import load_dotenv

from cocopen import Cocopen

load_dotenv()

def main():
    
    # initialize random seed
    random.seed(0)
    
    # Initialize Cocopen parameters
    # Root director
    root_dir = '.'
    # Dataset directory name
    dataset_directory_name = '20221123'
    # Number of training images
    num_of_train_images = 25
    # Number of val_easy images
    num_of_val_easy_images = 8

    # initialize cocopen object
    cocopen = Cocopen(root_dir=root_dir, dataset_directory_name=dataset_directory_name, num_of_train_images=num_of_train_images, num_of_val_easy_images=num_of_val_easy_images)
    
    # Make new directories
    cocopen.make_new_dirs(root_dir=root_dir)
    
    # Creating hand labeled validation dataset
    cocopen.create_val_dataset()

    # Initializing Azure connection
    foreground_image_container = 'single-wire' # name of the foreground image container on Azure
    background_image_container = 'background' # name of the background image container on Azure
    cocopen.init_azure(foreground_image_container=foreground_image_container, background_image_container=background_image_container)

    # Creating foreground image list
    cocopen.create_foreground_image_list()

    # Creating background image list
    cocopen.create_background_image_list()
    
    # Generate training data
    cocopen.generate_train_data()

    # Generate val_easy data
    cocopen.generate_val_easy_data()
       
    # Delete certain directories
    cocopen.delete_dirs(root_dir=root_dir)

    # Zip all files
    cocopen.zip(base_name=f'./datasets/zip/{dataset_directory_name}', format='zip', root_dir=f'./datasets/{dataset_directory_name}')

if __name__ == '__main__':
    main()