# Import libraries
import random
from dotenv import load_dotenv

from cocopen import Cocopen

load_dotenv()

def main():
    
    # initialize random seed
    random.seed(1)
    
    # Initialize Cocopen parameters

    # Dataset directory name
    date = '20221109'
    # System username
    username = 'wall-e'
    # Number of training images
    num_of_train_images = 25
    # Number of val_easy images
    num_of_val_easy_images = 8

    # initialize cocopen object
    cocopen = Cocopen(date=date, username=username, num_of_train_images=num_of_train_images, num_of_val_easy_images=num_of_val_easy_images)
    
    # Make new directories
    cocopen.make_new_dirs()
    
    # Creating hand labeled validation dataset
    cocopen.create_val_dataset()

    # Initializing Azure connection
    foreground_image_container = 'single-wire'
    background_image_container = 'background'
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
    cocopen.delete_dirs()

    # Zip all files
    cocopen.zip()

if __name__ == '__main__':
    main()