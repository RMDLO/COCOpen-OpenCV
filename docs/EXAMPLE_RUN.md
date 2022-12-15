# **Example Demo**
Provided below is a demo of generating data from the UIUC wires dataset. Once you're done with the [installation](./INSTALLATION.md), you will be able to replicate this demo and generate sample data to get a taste of how the API works.

## **1. Open COCOpen**
Start off by navigating into the COCOpen repository.

```bash
# Navigate into the COCOpen directory
$ cd COCOCpen-OpenCV
```

## **2. User configurations**
a. The provided `config/parameters.yml` file already contains an Azure connection string that's ready to generate data from the UIUC wires dataset.

```yaml
# Pointer to raw input imagery and directory structuring
user_defined:
  root_dir: "." # ignore
  dataset_directory_name: "cocopen-dataset-4" # ignore
  AZURE_STORAGE_CONNECTION_STRING: 'DefaultEndpointsProtocol=https;AccountName=uiucwiresdataset;AccountKey=VkJ1HT3LkDuiLTFK8yd+eAFLvhLKJNqLDIealTPY9Lv6Dp7VDFVWKIvhnNXqC+GCQYjh7NQVuH1r+ASt/tVk7g==;EndpointSuffix=core.windows.net' # UIUC's Azure connection string
```

## **3. Adjusting parameters**
Open the `config/parameters.yml` file.

Here you can tweak parameters like `dataset_name` (the name of the generated dataset directory), `train_images` (the number of images in the generated training set), `threshold` (color thresholding values - we recommend keeping the default values for the provided wire and device images), and `max_instances` (the maximum number of objects of a particular category per image).

Set it to these default values:
```yaml
# Dataset Size and Split Parameters
dataset_params:
  train_images: 25
  val_images: 10
  train_split: 0.8

# Image Shape
shape:
  height: 1080
  width: 1920
  
# Object Categories and Corresponding Category IDs. 
# (int, required): an integer in the range [0, num_categories-1] representing the category label. 
# The value num_categories is reserved to represent the “background” category, if applicable.
categories:
  device: 1
  wire: 2
  background: 3

# Color Thresholding Values for Each Object Category
color_threshold:
  device: 50
  wire: 30

# Contur Minimum Area (in pixels) Threshold Value
contour_threshold: 1000
  
# Maximum Number of Instances Per Image for Each Object Category
max_instances:
  device: 2
  wire: 4

# Parameters for Scale Jittering
scale_jittering:
  apply_scale_jittering: True

  # if True, different objects use different scale factors
  individual_scale_jittering: True

  # scale factor range
  scale_factor_min: 0.75
  scale_factor_max: 1.25

# Parameters for Color Augmentation
color_augmentation:
  apply_color_augmentation: True

  # if True, different objects use different values
  individual_color_augmentation: t
  change_saturation: True
  change_brightness: True
  change_contrast: True
  change_hue: True

  # Lower/Upper Limit of Color Augmentation
  enhancer_min: 0.75
  enhancer_max: 1.25

  # if True, apply color augmentation again to the combined image
  color_augmentation_on_combined_image: True

# Flag to designate whether the demo is performed
run_demo: True

# Parameters for which dataset to visualize and how many annotated images from the dataset to visualize.
# Options: "train" and "val"
dataset_verification:
  which_dataset: "val"
  number_of_images: 2
```

## **4. Running the script**
To execute the API, run the following:

```bash
# Run the run.py file
$ bash run.sh
```

## **5. Result**
You can now find the generated dataset in the `datasets` folder. The `datasets/zip/` folder provides a compressed .zip file of the generated dataset. Example annotations are provided in the images below.