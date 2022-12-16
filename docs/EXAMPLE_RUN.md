# **Example Demo**
This document provides a demonstration of automatically generating labeled using COCOpen and the UIUC wires dataset. After reading the [installation instructions]([./docs/INSTALLATION.md](https://github.com/RMDLO/COCOpen-OpenCV/blob/main/docs/INSTALLATION.md)), replicate this demo and generate sample data!

## **1. Open COCOpen**
Navigate into the COCOpen repository.

```bash
# Navigate into the COCOpen directory
$ cd COCOCpen-OpenCV
```

## **2. Configure Data **
a. The provided `config/parameters.yml` file already contains an Azure connection string that is ready to generate data using the UIUCWires dataset

```yaml
# Pointer to raw input imagery and directory structuring
user_defined:
  root_dir: "." # ignore
  dataset_directory_name: "cocopen-dataset" # ignore
  AZURE_STORAGE_CONNECTION_STRING: 'DefaultEndpointsProtocol=https;AccountName=uiucwiresdataset;AccountKey=VkJ1HT3LkDuiLTFK8yd+eAFLvhLKJNqLDIealTPY9Lv6Dp7VDFVWKIvhnNXqC+GCQYjh7NQVuH1r+ASt/tVk7g==;EndpointSuffix=core.windows.net' # UIUC's Azure connection string
```

## **3. Adjust Parameters**
Open the `config/parameters.yml` file.

Tweak parameters such as `dataset_name` (the name of the generated dataset directory), `train_images` (the number of images in the generated training set), `threshold` (color thresholding values - we recommend keeping the default values for the UIUCWires data), and `max_instances` (the maximum number of objects of a particular category per image).

These are some of the default values we use:
```yaml
# Dataset Size and Split Parameters
dataset_params:
  train_images: 20
  val_images: 8
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
max_inst:
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
  individual_color_augmentation: True
  saturation: True
  brightness: True
  contrast: True
  hue: False

  # Lower/Upper Limit of Color Augmentation
  enhancer_min: 0.75
  enhancer_max: 1.25

  # if True, apply color augmentation again to the combined image
  color_augment_combined: True

# Flag to designate whether to generate a new dataset
generate_dataset: True

# Flag to designate whether the demo is performed
demo_dataset: False

# Flag to designate whether to train a model from the detectron2 model zoo on the generated dataset
train_detectron2: False

# Flag to designate whether to perform prediction using a trained model
predict: False

# Parameters for which dataset to visualize and how many annotated images from the dataset to visualize.
# Options: "train" and "val"
dataset_verification:
  which_dataset: "val"
  number_of_images: 2

dataset_prediction:
  number_of_images: 5

# Pointer to raw input imagery and directory structuring
directory:
  root_dir: "."
  dataset_dir_name: "cocopen-dataset-review"
  AZURE_STORAGE_CONNECTION_STRING: 'DefaultEndpointsProtocol=https;AccountName=uiucwiresdataset;AccountKey=VkJ1HT3LkDuiLTFK8yd+eAFLvhLKJNqLDIealTPY9Lv6Dp7VDFVWKIvhnNXqC+GCQYjh7NQVuH1r+ASt/tVk7g==;EndpointSuffix=core.windows.net'
```

## **4. Running the script**
To execute the API, run the following:

```bash
# Run the run.py file
$ bash run.sh
```

## **5. Result**
You can now find the generated dataset in the `datasets` folder. The `datasets/zip/` folder provides a compressed .zip file of the generated dataset. Example annotations are provided in the images below.
