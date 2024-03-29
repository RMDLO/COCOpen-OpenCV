# **Example Demonstration**

This document provides a demonstration of automatically generating labeled data using COCOpen and the UIUC wires dataset. After reading the [installation instructions](https://github.com/RMDLO/COCOpen-OpenCV/blob/main/docs/INSTALLATION.md), replicate this demo and generate sample data!

For this example demo we will be using foreground (wire and device) and background images from the [UIUC wires dataset](https://uofi.box.com/s/b8llku4yrvq44ijedw0lol1oz5sx7rja). This dataset is provided to test COCOpen by default.

## **1. Open COCOpen**

Navigate into the COCOpen directory

```bash
cd COCOCpen-OpenCV
```

## **2. Configure Data**

The provided `config/config.json` file already contains the Box application settings that are ready to generate data using the UIUC wires dataset. To connect to your own Box application that stores your own dataset, follow [these instructions](http://opensource.box.com/box-python-sdk/).

## **3. Adjust Parameters**

Open the `config/parameters.yaml` file.

Modify parameters such as the name of the generated dataset directory, `dataset_dir_name`, the number of images in the training data set, `train_images`, the threshold for color-based segmentation, `threshold`, and the maximum number of objects of a particular category per image, `max_instances`. We recommend keeping the [default](../config/parameters.yaml) values provided for this example run. Below are some of the tunable parameters defining a new dataset:

```yaml
# Dataset Size and Split Parameters
dataset_params:
  train_images: <number of train images here>
  val_images: <number of val images here>
  train_split: <train split here>
  ```
```yaml
# Color Thresholding Values for Each Object Category
color_threshold:
  device: <color thresholding value for device category images>
  wire: <color thresholding value for wire category images>
```
```yaml
# Contur Minimum Area (in pixels) Threshold Value
contour_threshold: <contour threshold value>
```
```yaml
# Maximum Number of Instances Per Image for Each Object Category
max_inst:
  device: <max number of devices in each image>
  wire: <max number of wires in each image>
```
## **4. Run COCOpen**

To execute COCOpen, run
```bash
./run.sh
```

## **5. Result**

The generated dataset will appear in the `datasets` folder. Example annotations are provided in the image below.
<p align="center">
  <img src="https://github.com/RMDLO/COCOpen-OpenCV/blob/main/docs/images/0.png?raw=true" title="Visualization of COCOpen Automatic Instance Segmentation" width="600px"> <figcaption>This is an example COCOpen-produced synthetic image containing multiple objects of interest superimposed on a randomly selected background. It visualizes ground truth instance segmentation mask, object category, and bounding box labels.</figcaption>
</p>
