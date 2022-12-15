# **Dataset Configuration**
This document contains basic information on how to configure a new dataset, including information about interfacing with Microsoft Azure and modifying dataset generation parameters. After completing [installation / setup](./docs/INSTALLATION.md), follow the below instructions.

## **Open COCOpen**
Navigate into the COCOpen directory:

```bash
# Navigate into the COCOpen directory
$ cd COCOCpen-OpenCV
```

## **User configurations**
1. Copy `connection string` from Azure Storage Account. Click [here](https://learn.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage?toc=%2Fazure%2Fstorage%2Fblobs%2Ftoc.json&bc=%2Fazure%2Fstorage%2Fblobs%2Fbreadcrumb%2Ftoc.json&tabs=azure-portal#view-account-access-keys) to learn how to access it.

2. Paste the connection string in the `config/parameters.yml` file under

```bash
# Pointer to raw input imagery and directory structuring
user_defined:
  root_dir: "." # ignore
  dataset_directory_name: "cocopen-dataset" # ignore
  AZURE_STORAGE_CONNECTION_STRING: '<paste here within single quotes>'
```

## **Running the API**

### **Adjusting parameters**
Open the `config/parameters.yml` file.

Here you can tweak parameters like `dataset_name` (the name of the generated dataset directory), `train_images` (the number of images in the generated training set), `threshold` (color thresholding values - we recommend keeping the default values for the provided wire and device images), and `max_instances` (the maximum number of objects of a particular category per image).

### **Running the script**
To execute the API, run the following:

```bash
# Run the run.py file
$ bash run.sh
```

## **Result**
You can now find the generated dataset in the `datasets` folder. The `datasets/zip/` folder provides a compressed .zip file of the generated dataset. An example annotation is visualized with the detectron2 visualizer below.

<p align="center">
  <img src="https://github.com/RMDLO/COCOpen-OpenCV/blob/1ce7c5c82115dcc193adae881033d168e462caba/demo/cocopen-dataset-review/visualization/0.png?raw=true" title="Visualization of COCOpen Automatic Instance Segmentation">
</p>
