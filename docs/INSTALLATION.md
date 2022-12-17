# **Installation**

This guide contains all installation and setup related information.

### **System Requirements**

Installation and execution of COCOpen was verified with the below environment.
- Operating System: Ubuntu 20.04.5 LTS
- Kernel: Linux 5.15.0-56-generic
- Architecture: x86-64
- Python: 3.9.15
- Conda: 22.9.0

All dependencies were also verified with every combination of [ubuntu-18.04, ubuntu-20.04, ubuntu-22.04] x [python-3.7, python-3.8, python-3.9, python-3.10] through Github continuous integration testing. 

For detailed versions of package dependencies, please see [`config/data_environment.yaml`](https://github.com/RMDLO/COCOpen-OpenCV/blob/main/config/data_environment.yaml).

## **Clone COCOpen-OpenCV Repository**

Clone this COCOpen-OpenCV in your desired location by running the following command in a terminal:
```bash
# Clone the repository
$ git clone https://github.com/RMDLO/COCOpen-OpenCV.git
```

## **Use Conda**

COCOpen-OpenCV uses an conda environment to manage versions of all dependencies. To get started with installing `conda` please follow [these instructions](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

For ease of creating a conda environment, COCOpen provides an `data_environment.yaml` file in the `config/` directory of this repository. The first line of the `data_environment.yaml` file defines the name of the new environment. This environment is used to generate a synthetic dataset using `src/cocopen.py`. To visualize the generated dataset, we include dependencies for the object detection library we use, [detectron2](https://github.com/facebookresearch/detectron2). The conda environment includes a cpu-only installation of PyTorch 1.10 on which detectron2 visualization depends. The detectron2 library cannot be installed with `conda` because it will not build properly with PyTorch. To use COCOpen to generate and visualize a dataset, please run the below commands to install dependencies.

```bash
# Navigate into the COCOpen directory
$ cd COCOCpen-OpenCV
```
```bash
# Clone the conda environment
$ conda env create -f config/data_environment.yaml
```
```bash
# Activate the conda environment
$ conda activate cocopen-data
```
```bash
# Install the prebuilt detectron2 library
$ python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
```

Please also set the `demo_dataset` value in `config/parameters.yaml` to `True` to perform visualization.

To train an object detection model, we provide a `train_environment.yaml` file which contains an installation of PyTorch 1.10 with CUDA 11.3. We also use the detectron2 library to train detection models. To set up a conda enviornment to use the `src/train.py` file to train and predict on a dataset, please run:

```bash
# Navigate into the COCOpen directory
$ cd COCOCpen-OpenCV
```
```bash
# Clone the conda environment
$ conda env create -f config/train_environment.yaml
```
```bash
# Activate the conda environment
$ conda activate cocopen-train
```
```bash
# Install the prebuilt detectron2 library
$ python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

Please also set the `train_dataset` value in `config/parameters.yaml` to `True` to train a model. The model training configuration can be adjusted with user-defined training parameters in the `src/train.py` script. After training a model, you can perform inference with the model on validation set images by setting the `predict_dataset` value in `config/parameters.yaml` to `True`.

## **Create an Azure Storage Container**

Follow [these instructions](https://github.com/RMDLO/COCOpen-OpenCV/blob/main/docs/README_AZURE.md) to create an Azure storage container to store input data.

## **Connect to Azure Storage Container**

To connect to an Azure storage container, perform the steps below.

1. Copy `connection string` from Azure Storage Account. Click [here](https://learn.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage?toc=%2Fazure%2Fstorage%2Fblobs%2Ftoc.json&bc=%2Fazure%2Fstorage%2Fblobs%2Fbreadcrumb%2Ftoc.json&tabs=azure-portal#view-account-access-keys) to learn how to access it.

2. Paste the connection string in the `config/parameters.yaml` file:

```bash
# Pointer to raw input imagery and directory structuring
user_defined:
  root_dir: "." # ignore
  dataset_directory_name: "cocopen-dataset" # ignore
  AZURE_STORAGE_CONNECTION_STRING: '<paste here within single quotes>'
```

## **Running COCOpen**

Open the `config/parameters.yaml` file and modify parameters like `dataset_name` (the name of the generated dataset directory), `train_images` (the number of images in the generated training set), and `max_instances` (the maximum number of objects of a particular category per image). Run COCOpen by performing:

```bash
# Run COCOpen
$ ./run.sh
```

The generated dataset saves to the `datasets` directory under the root directory of this repository. See [example run](https://github.com/RMDLO/COCOpen-OpenCV/blob/main/docs/EXAMPLE_RUN.md) to see a demonstration of generating a simple dataset of ethernet cables and ethernet devices with category, bounding box, and instance segmentation mask annotations in the COCO format.