# **Installation**
This guide contains all installation and setup related information.

### **System Requirements**

Installation and execution of COCOpen was verified with the below environment.
- Operating System: Ubuntu 20.04.5 LTS
- Kernel: Linux 5.15.0-56-generic
- Architecture: x86-64
- Python: 3.9.15
- Conda: 22.9.0

We expect this to work in other UNIX based systems and WSL, with python 3. However, furthermore extensive testing needs to be performed to confirm all the environments this will work in.

For detailed versions of package dependencies, please see [`config/data_environment.yaml`](https://github.com/RMDLO/COCOpen-OpenCV/blob/main/config/data_environment.yaml).

## **Clone COCOpen-OpenCV Repository**
Clone this COCOpen-OpenCV in your desired location by running the following command in a terminal:
```bash
# Clone the repository
$ git clone https://github.com/RMDLO/COCOpen-OpenCV.git
```

## **Use Conda**
COCOpen-OpenCV uses an conda environment to manage versions of all dependencies. To get started with installing `conda` please follow [these instructions](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

For ease of creating a conda environment, COCOpen provides an `data_environment.yaml` file in the `config/` directory of this repository. The first line of the `data_environment.yaml` file defines the name of the new environment. This environment is used to generate a synthetic dataset using `src/cocopen.py`. To visualize the generated dataset, we include dependencies for the object detection library we use, [detectron2](https://github.com/facebookresearch/detectron2). The conda environment includes a cpu-only installation of PyTorch 1.10 on which detectron2 visualization depends. The detectron2 library cannot be installed with `conda` because it will not build properly with PyTorch. To use COCOpen to generate and visualize a dataset, please run:

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

## **Setup Azure Storage Container**
Follow [these instructions](https://github.com/RMDLO/COCOpen-OpenCV/blob/main/docs/README_AZURE.md) to setup your Azure storage container.

## **Connect your Storage Container**

To connect your storage container, perform the steps below.

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

Open the `config/parameters.yaml` file. Here you can tweak parameters like `dataset_name` (the name of the generated dataset directory), `train_images` (the number of images in the generated training set), `threshold` (color thresholding values - we recommend keeping the default values for the provided wire and device images), and `max_instances` (the maximum number of objects of a particular category per image).

To execute the API, run the following:

```bash
# Run COCOpen
$ ./run.sh
```

Furthermore, see [example run](https://github.com/RMDLO/COCOpen-OpenCV/blob/main/docs/EXAMPLE_RUN.md) to see a demo automatically generating a simple dataset of ethernet cables and ethernet devices with category, bounding box, and instance segmentation mask annotations in the COCO format.

## **Result**
You can now find the generated dataset in the `datasets` folder. The `datasets/zip/` folder provides a compressed .zip file of the generated dataset. 

Here is an example of an annotated image, as visualized with detectron2.

<p align="center">
  <img src="https://github.com/RMDLO/COCOpen-OpenCV/blob/main/docs/images/0.png?raw=true" title="Visualization of COCOpen Automatic Instance Segmentation">
</p>
