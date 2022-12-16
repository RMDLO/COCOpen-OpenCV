# **Installation**
This notebook contains all installation and setup related information.

### **Environment Information**

Installation and execution of COCOpen was verified with the below environment.
- Operating System: Ubuntu 20.04.5 LTS
- Kernel: Linux 5.15.0-56-generic
- Architecture: x86-64
- Python: 3.9.15

For detailed versions of package dependencies, please see [`config/environment.yaml`](https://github.com/RMDLO/COCOpen-OpenCV/blob/976083972a07d0fecb5fe4c5c0e6d16d73c7df46/config/environment.yaml).

## **Clone COCOpen-OpenCV Repository**
Clone this repository in your desired location by running the following command in a terminal:
```bash
# Clone the repository
$ git clone https://github.com/RMDLO/COCOpen-OpenCV.git
```

## **Use Anaconda**
COCOpen-OpenCV uses an anaconda environment to manage versions of all dependencies. To get started with installing `conda` please follow [these instructions](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

For ease of creating a conda environment, COCOpen provides an `environment.yaml` file in the `config/` directory of this repository. The first line of the `environment.yaml` file defines the name of the new environment. This environment is used to generate a synthetic dataset using `src/cocopen.py`. To visualize the generated dataset, we include dependencies for the object detection library we use, [detectron2](https://github.com/facebookresearch/detectron2). The conda environment includes a cpu-only installation of PyTorch 1.10 on which detectron2 visualization depends. The detectron2 library cannot be installed with `conda` because it will not build properly with PyTorch. To use COCOpen to generate and visualize a dataset, please run:

```bash
# Navigate into the COCOpen directory
$ cd COCOCpen-OpenCV
# Clone the conda environment
$ conda env create -f config/environment.yaml
# Activate the conda environment
$ conda activate cocopen
# Install the prebuilt detectron2 library
$ python -m pip install detectron2 -f \ https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
```

To train an object detection model, we provide a `train_environment.yml` file which contains an installation of PyTorch 1.10 with CUDA 11.3. We also use the detectron2 library to train detection models. To set up a conda enviornment to use the `src/train.py` file to train and predict on a dataset, please run:

```bash
# Navigate into the COCOpen directory
$ cd COCOCpen-OpenCV
# Clone the conda environment
$ conda env create -f config/train_environment.yaml
# Activate the conda environment
$ conda activate cocopen-train
# Install the prebuilt detectron2 library
$ python -m pip install detectron2 -f \ https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

## **Configure Azure Storage Container**

To learn how to set up your dataset on Azure, read [this](https://github.com/RMDLO/COCOpen-OpenCV/blob/main/docs/README_AZURE.md).

## **Configure a New Dataset**
This section contains basic information on how to configure a new dataset, including information about interfacing with Microsoft Azure and modifying dataset generation parameters. 

## **User Configurations**
1. Copy `connection string` from Azure Storage Account. Click [here](https://learn.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage?toc=%2Fazure%2Fstorage%2Fblobs%2Ftoc.json&bc=%2Fazure%2Fstorage%2Fblobs%2Fbreadcrumb%2Ftoc.json&tabs=azure-portal#view-account-access-keys) to learn how to access it.

2. Paste the connection string in the `config/parameters.yml` file:

```bash
# Pointer to raw input imagery and directory structuring
user_defined:
  root_dir: "." # ignore
  dataset_directory_name: "cocopen-dataset" # ignore
  AZURE_STORAGE_CONNECTION_STRING: '<paste here within single quotes>'
```

## **Run the COCOpen Example**

Open the `config/parameters.yml` file. Here you can tweak parameters like `dataset_name` (the name of the generated dataset directory), `train_images` (the number of images in the generated training set), `threshold` (color thresholding values - we recommend keeping the default values for the provided wire and device images), and `max_instances` (the maximum number of objects of a particular category per image).

To execute the API, run the following:

```bash
# Run the run.py file
$ bash run.sh
```

## **Result**
You can now find the generated dataset in the `datasets` folder. The `datasets/zip/` folder provides a compressed .zip file of the generated dataset. An example annotation is visualized with the detectron2 visualizer below.

<p align="center">
  <img src="https://github.com/RMDLO/COCOpen-OpenCV/blob/main/docs/images/0.png?raw=true" title="Visualization of COCOpen Automatic Instance Segmentation">
</p>
