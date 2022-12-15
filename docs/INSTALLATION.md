# **Installation**
This notebook contains all installation and setup related information.

### **Environment**

Installation and execution of COCOpen was verified with the below environment.
- Operating System: Ubuntu 20.04.5 LTS
- Kernel: Linux 5.15.0-56-generic
- Architecture: x86-64
- Python: 3.9.15

For details regarding package dependencies, please see [here](config/environment.yml).

## **Cloning COCOpen-OpenCV repository**
Clone this repository in your desired location by running the following command in a terminal:
```bash
# Clone the repository
$ git clone https://github.com/RMDLO/COCOpen-OpenCV.git
```

## **Installing Anaconda**
COCOpen-OpenCV uses an anaconda environment to manage versions of all dependencies. To get started with installing `conda` please follow [these instructions](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

## **Creating Anaconda Environments**
For ease of creating a conda environment, COCOpen provides an `environment.yaml` file in the `config/` directory of this repository. The first line of the `environment.yaml` file defines the name of the new environment. This environment is used to generate a synthetic dataset using `src/cocopen.py`.

To create the `conda` environment from the `environment.yaml` file, run:

```bash
# Navigate into the COCOpen directory
$ cd COCOCpen-OpenCV
# Clone the conda environment
$ conda env create -f config/environment.yaml
# Activate the conda environment
$ conda activate cocopen
```

To visualize a generated dataset, we provide a `demo_environment.yml` file which contains a cpu-only installation of PyTorch 1.10. We used the [detectron2](https://github.com/facebookresearch/detectron2) library to perform visualization. This library cannot be installed with `conda` because it will not build properly with PyTorch. To use the `src/demo.py` file to visualize a dataset, please run:

```bash
# Navigate into the COCOpen directory
$ cd COCOCpen-OpenCV
# Clone the conda environment
$ conda env create -f config/demo_environment.yaml
# Activate the conda environment
$ conda activate cocopen-demo
# Install the prebuilt detectron2 library
$ python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
```

To train an object detection model, we provide a `train_environment.yml` file which contains an installation of PyTorch 1.10 with CUDA 11.3. We use the [detectron2](https://github.com/facebookresearch/detectron2) library to train detection models. To set up a conda enviornment to use the `src/train.py` file to train and predict on a dataset, please run:

```bash
# Navigate into the COCOpen directory
$ cd COCOCpen-OpenCV
# Clone the conda environment
$ conda env create -f config/train_environment.yaml
# Activate the conda environment
$ conda activate cocopen-train
# Install the prebuilt detectron2 library
$ python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

## **Setting up Azure Storage Container**
To learn how to set up your dataset on Azure, read [this](./docs/README_AZURE.md).