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

## **Creating an Anaconda Environment**
For ease of creating a conda environment, COCOpen provides an `environment.yaml` file in the `config/` directory of this repository. The first line of the `environment.yaml` file defines the name of the new environment.

To create the `conda` environment from the `environment.yaml` file, run:

```bash
# Navigate into the COCOpen directory
$ cd COCOCpen-OpenCV
# Clone the conda environment
COCOCpen-OpenCV$ conda env create -f config/environment.yaml
# Activate the conda environment
COCOpen-OpenCV$ conda activate cocopen
```

## **Setting up Azure Storage Container**
To learn how to set up your dataset on Azure, read [this](./README_AZURE.md).
