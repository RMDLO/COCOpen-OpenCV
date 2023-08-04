# **Installation**

This guide contains all installation and setup related information.

### **System Requirements**

Installation and execution of COCOpen was verified with the below environment.
- Operating System: Ubuntu 20.04.5 LTS
- Kernel: Linux 5.15.0-56-generic
- Architecture: x86-64
- Python: 3.9.17
- Conda: 22.9.0

See the repository GitHub CI/CD workflow [file](https://github.com/RMDLO/COCOpen-OpenCV/blob/main/.github/workflows/env.yaml) for more information on system and python compatibility.

For detailed versions of package dependencies, please see [`config/data_environment.yaml`](https://github.com/RMDLO/COCOpen-OpenCV/blob/main/config/data_environment.yaml).

## **Clone Repository**

Clone this repository with:

```bash
git clone https://github.com/RMDLO/COCOpen-OpenCV.git
```

## **Demonstrate Data Generation**

COCOpen-OpenCV uses an conda environment to manage versions of all dependencies. To get started with installing `conda` please follow [these instructions](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

For ease of creating a conda environment, COCOpen provides an `data_environment.yaml` file in the `config/` directory of this repository. The first line of the `data_environment.yaml` file defines the name of the new environment. This environment is used to generate a synthetic dataset using `src/cocopen.py`. To visualize the generated dataset, we include dependencies for the object detection library we use, [detectron2](https://github.com/facebookresearch/detectron2). The conda environment includes a cpu-only installation of PyTorch 1.10 on which detectron2 visualization depends. The detectron2 library cannot be installed with `conda` because it will not build properly with PyTorch. To use COCOpen to generate and visualize a dataset, please run the below commands to install dependencies.

Navigate into the COCOpen directory

```bash
cd COCOCpen-OpenCV
```

Clone the conda environment

```bash
conda env create -f config/data_environment.yaml
```

Activate the conda environment

```bash
conda activate cocopen-data
```

Install the prebuilt detectron2 library
```bash
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
```

Verify the `demo_dataset` value in `config/parameters.yaml` is set to `True` to perform visualization.

## **Train an Instance Segmentation Model**

The `train_environment.yaml` requirements file contains all dependencies required for training a detectron2 object instance segmentation model. To set up this conda environment (which the `src/train.py` file requires), please run:

Navigate into the COCOpen directory

```bash
cd COCOpen-OpenCV
```

Clone the conda environment

```bash
conda env create -f config/train_environment.yaml
```

Activate the conda environment

```bash
conda activate cocopen-train
```

Install the prebuilt detectron2 library

```bash
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

Please also set the `train_dataset` value in `config/parameters.yaml` to `True` to train a model. The model training configuration can be adjusted with user-defined training parameters in `src/train.py`. After training a model, perform inference with the model on validation set images by setting the `predict_dataset` value in `config/parameters.yaml` to `True`.

## **Create a Dataset with Box Cloud Storage**

These instructions are only necessary for using COCOpen-OpenCV With user-supplied image data. Access to the UIUC Wires dataset, comprising wire, device, and background images, is provided by default.

Follow the [instructions](https://github.com/box/box-python-sdk) provided by BoxDev to create a Box application to store input data. To connect to the Box application, modify the Box application settings in `config/config.json` by following these [instructions](https://developer.box.com/guides/authentication/jwt/jwt-setup/#generate-a-keypair-recommended).

## **Run COCOpen**

Run COCOpen with

```bash
./run.sh
```

The generated dataset saves to the `datasets` directory under the root directory of this repository. See [example run](https://github.com/RMDLO/COCOpen-OpenCV/blob/main/docs/EXAMPLE_RUN.md) to see a demonstration of generating a simple dataset of ethernet cables and ethernet devices with category, bounding box, and instance segmentation mask annotations in the COCO format.
