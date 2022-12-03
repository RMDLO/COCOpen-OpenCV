# **UIUC COCOpen**

## **Abstract**
COCOpen is a scalable & open source method of generating a dataset of colored images of any object category automatically labelled
with unique object instance segmentation masks in the COCO format.

Currently, the repository demonstrates the performance of the UIUC COCOpen library by using it to generate the UIUC wires dataset, a dataset of images comprising instances of wires and networking devices.


## **Brief Problem Statement**
- Microsoft’s Common Objects in Context (COCO) dataset which consists of 91 object categories, with 2.5 million labeled instances across 328k images, was labeled manually by workers at AMT.
- The labeling process involved Category labeling, Instance spotting and Instance segmentation.
- This took 81,168 worker hours and cost a lot of money.
- In our lab, the UIUC wires validation dataset, consisting of 663 labels, took two people 57 hours.
- This problem is widely prevalent in the field of computer vision.

## **Proposed Solution**

<p align="center">
  <img src="https://github.com/RMDLO/.github/blob/master/images/lucid_chart_cocopen_1.png" width="300" title="API workflow chart">
</p>

1. Reads images of single objects against blank backgrounds from cloud storage uploaded by the user.
2. Automatically obtains object instance segmentation mask labels using color thresholding and contour detection.
3. Masks the original object image and randomly applies color, hue, orientation, and scale jittering augmentations.
4. It combines the masked object image with other labeled and masked single-object images into a combined image using the Copy-Paste Augmentation technique.
5. It merges the combined image with a background image.
6. It saves all image names and annotations to a dictionary file which can be used to load the data to train object detection models.

## **Getting Started**
### **Installing Anaconda**
COCOpen-OpenCV uses an anaconda environment to manage versions of all dependencies. To get started with installing `conda` please follow [these instructions](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

### **Cloning COCOpen-OpenCV repository**
Using the RMDLO COCOpen-OpenCV library requires desktop [configuration of a GitHub SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

Clone this repository in your desired location by running the following command in a terminal:
```bash
# Clone the repository
$ git clone git@github.com:RMDLO/COCOpen-OpenCV.git
```
### **Creating Anaconda Environment**
For ease of creating a conda environment we have provided you with an `environment.yml` file in the root directory of this repository.

> Note : The first line of the yml file sets the new environment's name.

To create the `conda` environment from the `environment.yml` file run:
```bash
# Navigate into the COCOpen directory
$ cd COCOCpen-OpenCV
# Clone the conda environment
COCOCpen-OpenCV$ conda env create -f environment.yml
# Activate the conda environment
COCOpen-OpenCV$ conda activate cocopen-env
```
## **Running the API**

### **Adjusting parameters**
Open `src/run.py` file.

Here you can tweak parameters like `dataset_directory_name` (name of your dataset), `num_of_train_images` (number of training set images you want to generate) and `num_of_val_images` (number of validation set images you want to generate).
### **Running the script**
To execute the API, run the following command
```bash
# Run the run.py file
(cocopen-env) COCOpen-OpenCV$ python ./src/run.py
```
### **Result**
You can now find the generated dataset in the `datasets` folder. Furthermore, in `datasets/zip/` you can access a zip file containing the generated dataset.
