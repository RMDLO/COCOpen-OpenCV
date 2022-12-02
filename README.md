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

![API workflow chart](https://github.com/RMDLO/.github/blob/master/images/lucid_chart_cocopen_1.png)
<br>

1. Reads images of single objects against blank backgrounds from cloud storage uploaded by the user.
2. Automatically obtains object instance segmentation mask labels using color thresholding and contour detection.
3. Masks the original object image and randomly applies color, hue, orientation, and scale jittering augmentations.
4. It combines the masked object image with other labeled and masked single-object images into a combined image using the Copy-Paste Augmentation technique.
5. It merges the combined image with a background image.
6. It saves all image names and annotations to a dictionary file which can be used to load the data to train object detection models.


## Installation

Using the RMDLO COCOpen library requires desktop [configuration of a GitHub SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

To install dependencies using `conda`, perform the below command in a terminal.
```bash
# Clone the repository
$ git clone git@github.com:RMDLO/COCOpen.git
# Install dependencies
$ cd COCOpen
# In the environment.yml file, change `name` to the name you would like for the conda environment and run
$ conda env create -f environment.yml
# Activate the environment
$ conda activate <env name>
```