# **UIUC COCOpen**

## **Abstract**
The UIUC COCOpen library is a scalable and open source method of generating a labelled dataset of colored images of any object category automatically. The library annotates each object with its unique category identification number, bounding box, and instance segmentation mask in the Microsoft Common Objects in Context (COCO) format [1]. This repository uses the UIUC COCOpen Library to generate the UIUC wires dataset, a dataset of images comprising instances of wires and networking devices, for training wire object instance segmentation models [2].

## **Brief Problem Statement**
- Microsoft’s Common Objects in Context (COCO) dataset which consists of 91 object categories, with 2.5 million labeled instances across 328k images, was labeled manually by workers at AMT.
- The labeling process involved Category labeling, Instance spotting and Instance segmentation.
- This took 81,168 worker hours and cost a lot of money.
- In our lab, the UIUC wires validation dataset, consisting of 663 labels, took two people 57 hours.
- This problem is widely prevalent in the field of computer vision.

## **Proposed Solution**

COCOpen performs the following tasks to automatically generate labeled object instance data.

1. Read an image of a single object against a blank background from cloud storage uploaded by the user.
2. Apply color thresholding and contour detection to the image to automatically obtain an object bounding box and instance segmentation mask.
3. Mask the original object image and randomly apply color, hue, orientation, and scale jittering augmentations.
4. Combine the masked object image with other labeled and masked single-object images into a single image using the Copy-Paste Augmentation technique [3].
5. Apply the combined image to a randomly selected background image.
6. Save all image names and annotations to a dictionary file which can be used to load the data to train object detection, localization, and instance segmentation models.

<p align="center">
  <img src="https://github.com/RMDLO/.github/blob/master/images/lucid_chart_cocopen_1.png" width="350" title="API workflow chart">
</p>

## **Getting Started**
### **Installing Anaconda**
COCOpen-OpenCV uses an anaconda environment to manage versions of all dependencies. To get started with installing `conda` please follow [these instructions](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

### **Cloning COCOpen-OpenCV repository**
Clone this repository in your desired location by running the following command in a terminal:
```bash
# Clone the repository
$ git clone https://github.com/RMDLO/COCOpen-OpenCV.git
```
### **Creating Anaconda Environment**
For ease of creating a conda environment we have provided you with an `environment.yml` file in the root directory of this repository.

> Note : The first line of the yml file sets the new environment's name.

To create the `conda` environment from the `environment.yml` file run:
```bash
# Navigate into the COCOpen directory
$ cd COCOCpen-OpenCV
# Clone the conda environment
COCOCpen-OpenCV$ conda env create -f config/environment.yml
# Activate the conda environment
COCOpen-OpenCV$ conda activate cocopen-env
```
## **Running the API**

### **Adjusting parameters**
Open `config/parameters.yml` file.

Here you can tweak parameters like `dataset_name` (the name of the generated dataset directory), `train_images` (the number of images in the generated training set) and `max_instances` (the maximum number of objects of a particular category per image).

### **Running the script**
To execute the API, run the following command
```bash
# Run the run.py file
(cocopen-env) COCOpen-OpenCV$ python ./src/run.py
```
### **Result**
You can now find the generated dataset in the `datasets` folder. Furthermore, in `datasets/zip/` you can access a zip file containing the generated dataset.

## References
<a id="1">[1]</a> 
T. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. Lawrence Zitnick, "Microsoft COCO: Common Objects in Context," in Eur. Conf. Comput. Vis. (ECCV), Sep. 2014, pp. 740-755.

<a id="2">[2]</a> 
K. He, G. Gkioxari, P. Dollár, and R. Girshick, "Mask R-CNN," in IEEE Int. Conf. Comput. Vis. Pattern Recognit. (CVPR), Oct. 2017, pp. 2980-2988.

<a id="3">[3]</a> 
G. Ghiasi, Y. Cui, A. Srinivas, R. Qian, T. Lin, E. Cubuk, Q. Le, and B. Zoph, "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation," in IEEE Int. Conf. Comput. Vis. Pattern Recognit. (CVPR), June 2020, pp. 9796-9805.
