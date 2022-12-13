<p align="left">
  <img src="https://github.com/RMDLO/COCOpen-OpenCV/blob/review/demo/visualization/uiuc_logo.png" title="UIUC logo">
</p>

# **UIUC COCOpen**

## **Abstract**
The UIUC COCOpen library is a scalable and open source method of generating a labelled dataset of colored images of any object category automatically. The library annotates each object with its unique category identification number, bounding box, and instance segmentation mask in the Microsoft Common Objects in Context (COCO) format [1]. This repository uses the UIUC COCOpen Library to generate the UIUC wires dataset, a dataset of images comprising instances of wires and networking devices, for training wire object instance segmentation models [2].

<p align="center">
  <img src="https://github.com/RMDLO/COCOpen-OpenCV/blob/review/demo/visualization/0.png" width="1000" title="Visualization of COCOpen Automatic Instance Segmentation">
</p>

<p align="center">
  <img src="https://github.com/RMDLO/COCOpen-OpenCV/blob/review/demo/masks/0.png" width="1000" title="Visualization of COCOpen Object Instance Masks">
</p>

## **Learn more about COCOpen**
To learn more about COCOpen, click [here](.github/LEARN_MORE.md).

## **Installation / Setup**
See [installation instructions](.github/INSTALLATION.md)

## **Getting Started**
See [getting started with COCOpen](.github/GETTING_STARTED.md) to learn about the basic usage.

## **Example Run**
See [example run](.github/EXAMPLE_RUN.md) to see a demo of us generating the UIUC wires dataset in COCO format.

## References
<a id="1">[1]</a> 
T. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. Lawrence Zitnick, "Microsoft COCO: Common Objects in Context," in Eur. Conf. Comput. Vis. (ECCV), Sep. 2014, pp. 740-755.

<a id="2">[2]</a> 
K. He, G. Gkioxari, P. Dollár, and R. Girshick, "Mask R-CNN," in IEEE Int. Conf. Comput. Vis. Pattern Recognit. (CVPR), Oct. 2017, pp. 2980-2988.

<a id="3">[3]</a> 
G. Ghiasi, Y. Cui, A. Srinivas, R. Qian, T. Lin, E. Cubuk, Q. Le, and B. Zoph, "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation," in IEEE Int. Conf. Comput. Vis. Pattern Recognit. (CVPR), June 2020, pp. 9796-9805.
