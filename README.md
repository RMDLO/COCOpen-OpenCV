<p align="center">
  <img src="https://github.com/RMDLO/COCOpen-OpenCV/blob/main/docs/images/logo.png?" title="COCOpen Logo">
</p>

The UIUC COCOpen library is a scalable and open source method of generating a labelled dataset of colored images of any object category automatically. The library annotates each object with its unique category identification number, bounding box, and instance segmentation mask in the Microsoft Common Objects in Context (COCO) format [1]. This repository uses the UIUC COCOpen Library to generate the UIUC wires dataset, a dataset of images comprising instances of wires and networking devices, for training wire object instance segmentation models [2].


<p align="center">
  <img src="https://github.com/RMDLO/COCOpen-OpenCV/blob/1ce7c5c82115dcc193adae881033d168e462caba/demo/cocopen-dataset-review/visualization/0.png?raw=true" title="Visualization of COCOpen Automatic Instance Segmentation"> <figcaption>This is an example COCOpen-produced synthetic image containing multiple objects of interest superimposed on a randomly selected background. It visualizes ground truth instance segmentation mask, object category, and bounding box labels.</figcaption>
</p>

## **Learn More**
To learn more about COCOpen, click [here](https://github.com/RMDLO/COCOpen-OpenCV/blob/main/docs/LEARN_MORE.md).

## **Get Started**
See [installation instructions](https://github.com/RMDLO/COCOpen-OpenCV/blob/main/docs/INSTALLATION.md) to learn about configuring a dataset and installing dependencies.

## **Run an Example**
See [example run](https://github.com/RMDLO/COCOpen-OpenCV/blob/main/docs/EXAMPLE_RUN.md) to see a demo automatically generating a simple dataset of ethernet cables and ethernet devices with category, bounding box, and instance segmentation mask annotations in the COCO format.

## **References**
<a id="1">[1]</a> 
T. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. Lawrence Zitnick, "Microsoft COCO: Common Objects in Context," in Eur. Conf. Comput. Vis. (ECCV), Sep. 2014, pp. 740-755. [[paper]](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48)

<a id="2">[2]</a> 
K. He, G. Gkioxari, P. Dollár, and R. Girshick, "Mask R-CNN," in IEEE Int. Conf. Comput. Vis. Pattern Recognit. (CVPR), Oct. 2017, pp. 2980-2988. [[paper]](https://ieeexplore.ieee.org/document/8237584)

<a id="3">[3]</a> 
G. Ghiasi, Y. Cui, A. Srinivas, R. Qian, T. Lin, E. Cubuk, Q. Le, and B. Zoph, "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation," in IEEE Int. Conf. Comput. Vis. Pattern Recognit. (CVPR), June 2020, pp. 9796-9805. [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf)