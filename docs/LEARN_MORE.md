# **COCOpen**

COCOpen is a scalable method of generating a dataset of colored images of any object category automatically labelled with unique object instance segmentation masks in the Microsoft Common Objects in Context (COCO) format [1]. It uses the OpenCV library to perform rotations, flips, hue jittering, and brightness jittering augmentations [2]. It uses contour (blob) filtering and color thresholding on images of single objects against black backgrounds to obtain clean image annotations automatically. It masks the individual object images and performs a "copy-paste" image combination operation whereby individual object and background images are combined to create a labeled dataset with occlusion. This technique enables users to generate an instance segmentation dataset

* With a total number of unique synthetic images combinatorial in the number of unique input images.
* With automatically-generated object instance segmentation, bounding box, and category labels.
* With the inclusion of any object category for training models tailored to specific applications.
* With automatic handling of object occlusion.

## **COCOpen Workflow**

COCOpen performs the following tasks to automatically generate labeled object instance data.

1. Read an image of a single object against a blank background from cloud storage uploaded by the user.
2. Apply color thresholding and contour detection to the image to automatically obtain an object bounding box and instance segmentation mask.
3. Mask the original object image and randomly apply color, hue, orientation, and scale jittering augmentations.
4. Combine the masked object image with other labeled and masked single-object images into a single image using the Copy-Paste Augmentation technique [3].
5. Apply the combined image to a randomly selected background image.
6. Save all image names and annotations to a dictionary file which can be used to load the data to train object detection, localization, and instance segmentation models.

This workflow is shown in the figure below.

<p align="center">
  <img src="https://github.com/RMDLO/COCOpen-OpenCV/blob/review/docs/images/COCOpen.png?raw=true" title="COCOpen Workflow" width="700px"> 
</p>

## **Custom Data**

The UIUCWires dataset stored on Azure is provided to demonstrate COCOpen by default. The dataset includes images of single wire and single devices against a black background. It also includes a folder of background images which are applied to generate background noise in the dataset as a form of data augmentation. Each image type (wire, device, and background) is stored in its own Azure storage container labeled with the object's category. All single-object images were captured with an Intel RealSense d435 camera with 1920x1080 resolution. The table below shows the number of images by object category.

<div align="center">

|            	| Category 	|               |
|:----------:	|:--------:	|:-------------:|
| **device** 	| **wire** 	| **background**|
|     440    	|   5577   	|       90      |
</div>

The color contrast between the objects in each image and the background allows for using color thresholding to automatically annotate the original images. A user who wishes to use COCOpen to perform copy-paste augmentation on their own dataset will need to ensure the scene images only contain a single object. The scene for the images fed into COCOpen must also minimize shadowing. In our experience, the scene object can generate a shadow onto itself and the camera apparatus can generate a shadow onto the scene. For best results, we recommend checking lighting conditions before collecting data.

After users generate their own dataset of single-object RGB images, the images should be uploaded to individual storage containers on Azure. The container should contain only folders of images, and the title of each container of images should correspond to the object category label (all lower-case). See our [Configure Azure Storage Container](https://github.com/RMDLO/COCOpen-OpenCV/blob/main/docs/README_AZURE.md) documentation for more details and to see the directory structure for our container.

## **References**

<a id="1">[1]</a> 
T. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. Lawrence Zitnick, "Microsoft COCO: Common Objects in Context," in Eur. Conf. Comput. Vis. (ECCV), Sep. 2014, pp. 740-755. [[paper]](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48)

<a id="2">[2]</a> 
OpenCV, "Open Source Computer Vision Library," (2015).