## **Custom Data**

This repository applies COCOpen to the UIUCWires dataset for demonstration. The dataset includes images of single wires and single devices against a black background. It also includes a folder of background images which are applied to generate background noise in the dataset as a form of data augmentation. These images were captured with an Intel RealSense d435 camera and every image has dimensions 1920x1080. The table below shows the number of images by category.

<div align="center">

|            	| Category 	|               |
|:----------:	|:--------:	|:-------------:|
| **device** 	| **wire** 	| **background**|
|     440    	|   5717   	|       90      |
</div>

The color contrast between the objects in each image and the background allows for using color thresholding to automatically annotate the original images. A user who wishes to use COCOpen to perform copy-paste augmentation on their own dataset will need to ensure the scene images only contain a single object. The scene for the images fed into COCOpen must also minimize shadowing. In our experience, the scene object can generate a shadow onto itself and the camera apparatus can generate a shadow onto the scene. For best results, we recommend checking lighting conditions before collecting data.

After users generate their own single-object RGB images, the images should be uploaded to a container on Azure. The container should contain only folders of images, and the title of the folder of images should correspond to the object category label (all lower-case). See our [getting started with Azure](https://github.com/RMDLO/COCOpen-OpenCV/blob/976083972a07d0fecb5fe4c5c0e6d16d73c7df46/docs/README_AZURE.md) documentation for more details and to see the directory structure for our container.