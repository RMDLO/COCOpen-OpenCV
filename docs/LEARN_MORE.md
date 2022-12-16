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
  <img src="https://github.com/RMDLO/COCOpen-OpenCV/blob/review/.github/images/COCOpen-working-flowchart.png" title="API workflow chart">
</p>

> Click [here](https://lucid.app/lucidchart/0a4a27fb-8f8a-474d-a976-541deb310a02/edit?viewport_loc=167%2C1568%2C2219%2C1041%2C0_0&invitationId=inv_83183c78-2c16-47fc-9b46-14bd97aa2f5f) to visualize the workflow chart