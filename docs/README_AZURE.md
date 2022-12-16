# **Configure Azure Storage Container**

## **1. Register for an Azure Account**

If you do not already have an account with Azure, [create one](https://azure.microsoft.com/en-us/).

## **2. Create a Storage Account**

The Azure storage account houses individual storage containers which act like file folders on the cloud. Each individual storage container contains the images for the project's dataset. Azure provides [instructions on how to create a storage account](https://learn.microsoft.com/en-us/azure/storage/common/storage-account-create?toc=/azure/storage/blobs/toc.json&bc=/azure/storage/blobs/breadcrumb/toc.json).

## **3. Create and Manage Storage Containers**

Each individual storage container will store its own unique object category. Example categories for a dataset may include "cats," "dogs," and "background." Each of these categories requires its own storage container to store the single-object images of that category. To open-source our data, we also set the access level for our container to `Container (anonymous public read access for containers and blobs)`. This enables anonymous connection to our container.  To learn more about creating and managing storage containers, read [Manage Blob Containers Using the Azure Portal](https://learn.microsoft.com/en-us/azure/storage/blobs/blob-containers-portal).

## **4. Upload Images to Storage Container**

Use the Azure Portal to upload images to a storage container.

1. First, click on the storage container you would like to upload images to. Our storage containers are shown below as an example.

<p align="center">
  <img src="https://github.com/RMDLO/COCOpen-OpenCV/blob/main/docs/images/storage_containers_image.png" width="700" height="200" title="Storage Containers Image">
</p>

2. Next, click 'Upload'.

3. Select all image files you want to upload from local storage and begin the upload process.
