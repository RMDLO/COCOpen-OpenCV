# **Azure Storage Container Setup**
## **1. Azure Sign up**
If you do not already have an account with Azure, create one [here](https://azure.microsoft.com/en-us/).
## **2. Create a Storage account**
> Note: This is your storage account in which individual storage containers (which contain your dataset images) will reside.

To learn how to create a storage account, click [here](https://learn.microsoft.com/en-us/azure/storage/common/storage-account-create?toc=/azure/storage/blobs/toc.json&bc=/azure/storage/blobs/breadcrumb/toc.json).
## **3. Create and manage Storage Containers**
> Note: These each individual storage container will store its own unique object category. Ex: If you have images of cats, dogs and background, you will need to create a container for cats, dogs and backgrounds.

> Make sure also to set the container's access level by choosing `Container (anonymous read access for containers and blobs)`

To learn how to create and manage storage containers, click [here](https://learn.microsoft.com/en-us/azure/storage/blobs/blob-containers-portal).
## **4. Uploading images to container**
Our recommended method would be to simply use the Azure Portal for this task.

1. First, click on the container you'd like to upload images to.

<p align="center">
  <img src="https://github.com/RMDLO/COCOpen-OpenCV/blob/976083972a07d0fecb5fe4c5c0e6d16d73c7df46/docs/images/storage_containers_image.png?raw=true" width="700" height="200" title="Storage Containers Image">
</p>

2. Next click 'Upload'.

3. Select all image files you want to upload from your local computer and begin the upload process.
