# **Getting Started**
This notebook contains all basic information to run COCOpen.

Once you have completed the [installation / setup](./INSTALLATION.md) follow the below instructions.

## **Open COCOpen**
Start off by navigating into the COCOpen repository.

```bash
# Navigate into the COCOpen directory
$ cd COCOCpen-OpenCV
```

## **User configurations**
1. Copy `connection string` from Azure Storage Account. Click [here](https://learn.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage?toc=%2Fazure%2Fstorage%2Fblobs%2Ftoc.json&bc=%2Fazure%2Fstorage%2Fblobs%2Fbreadcrumb%2Ftoc.json&tabs=azure-portal#view-account-access-keys) to learn how to access it.

2. Paste the connection string in the `config/parameters.yml` file under

```bash
# User defined parameters
user_defined:
  root_dir: "." # ignore
  dataset_directory_name: "cocopen-dataset-4" # ignore
  AZURE_STORAGE_CONNECTION_STRING: '<paste here within single quotes>'
```

## **Running the API**

### **Adjusting parameters**
Open the `config/parameters.yml` file.

Here you can tweak parameters like `dataset_name` (the name of the generated dataset directory), `train_images` (the number of images in the generated training set), `threshold` (color thresholding values - we recommend keeping the default values for the provided wire and device images), and `max_instances` (the maximum number of objects of a particular category per image).

### **Running the script**
To execute the API, run the following:

```bash
# Run the run.py file
(cocopen-env) COCOpen-OpenCV$ bash run.sh
```

### **Result**
You can now find the generated dataset in the `datasets` folder. The `datasets/zip/` folder provides a compressed .zip file of the generated dataset. Example annotations are provided in the images below.