# Import libraries
import os
import shutil
import json
from tqdm import tqdm
import random
import numpy as np
import copy
from PIL import Image, ImageEnhance
import cv2
from pycocotools import mask as pycocomask
from azure.storage.blob import BlobServiceClient

# Class for the cocopen object
class COCOpen:
    # Constructor
    def __init__(
        self,
        parameters: dict,
    ) -> None:
        # Initializing parameters
        self.parameters = parameters

        # Initializing root and destination directory
        self.root_dir = self.parameters["directory"]["root_dir"]
        self.dataset_directory_name = self.parameters["directory"][
            "dataset_directory_name"
        ]

        # Initializing supercategories dictionary
        self.categories = []

        # Saving all directory names
        self.dataset_dir = self.root_dir + f"/datasets/{self.dataset_directory_name}"
        self.train = self.dataset_dir + "/train"
        self.val = self.dataset_dir + "/val"

        # Initialize height and width
        self.height = parameters["shape"]["height"]
        self.width = parameters["shape"]["width"]

        # Initialize scale jittering parameters
        self.apply_scale_jittering = self.parameters["scale_jittering"][
            "apply_scale_jittering"
        ]
        self.individual_scale_jittering = self.parameters["scale_jittering"][
            "individual_scale_jittering"
        ]
        self.scale_factor_min = self.parameters["scale_jittering"]["scale_factor_min"]
        self.scale_factor_max = self.parameters["scale_jittering"]["scale_factor_max"]

        # Initialize color augmentation parameters
        self.apply_color_augmentation = self.parameters["color_augmentation"][
            "apply_color_augmentation"
        ]
        self.individual_color_augmentation = self.parameters["color_augmentation"][
            "individual_color_augmentation"
        ]
        self.change_saturation = self.parameters["color_augmentation"][
            "change_saturation"
        ]
        self.change_brightness = self.parameters["color_augmentation"][
            "change_brightness"
        ]
        self.change_contrast = self.parameters["color_augmentation"]["change_contrast"]
        self.change_hue = self.parameters["color_augmentation"]["change_hue"]
        self.enhancer_min = self.parameters["color_augmentation"]["enhancer_min"]
        self.enhancer_max = self.parameters["color_augmentation"]["enhancer_max"]
        self.color_augmentation_on_combined_image = self.parameters[
            "color_augmentation"
        ]["color_augmentation_on_combined_image"]

    # Making new directories
    def make_new_dirs(self) -> None:
        """
        Making new directories for the COCOpen dataset
        """
        try:
            os.mkdir("./datasets")
        except:
            print("datasets directory already exists!")
        try:
            os.mkdir(self.dataset_dir)
        except:
            print(f"{self.dataset_dir} already exists!")
        try:
            os.mkdir(self.train)
        except:
            print(f"{self.train} directory already exists!")
        try:
            os.mkdir(self.val)
        except:
            print(f"{self.val} directory already exists!")
        print("Created Directories")

    # Generating segmentation annotations in object semantics format
    def object_semantics(
        self,
        coco,
        ann_id,
        img_id,
        file_name=None,
        mask=None,
        category_id=None,
    ):
        """
        This function generates segmentation annotations with the object semantics format.
        """
        mask = np.asfortranarray(mask)
        bitmask = mask.astype(np.uint8)
        file_name = str(img_id) + ".png"

        segmentation = pycocomask.encode(bitmask)
        segmentation["counts"] = segmentation["counts"].decode("ascii")
        area = int(pycocomask.area(segmentation))
        bbox = cv2.boundingRect(mask)

        annotation = {
            "category_id": category_id,
            "segmentation": segmentation,
            "image_id": img_id,
            "id": ann_id,
            "iscrowd": 0,
            "area": area,
            "bbox": bbox,
            "image_name": file_name,
        }
        coco["annotations"].append(annotation)
        ann_id += 1

        return coco, ann_id

    # Initializing Azure connection for for accessing blob storage
    def init_azure(
        self,
    ) -> None:
        """
        Initializing Azure connection for for accessing blob storage
        """
        # Initializing connection with Azure storage account
        connection_string = self.parameters["directory"][
            "AZURE_STORAGE_CONNECTION_STRING"
        ]
        blob_service_client = BlobServiceClient.from_connection_string(
            conn_str=connection_string
        )
        # Creating a dictionary mapping category names to blob_service_client of that category
        self.category_to_container_client = {}
        for category in self.categories:
            self.category_to_container_client[
                category["name"]
            ] = blob_service_client.get_container_client(category["name"])

    # Creating foreground image list
    def create_image_list(self) -> None:
        """
        Creating foreground image list from images on Azure and splitting the list into 'train' and 'val' sets
        """
        # Create a dictionary mapping category names to list of foreground images of that category
        self.category_to_train_image_list = {}
        self.category_to_val_image_list = {}
        for category in self.categories:
            # Creating list of all foreground images of that category
            azure_category_all_image_list = self.category_to_container_client[
                category["name"]
            ].list_blobs()
            # Splitting foreground images into 'train' and 'val' sets
            train_image_list = []
            val_image_list = []
            for blob in azure_category_all_image_list:
                rand = random.random()
                if 0 <= rand < self.parameters["dataset_params"]["train_split"]:
                    train_image_list.append(blob.name)
                else:
                    val_image_list.append(blob.name)
            self.category_to_train_image_list[category["name"]] = train_image_list
            self.category_to_val_image_list[category["name"]] = val_image_list

    # Downloading wire foreground image from Azure
    def download_image_from_azure(self, img, category):
        """
        Downloads wire image from Azure blob storage
        """
        # Get blob and read image data into memory cache
        blob = self.category_to_container_client[category].get_blob_client(img)
        img_data = blob.download_blob().readall()
        # Convert image data to numpy array
        img_array = np.asarray(bytearray(img_data), dtype=np.uint8)
        # Decode image
        src = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return src

    # Getting contour filters
    def contour_filter(self, frame, contour_max_area=2075000):
        """
        Perform contour filtering
        """
        contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new_frame = np.zeros(frame.shape, np.uint8)
        for i, contour in enumerate(contours):
            c_area = cv2.contourArea(contour)
            if self.parameters["contour_threshold"] <= c_area <= contour_max_area:
                mask = np.zeros(frame.shape, np.uint8)
                cv2.drawContours(mask, contours, i, 255, cv2.FILLED)
                mask = cv2.bitwise_and(frame, mask)
                new_frame = cv2.bitwise_or(new_frame, mask)
        frame = new_frame

        return frame

    # Generate super categories, used when an object super category (like "wire") may contain subcategories (like "ethernet")
    def generate_supercategories(self):
        """
        Generate dictionary for super categories based on parameters .yaml file
        """
        for key in self.parameters["categories"]:
            supercategory_dict = {
                "supercategory": key,
                "id": self.parameters["categories"][key],
                "name": key,
            }
            self.categories.append(supercategory_dict)
        print("Generated Categories Dictionary from Parameters")

    # Returning wire information
    def get_object_info(self, img, category):
        """
        This function returns the image array and the mask array
        """
        # Get mask
        src = self.download_image_from_azure(img=img, category=category)
        median = cv2.medianBlur(src, 9)  # Blur image
        gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        _, mask = cv2.threshold(
            gray, self.parameters["color_threshold"][category], 255, cv2.THRESH_BINARY
        )  # Color threshold step
        new_mask = self.contour_filter(mask)  # blob filtering step

        # Encode
        mask_array = np.asarray(new_mask, order="F")

        return src, mask_array

    # Scaling image
    def scale_image(self, img_arr, masks, scale):
        """
        This function scales the input image and outputs the updated image array and mask array
        """
        # resize image and mask
        new_img_arr = cv2.resize(
            img_arr, (int(self.width * scale), int(self.height * scale))
        )
        new_masks = []
        for mask in masks:
            new_masks.append(
                cv2.resize(mask, (int(self.width * scale), int(self.height * scale)))
            )

        # depend on if the image is smaller or larger than width x height, perform different actions
        # images directly gotten from labelbox has 4 channels
        if len(img_arr[0][0]) == 4:
            final_img_arr = np.zeros((self.height, self.width, 4)).astype("uint8")
        else:
            final_img_arr = np.zeros((self.height, self.width, 3)).astype("uint8")

        final_masks = []
        for i in range(0, len(new_masks)):
            final_masks.append(np.zeros((self.height, self.width)).astype("uint8"))

        # if the scaled image is smaller than original
        if (
            int(self.width * scale) < self.width
            or int(self.height * scale) < self.height
        ):
            # fisr detemrine the range the upper left corner can take
            max_h_offset = self.width - int(self.width * scale)
            max_v_offset = self.height - int(self.height * scale)
            upper_left_x = int(random.random() * max_h_offset)
            upper_left_y = int(random.random() * max_v_offset)

            # concat arrays
            for i in range(0, len(final_masks)):
                for j in range(0, int(self.height * scale)):
                    final_img_arr[
                        j + upper_left_y,
                        upper_left_x : (upper_left_x + int(self.width * scale)),
                    ] = new_img_arr[j]
                    final_masks[i][j + upper_left_y][
                        upper_left_x : (upper_left_x + int(self.width * scale))
                    ] = new_masks[i][j]

        # if the scaled image is larget than original
        else:
            # fisr detemrine the range the upper left corner can take
            max_h_offset = len(new_img_arr[0]) - self.width
            max_v_offset = len(new_img_arr) - self.height
            upper_left_x = int(random.random() * max_h_offset)
            upper_left_y = int(random.random() * max_v_offset)

            # concat arrays
            final_img_arr = new_img_arr[
                upper_left_y : (upper_left_y + self.height),
                upper_left_x : (upper_left_x + self.width),
            ]
            for i in range(0, len(final_masks)):
                final_masks[i] = new_masks[i][
                    upper_left_y : (upper_left_y + self.height),
                    upper_left_x : (upper_left_x + self.width),
                ]

        return final_img_arr, final_masks

    # Color augmentation
    def color_augmentation(
        self,
        img_arr,
        enhancer_range,
        change_brightness,
        change_contrast,
        change_saturation,
        change_hue,
    ):
        """
        This function takes np.ndarray as input, performs color augmentation using PIL, then output the new np.ndarray
        """
        new_img_arr = Image.fromarray(img_arr)

        # this step is extremely slow
        if change_hue == True:
            rgb_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            hsv_img = Image.fromarray(rgb_img)
            hsv_img = hsv_img.convert("HSV")
            # convert back to np array for operations
            hsv_img = np.array(hsv_img)

            # randomly change hue
            rand_value = int(random.random() * 255 * 2) - 255

            hsv_img_int = hsv_img.astype("int32")
            hsv_img_int[:, :, 0] += rand_value + 255
            hsv_img_int[:, :, 0] = hsv_img_int[:, :, 0] % 255
            hsv_img = hsv_img_int.astype("uint8")

            # convert back to PIL image
            new_img_arr = Image.new("HSV", (self.width, self.height), (0, 0, 0))
            new_img_arr = Image.fromarray(hsv_img, mode="HSV")
            new_img_arr = new_img_arr.convert("RGB")

        if change_brightness == True:
            modifier = ImageEnhance.Brightness(new_img_arr)
            new_img_arr = modifier.enhance(
                enhancer_range[0]
                + random.random() * (enhancer_range[1] - enhancer_range[0])
            )
        if change_contrast == True:
            modifier = ImageEnhance.Contrast(new_img_arr)
            new_img_arr = modifier.enhance(
                enhancer_range[0]
                + random.random() * (enhancer_range[1] - enhancer_range[0])
            )
        if change_saturation == True:
            modifier = ImageEnhance.Color(new_img_arr)
            new_img_arr = modifier.enhance(
                enhancer_range[0]
                + random.random() * (enhancer_range[1] - enhancer_range[0])
            )

        return np.array(new_img_arr)

    # Random operations on image data
    def random_operations(self, img, angle, flip_h, flip_v):
        """
        Rotates and flips the input image
        """
        # rotation
        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(
            image_center,
            angle,
            self.width
            / (self.height * self.width / np.sqrt(self.width**2 + self.height**2)),
        )
        img_new = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        # flip
        if flip_h:
            # horizontal flip
            img_new = np.flip(img_new, 1)
        if flip_v:
            # vertical flip
            img_new = np.flip(img_new, 0)
        # if fx_scale and fy_scale:
        #     img_new = cv2.resize(img_new, dsize = (self.width, self.height), fx=fx_scale, fy=fy_scale)
        return img_new

    # Combining foreground and background images
    def combine(
        self,
        dataset_type: str,
        target_dir,
        num_images,
    ):
        """
        Combining foreground and background images
        """

        coco_sem = {
            "images": [],
            "annotations": [],
            "categories": self.categories,
        }

        # Creating a copy of category_to_train_img_list and category_to_val_img_list based on dataset_type
        image_list = {}
        if dataset_type == "train":
            image_list = copy.deepcopy(self.category_to_train_image_list)
        else:
            image_list = copy.deepcopy(self.category_to_val_image_list)
        
        # this sets the lower/upper limit of color augmentation
        enhancer_range = [self.enhancer_min, self.enhancer_max]
        scale_range = [self.scale_factor_min, self.scale_factor_max]

        # this needs to be tracked at all time - annotation id must be unique across all instances in the entire dataset
        ann_id_sem = 0

        for i in tqdm(range(num_images)):

            # scale factor for this image
            scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

            image = {
                "id": i,  # this can be updated later in a for loop
                "width": self.width,
                "height": self.height,
                "file_name": str(i) + ".png",
                "source": [], # list of source images used
            }

            # create mapping of category name to max_instances for each category
            category_to_num_instances = {}
            for category in self.categories:
                if category["name"] != "background":
                    category_to_num_instances[category["name"]] = (int(random.random() * self.parameters["max_instances"][category["name"]]) + 1)

            img_list = copy.deepcopy(image_list)

            # arrays storing category image info
            all_img_arr = []
            binary_mask_arr = []
            all_category_ids = []

            # sum of number of all instances of all categories
            total_num_instances = 0

            # get mask info for each category
            for category in self.categories:
                if category["name"] != "background":
                    # get category info
                    for j in range(0, category_to_num_instances[category["name"]]):
                        total_num_instances += 1
                        index = int(len(img_list[category["name"]]) * random.random())

                        # Record source image path (category + filename)
                        image["source"].append(os.path.join(category["name"], img_list[category["name"]][index]))

                        img, mask = self.get_object_info(img_list[category["name"]][index], category["name"])
                        # Raise exception if object image height and width do not match image shape parameter
                        if img.shape[0:2] != (self.height, self.width):
                            raise Exception(
                                f"""Object image height and width do not match image shape parameter
                                            ({img.shape[0]},{img.shape[1]}) ({self.height},{self.width})"""
                            )
                        img_list[category["name"]].pop(index)

                        mask = mask / 255
                        mask = mask.astype("uint8")

                        msk = []
                        msk.append(mask)

                        # scaling
                        if self.apply_scale_jittering:
                            if self.individual_scale_jittering:
                                scale = scale_range[0] + random.random() * (
                                    scale_range[1] - scale_range[0]
                                )
                                img, msk = self.scale_image(img, msk, scale)
                            else:
                                img, msk = self.scale_image(img, msk, scale)

                        # color augmentation
                        if self.apply_color_augmentation:
                            if self.individual_color_augmentation:
                                img = self.color_augmentation(
                                    img,
                                    enhancer_range,
                                    self.change_brightness,
                                    self.change_contrast,
                                    self.change_saturation,
                                    self.change_hue,
                                )

                        # Add the image and mask to the array
                        all_img_arr.append(img)
                        binary_mask_arr.append(msk)
                        all_category_ids.append(category["id"])

            # perform random flip
            for m in range(0, total_num_instances):

                horizontal = int(2 * random.random())
                vertical = int(2 * random.random())

                if horizontal == 1:
                    all_img_arr[m] = cv2.flip(all_img_arr[m], 1)
                    for n in range(0, len(binary_mask_arr[m])):
                        binary_mask_arr[m][n] = np.flip(binary_mask_arr[m][n], 1)

                if vertical == 1:
                    all_img_arr[m] = cv2.flip(all_img_arr[m], 0)
                    for n in range(0, len(binary_mask_arr[m])):
                        binary_mask_arr[m][n] = np.flip(binary_mask_arr[m][n], 0)

            # now have a complete list of image arrays and masks, can start combining
            # choose a random element from the img list
            final_img = []

            # choose the first image
            randint = int(random.random() * len(all_img_arr))

            # only ethernet and device
            category_id = all_category_ids[randint]

            img_id = i
            mask_array_1 = []

            # update coco info
            for mask1 in binary_mask_arr[randint]:

                mask_array_1.append(mask1)

                msk1 = np.zeros((self.height, self.width, 3)).astype("uint8")
                msk1[:, :, 0] = mask1
                msk1[:, :, 1] = mask1
                msk1[:, :, 2] = mask1
                final_img = cv2.bitwise_or(
                    msk1.astype("uint8"), all_img_arr[randint], mask=mask1
                )

                coco_sem, ann_id_sem = self.object_semantics(
                    coco=coco_sem,
                    ann_id=ann_id_sem,
                    img_id=img_id,
                    file_name=None,
                    mask=mask1,
                    category_id=category_id,
                )

            # save mask1 info for later use
            mask_array_1 = np.dstack(binary_mask_arr[randint])
            mask_array_1 = np.max(mask_array_1, axis=2).astype("uint8")

            # prevent repetition
            all_img_arr.pop(randint)
            binary_mask_arr.pop(randint)
            all_category_ids.pop(randint)

            # combine the rest
            for j in range(1, total_num_instances):

                # choose the second image
                randint2 = int(random.random() * len(all_img_arr))
                msk2 = np.zeros((self.height, self.width, 3)).astype("uint8")
                mask_array_2 = []

                category_id = all_category_ids[randint2]

                # update coco info
                for mask2 in binary_mask_arr[randint2]:
                    mask2 = mask2 & (~mask_array_1)
                    mask_array_2.append(mask2)

                    msk2 = np.zeros((self.height, self.width, 3)).astype("uint8")
                    msk2[:, :, 0] = mask2
                    msk2[:, :, 1] = mask2
                    msk2[:, :, 2] = mask2
                    masked_layer = cv2.bitwise_or(
                        msk2.astype("uint8"), all_img_arr[randint2], mask=mask2
                    )
                    final_img = cv2.add(final_img, masked_layer)

                    coco_sem, ann_id_sem = self.object_semantics(
                        coco_sem,
                        ann_id_sem,
                        img_id,
                        mask=mask2,
                        category_id=category_id,
                    )

                mask_array_2 = np.dstack(mask_array_2)
                mask_array_2 = np.max(mask_array_2, axis=2).astype("uint8")

                # update mask_array_1 to be the combination of img1 and img2
                mask_array_1 = mask_array_1 | mask_array_2

                # prevent repetition
                all_img_arr.pop(randint2)
                binary_mask_arr.pop(randint2)
                all_category_ids.pop(randint2)

            # add a random background
            randbg = int(random.random() * len(image_list["background"]))
            bg = image_list["background"][randbg]
            # Download background image from azure
            src = self.download_image_from_azure(img=bg, category="background")
            bg_arr = src.astype("uint8")
            # Raise exception if background image height and width do not match image shape parameter
            if bg_arr.shape[0:2] != (self.height, self.width):
                raise Exception(
                    f"""Background image height and width do not match image shape parameter
                                ({bg_arr.shape[0]},{bg_arr.shape[1]}) ({self.height},{self.width})"""
                )

            # background random operations
            bg_rot = random.random() * 360
            bg_flip_h = bool(random.random() / 0.5)
            bg_flip_v = bool(random.random() / 0.5)
            bg_arr = self.random_operations(bg_arr, bg_rot, bg_flip_h, bg_flip_v)

            mask_bg = np.ones((self.height, self.width)).astype("uint8")
            msk_bg = np.zeros((self.height, self.width, 3)).astype("uint8")
            mask_bg = mask_bg & (~mask_array_1)  # & (~ mask_array_2)

            msk_bg[:, :, 0] = mask_bg
            msk_bg[:, :, 1] = mask_bg
            msk_bg[:, :, 2] = mask_bg

            masked_bg = cv2.bitwise_or(msk_bg.astype("uint8"), bg_arr, mask=mask_bg)

            final_img = cv2.add(final_img, masked_bg)

            # color augmentation
            if (
                self.individual_color_augmentation
                and self.color_augmentation_on_combined_image
                and self.apply_color_augmentation
            ):
                final_img = self.color_augmentation(
                    final_img,
                    enhancer_range,
                    self.change_brightness,
                    self.change_contrast,
                    self.change_saturation,
                    False,
                )
            elif self.apply_color_augmentation and (
                not self.individual_color_augmentation
            ):
                final_img = self.color_augmentation(
                    final_img,
                    enhancer_range,
                    self.change_brightness,
                    self.change_contrast,
                    self.change_saturation,
                    False,
                )

            cv2.imwrite(os.path.join(target_dir, str(i) + ".png"), final_img)
            coco_sem["images"].append(image)

        return coco_sem

    # Generating training dataset
    def generate_train_data(self) -> None:
        """
        Generating training dataset
        """
        # generate train
        print("Generating 'train' data")
        coco_sem = self.combine(
            dataset_type="train",
            target_dir=self.train,
            num_images=self.parameters["dataset_params"]["train_images"],
        )
        f = open(self.train + "/train.json", "w")
        json.dump(coco_sem, f, sort_keys=True, indent=4)
        f.close()

    # Generating val dataset
    def generate_val_data(self) -> None:
        """
        Generating val dataset
        """
        print("Generating 'val' data")
        coco_sem = self.combine(
            dataset_type="val",
            target_dir=self.val,
            num_images=self.parameters["dataset_params"]["val_images"],
        )
        f = open(self.val + "/val.json", "w")
        json.dump(coco_sem, f, sort_keys=True, indent=4)
        f.close()

    # Creating zip file of the dataset
    def zip(self, base_name: str, root_dir: str, format="zip") -> None:
        """
        Function which zips all files in a directory
        """
        shutil.make_archive(format=format, base_name=base_name, root_dir=root_dir)
        print(self.dataset_directory_name + ".zip file successfully created!")