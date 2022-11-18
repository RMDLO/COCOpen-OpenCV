import os
import shutil
import json
import random
from tqdm import tqdm
import urllib
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from pycocotools import mask as pycocomask
from azure.storage.blob import BlobServiceClient

# Class for the cocopen object
class Cocopen:
    # Constructor
    def __init__(self, root_dir: str, dataset_directory_name: str, num_of_train_images: int, num_of_val_easy_images: int) -> None:
        # Saving all directory names
        self.dataset_directory_name = dataset_directory_name
        self.dataset_dir = root_dir + f'/datasets/{self.dataset_directory_name}'
        self.train = self.dataset_dir + '/train'
        self.val = self.dataset_dir + '/val'
        self.val_hard = self.dataset_dir + '/val_hard'
        self.val_easy = self.dataset_dir + '/val_easy'
        self.original = self.dataset_dir + '/original'
        self.val_easy_original = self.dataset_dir + '/val_easy_original'

        # Saving number of training and val_easy images
        self.num_of_train_images = num_of_train_images
        self.num_of_val_easy_images = num_of_val_easy_images

        # Saving export.json file
        self.export_json = os.path.join(self.dataset_dir, 'export.json')

    # Making new directories
    def make_new_dirs(self, root_dir: str) -> None:
        '''
        Making new directories for the COCOpen dataset
        '''
        try: os.mkdir(self.dataset_dir)
        except: print(f'{self.dataset_dir} already exists!') 
        try: os.mkdir(self.train)
        except: print(f'{self.train} directory already exists!')
        try: os.mkdir(self.val)
        except: print(f'{self.val} directory already exists!')
        try: os.mkdir(self.val_hard)
        except: print(f'{self.val_hard} directory already exists!')
        try: os.mkdir(self.val_easy)
        except: print(f'{self.val_easy} directory already exists!')
        try: os.mkdir(self.original)
        except: print(f'{self.original} directory already exists!')
        try: os.mkdir(self.val_easy_original)
        except: print(f'{self.val_easy_original} directory already exists!')
        print("Created Directories")

        try: shutil.copy(root_dir + '/export.json', self.export_json)
        except: print(f'export.json already exists in {self.dataset_dir}')

    # Sorting data into 'train' and 'val' categories
    def is_reviewed(self, json_data: dict) -> dict:
        '''
        This function reads in Labelbox data from a Labelbox export.json file. 
        It sorts the data into 'train' and 'val' folders based on image name.
        Data which have not been reviewed are not sorted (rejected).
        '''
        json_dict = {"original": [],
                    "val_easy_original": [],
                    "val": [],
                    "val_hard": []}
        prefixes = ['a','b','c','d']
        hard = ['a2.png','a6.png','a9.png','a10.png','a11.png','a12.png','b1.png','b2.png','b3.png','b4.png','b5.png','c1.png','c2.png','c3.png','c4.png','c5.png','c6.png', 'a31.png', 'a30.png', 'a28.png', 'a27.png', 'a13.png', 'a11.png', 'a8.png', 'a5.png', 'a4.png', 'a3.png', 'a1.png']
        for d in json_data:
            reviewed = d["Reviews"]
            if reviewed:
                if d["External ID"][0] in prefixes:
                    if d["External ID"] not in hard:        
                        json_dict["val"].append(d)
                    else:
                        json_dict["val_hard"].append(d)
                else:
                    try: t = d['Label']['objects'][0]['title']
                    except: continue
                    try: l = len(d['Label']['objects'])
                    except: continue
                    if t == 'device' and l == 1: 
                        rand = random.random()
                        if 0 <= rand < 0.8:
                            json_dict["original"].append(d)
                        else:
                            json_dict["val_easy_original"].append(d)

        return json_dict

    # Returning category data given label
    def get_label(self, label, class_dict):
        '''
        This function returns category data given a label, reads the image, and performs intensity thresholding.
        '''
        label_url = label["instanceURI"]
        try:
            urllib.request.urlretrieve(label_url, "label")
        except:
            return
        try:
            category_id = class_dict[label["classifications"][0]["answer"][0]["title"]]
        except:
            category_id = class_dict[label["title"]]
        label_image = cv2.imread("label", cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY) # Threshold in grayscale

        return label_image, gray, category_id

    # Generating segmentation annotations in object semantics format
    def object_semantics(self, coco, ann_id, img_id, file_name=None, label=None, class_dict=None, mask=None, category_id=None):
        '''
        This function generates segmentation annotations with the object semantics format.
        '''
        if label is not None:
            # If generating semantics from main loop for "originals" folder
            # print(label, class_dict)        
            label_image, gray, category_id = self.get_label(label,class_dict)
            mask = cv2.findNonZero(gray)
            b, g, r, a = cv2.split(label_image)
            bitmask = np.asarray(a, order="F")
        else:
            # If generating semantics within combine_images()
            mask = np.asfortranarray(mask)
            bitmask = mask.astype(np.uint8)
            file_name = str(img_id) + '.png'

        segmentation = pycocomask.encode(bitmask)
        segmentation['counts'] = segmentation['counts'].decode('ascii')
        area = int(pycocomask.area(segmentation))
        bbox = cv2.boundingRect(mask)

        annotation = {"category_id": category_id,
                    "segmentation": segmentation,
                    "image_id": img_id,
                    "id": ann_id,
                    "iscrowd": 0,
                    "area": area,
                    "bbox": bbox,
                    "image_name": file_name}
        coco["annotations"].append(annotation)
        ann_id += 1

        return coco, ann_id

    # Generating segmentation annotations in object segment semantics format
    def object_segment_semantics(self, coco, ann_id, img_id, file_name=None, label=None, class_dict=None, final_img=None, category_id=None):
        '''
        This function generates segmentation annotations with the object segment semantics format.
        '''
        if label is not None:
            # If generating semantics from main loop for "originals" folder
            _, gray, category_id = self.get_label(label, class_dict)
            contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            new_frame = np.zeros(gray.shape, np.uint8)
        else:
            # If generating semantics within combine_images()
            gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            new_frame = np.zeros(gray.shape, np.uint8)
            file_name = str(img_id) + '.png'

        # for each object segment
        for a, contour in enumerate(contours):
            c_area = cv2.contourArea(contour)
            if 1000 <= c_area:
                mask = np.zeros(gray.shape, np.uint8)
                mask = cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
                new_mask = cv2.bitwise_and(gray, mask)
                pt_frame = cv2.bitwise_or(new_frame, new_mask)
                points = cv2.findNonZero(pt_frame)

                bitmask = np.asarray(mask, order="F").astype(np.uint8)
                segmentation = pycocomask.encode(bitmask)
                area = int(pycocomask.area(segmentation))
                segmentation['counts'] = segmentation['counts'].decode('ascii')
                bbox = cv2.boundingRect(points)
                # file_name = str(img_id) + '.png'

                annotation = {"category_id": category_id,
                            "segmentation": segmentation,
                            "image_id": img_id,
                            "id": ann_id,
                            "iscrowd": 0,
                            "area": area,
                            "bbox": bbox,
                            "image_name": file_name}
                coco["annotations"].append(annotation)

                ann_id+=1
        return coco, ann_id

    # Creating the hand labeled validation dataset
    def create_val_dataset(self) -> None:
        '''
        Creating the hand labeled validation dataset
        '''
        # Creating directories dictionary mapping name of directory to path of directory
        dirs = {self.original: f'{self.dataset_dir}/original',
            self.val_easy_original: f'{self.dataset_dir}/val_easy_original',
            self.val: f'{self.dataset_dir}/val',
            self.val_hard: f'{self.dataset_dir}/val_hard'}

        # Loading export.json file data
        export_json_data = json.load(open(self.export_json))
        export_json_dict = self.is_reviewed(json_data=export_json_data)

        # Keys of directory names
        keys = ['original', 'val_easy_original', 'val', 'val_hard']
        
        # Categories of instances in our images
        self.class_dict = {"device": 1,
                        "ethernet": 2,
                        "phone": 3,
                        "power": 4,
                        "hdmi": 5,
                        "coaxial": 6}
        self.categories = [{"supercategory": "device", "id": self.class_dict["device"], "name": "device"},
                                {"supercategory": "cable", "id": self.class_dict["ethernet"], "name": "ethernet"},
                                {"supercategory": "cable", "id": self.class_dict["phone"], "name": "phone"},
                                {"supercategory": "cable", "id": self.class_dict["power"], "name": "power"},
                                {"supercategory": "cable", "id": self.class_dict["hdmi"], "name": "hdmi"},
                                {"supercategory": "cable", "id": self.class_dict["coaxial"], "name": "coaxial"}]

        # Generating COCO formatted dataset for hand labeled validation dataset
        for i, dir in enumerate(dirs):

            coco_obj_sem = {"images": [],
                    "annotations": [],
                    "categories": self.categories }
            coco_obj_seg_sem = {"images": [],
                    "annotations": [],
                    "categories": self.categories }

            ann_id_sem = 0
            ann_id_seg_sem = 0
            img_id = 0
            
            new_json_dict = []
            
            # try to read device only images
            if dir == self.original:
                new_json_dict = export_json_dict['original']
            elif dir == self.val:
                new_json_dict = export_json_dict['val']
            elif dir == self.val_hard:
                new_json_dict = export_json_dict['val_hard']
            else:
                new_json_dict = export_json_dict['val_easy_original']
            
            print(f'Generating \'{keys[i]}\' data')

            for idx, data in enumerate(tqdm(iterable = new_json_dict, total = len(new_json_dict))):
                
                # Image data
                id = idx
                file_name = data["External ID"]
                file_path = os.path.join(dirs[dir], file_name)
                img_url = data["Labeled Data"]
                try:
                    urllib.request.urlretrieve(img_url, "image")
                except:
                    continue
                img = Image.open("image")
                
                # save original image
                img.save(os.path.join(dir,file_name))
                width = img.size[0]
                height = img.size[1]

                image = {"id": id,
                        "width": width,
                        "height": height,
                        "file_name": file_name}
                coco_obj_sem["images"].append(image)
                coco_obj_seg_sem["images"].append(image)

                # Segmentation Annotation Data
                objects = data["Label"].get("objects")
                if objects:
                    labels = data["Label"]["objects"]
                    for label in labels:
                        coco_obj_sem,ann_id_sem = self.object_semantics(coco=coco_obj_sem,ann_id=ann_id_sem,img_id=img_id,file_name=file_name,label=label,class_dict=self.class_dict,mask=None,category_id=None)
                        coco_obj_seg_sem,ann_id_seg_sem = self.object_segment_semantics(coco=coco_obj_seg_sem,ann_id=ann_id_seg_sem,img_id=img_id,file_name=file_name,label=label,class_dict=self.class_dict,final_img=None,category_id=None)
                img_id += 1
            
            with open(os.path.join(self.dataset_dir+'/'+keys[i],f'{keys[i]}_obj_sem.json'), 'w') as outfile:
                json.dump(coco_obj_sem, outfile, sort_keys=True, indent=4)

            with open(os.path.join(self.dataset_dir+'/'+keys[i],f'{keys[i]}_obj_seg_sem.json'), 'w') as outfile:
                json.dump(coco_obj_seg_sem, outfile, sort_keys=True, indent=4)

        print('finished generating annotation data')

    # Initializing Azure connection for for accessing blob storage
    def init_azure(self, foreground_image_container: str, background_image_container: str) -> None:
        '''
        Initializing Azure connection for for accessing blob storage
        '''
        # Initializing connection with Azure storage account
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        blob_service_client = BlobServiceClient.from_connection_string(conn_str=connection_string)
        # Initializing connection with foreground image container
        self.single_wire_container_client = blob_service_client.get_container_client(foreground_image_container)
        # Initializing connection with background image storage container
        self.background_container_client = blob_service_client.get_container_client(background_image_container)

    # Creating foreground image list
    def create_foreground_image_list(self) -> None:
        '''
        Creating foreground image list from images on Azure and segregating them into 'train' and 'val_easy' categories
        '''
        # Creating list of all foreground images
        azure_all_wire_list = self.single_wire_container_client.list_blobs()

        # Segregating images into 'train' and 'val_easy' categories
        self.train_wire_lst = []
        self.val_easy_wire_lst = []
        for blob in azure_all_wire_list:
            rand = random.random()
            if 0 <= rand < 0.8:
                self.train_wire_lst.append(blob.name)
            else:
                self.val_easy_wire_lst.append(blob.name)

    # Creating background image list
    def create_background_image_list(self) -> None:
        '''
        Creating background image list from images on Azure
        '''
        # Creating list of all background images
        azure_all_background_list = self.background_container_client.list_blobs()

        # Segregating images into 'train background' and 'val_easy background' categories
        self.train_backgrounds_lst = []
        self.val_easy_backgrounds_lst = []
        for blob in azure_all_background_list:
            rand = random.random()
            if 0 <= rand < 0.8:
                self.train_backgrounds_lst.append(blob.name)
            else:
                self.val_easy_backgrounds_lst.append(blob.name)

    # Downloading foreground image from Azure
    def download_single_wire_image_from_azure(self, img):
        '''
        Downloads single_wire images from Azure blob storage
        '''
        # Get blob and read image data into memory cache
        blob = self.single_wire_container_client.get_blob_client(img)
        img_data = blob.download_blob().readall()
        # Convert image data to numpy array
        img_array = np.asarray(bytearray(img_data), dtype=np.uint8)
        # Decode image
        src = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return src

    # Downloading background image from Azure
    def download_background_image_from_azure(self, bg):
        '''
        Downloads background images from Azure blob storage
        '''
        # Get blob and read image data into memory cache
        blob = self.background_container_client.get_blob_client(bg)
        img_data = blob.download_blob().readall()
        # Convert image data to numpy array
        img_array = np.asarray(bytearray(img_data), dtype=np.uint8)
        # Decode image
        src = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return src

    # Getting contour filters
    def contour_filter(self, frame, contour_min_area=1000, contour_max_area=2075000):
        contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new_frame = np.zeros(frame.shape, np.uint8)
        for i, contour in enumerate(contours):
            c_area = cv2.contourArea(contour)
            if contour_min_area <= c_area <= contour_max_area:
                mask = np.zeros(frame.shape, np.uint8)
                cv2.drawContours(mask, contours, i, 255, cv2.FILLED)
                mask = cv2.bitwise_and(frame, mask)
                new_frame = cv2.bitwise_or(new_frame, mask)
        frame = new_frame

        return frame

    # Returning wire information
    def get_wire_info(self, img):
        '''
        This funciton returns the image array and the mask array
        '''
        # Get wire mask
        src = self.download_single_wire_image_from_azure(img=img)
        median = cv2.medianBlur(src, 9) # Blur image
        gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY) # Threshold step
        new_mask = self.contour_filter(mask) # blob filtering step

        # # code below was unused
        # b, g, r = cv2.split(src)
        # bgra = [b, g, r, new_mask]
        # wire_png = cv2.merge(bgra, 4) # Create image with transparency from threshold

        # wire_png = cv2.cvtColor(wire_png, cv2.COLOR_BGRA2RGBA) # Convert to RGBA (RGB + transparency)
        # wire_pil = Image.fromarray(wire_png)
        # left, upper, right, lower = wire_pil.getbbox() # Get bounding box of wire 

        # Encode
        mask_array = np.asarray(new_mask, order="F")
        
        return src, mask_array

    # Scaling image
    def scale_image(self, img_arr, masks, scale):
        '''
        This function scales the input image and outputs the updated image array and mask array
        '''
        # resize image and mask
        new_img_arr = cv2.resize(img_arr, (int(1920*scale), int(1080*scale)))
        new_masks = []
        for mask in masks:
            new_masks.append(cv2.resize(mask, (int(1920*scale), int(1080*scale))))
        
        # depend on if the image is smaller or larger than 1920x1080, perform different actions
        # images directly gotten from labelbox has 4 channels
        if len(img_arr[0][0]) == 4:
            final_img_arr = np.zeros((1080,1920,4)).astype('uint8')
        else:
            final_img_arr = np.zeros((1080,1920,3)).astype('uint8')
            
        final_masks = []
        for i in range (0, len(new_masks)):
            final_masks.append(np.zeros((1080,1920)).astype('uint8'))
        
        # if the scaled image is smaller than original
        if int(1920*scale) < 1920 or int(1080*scale) < 1080:
            # fisr detemrine the range the upper left corner can take
            max_h_offset = 1920 - int(1920*scale)
            max_v_offset = 1080 - int(1080*scale)
            upper_left_x = int(random.random()*max_h_offset)
            upper_left_y = int(random.random()*max_v_offset)
                    
            # concat arrays
            for i in range(0, len(final_masks)):
                for j in range(0, int(1080*scale)):
                    final_img_arr[j+upper_left_y, upper_left_x:(upper_left_x+int(1920*scale))] = new_img_arr[j]
                    final_masks[i][j+upper_left_y][upper_left_x:(upper_left_x+int(1920*scale))] = new_masks[i][j]
        
        # if the scaled image is larget than original
        else:
            # fisr detemrine the range the upper left corner can take
            max_h_offset = len(new_img_arr[0]) - 1920
            max_v_offset = len(new_img_arr) - 1080
            upper_left_x = int(random.random()*max_h_offset)
            upper_left_y = int(random.random()*max_v_offset)
            
            # concat arrays
            final_img_arr = new_img_arr[upper_left_y:(upper_left_y+1080), upper_left_x:(upper_left_x+1920)]
            for i in range(0, len(final_masks)):
                final_masks[i] = new_masks[i][upper_left_y:(upper_left_y+1080), upper_left_x:(upper_left_x+1920)]
        
        return final_img_arr, final_masks 

    # Color augmentation
    def color_augmentation(self, img_arr, enhancer_range, change_brightness, change_contrast, change_saturation, change_hue):
        '''
        This function takes np.ndarray as input, performs color augmentation using PIL, then output the new np.ndarray
        '''
        new_img_arr = Image.fromarray(img_arr)
        
        # this step is extremely slow
        if change_hue == True:
            rgb_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            hsv_img = Image.fromarray(rgb_img)
            hsv_img = hsv_img.convert('HSV')
            # convert back to np array for operations
            hsv_img = np.array(hsv_img)
            
            # randomly change hue
            rand_value = int(random.random()*255*2) - 255
            
            hsv_img_int = hsv_img.astype('int32')
            hsv_img_int[:, :, 0] += rand_value + 255
            hsv_img_int[:, :, 0] = hsv_img_int[:, :, 0] % 255
            hsv_img = hsv_img_int.astype('uint8')
            
            # convert back to PIL image
            new_img_arr = Image.new('HSV', (1920, 1080), (0, 0, 0))
            new_img_arr = Image.fromarray(hsv_img, mode='HSV')
            new_img_arr = new_img_arr.convert('RGB')
        
        if change_brightness == True:
            modifier = ImageEnhance.Brightness(new_img_arr)
            new_img_arr = modifier.enhance(enhancer_range[0] + random.random()*(enhancer_range[1]-enhancer_range[0]))
        if change_contrast == True:
            modifier = ImageEnhance.Contrast(new_img_arr)
            new_img_arr = modifier.enhance(enhancer_range[0] + random.random()*(enhancer_range[1]-enhancer_range[0]))
        if change_saturation == True:
            modifier = ImageEnhance.Color(new_img_arr)
            new_img_arr = modifier.enhance(enhancer_range[0] + random.random()*(enhancer_range[1]-enhancer_range[0]))
        
        return np.array(new_img_arr)

    # Random operations on image data
    def random_operations(self, img, angle, flip_h, flip_v):
        '''
        Rotates and flips the input image
        '''
        # rotation
        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1920/(1080*1920/np.sqrt(1920**2 + 1080**2)))
        img_new = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        # flip
        if flip_h:
            # horizontal flip
            img_new = np.flip(img_new, 1)
        if flip_v:
            # vertical flip
            img_new = np.flip(img_new, 0)
        # if fx_scale and fy_scale:
        #     img_new = cv2.resize(img_new, dsize = (1920,1080), fx=fx_scale, fy=fy_scale)
        return img_new

    # Combining foreground and background images
    def combine(self, dataset_dir, orig_device_dir, orig_wire_img_lst, background_img_lst, target_dir, num_of_images, categories, class_dict, single_wire_container_client, background_container_client):
        '''
        Combining foreground and background images
        '''
        # different data augmentation methods
        # scale jittering
        standard_scale_jittering = False   # different objects use the same scale factor
        large_scale_jittering = False
        individual_standard_scale_jittering = False   # different objects use different scale factor
        individual_large_scale_jittering = True
        
        # color augmentation
        individual_color_augmentation = True   # if set to true, different objects use different values
        change_saturation = True
        change_brightness = True
        change_contrast = True
        change_hue = True
        
        # this sets the lower/upper limit of color augmentation
        enhancer_range = [0.5, 1.5]
        scale_range_standard = [0.8, 1.25]
        scale_range_large = [0.3, 3.0]
        
        wire_lst = orig_wire_img_lst.copy()
        
        device_lst = os.listdir(orig_device_dir)
        device_img_lst = []
        # this list contains a json file
        for k in range (0, len(device_lst)):
            _, extension = os.path.splitext(device_lst[k])
            if extension == '.png':
                device_img_lst.append(device_lst[k])
        
        # modified coco_new category id 
        coco_new_obj_sem = {"images": [],
                    "annotations": [],
                    "categories": categories}

        coco_new_obj_seg_sem = {"images": [],
                    "annotations": [],
                    "categories": categories}

        # this needs to be tracked at all time - annotation id must be unique across all instances in the entire dataset
        ann_id_sem = 0
        ann_id_seg_sem = 0
        
        for i in tqdm(range(num_of_images)):
            
            scale = 1   # initialization
            
            if standard_scale_jittering == True:
                # standard scale jittering, all objects use the same scale factor
                scale = scale_range_standard[0] + random.random()*(scale_range_standard[1] - scale_range_standard[0])
            
            if large_scale_jittering == True:
                # large scale jittering, all objects use the same scale factor
                scale = scale_range_large[0] + random.random()*(scale_range_large[1] - scale_range_large[0])
            
            # has to load coco data again for some weird reason
            coco_data = []
            if os.path.split(orig_device_dir)[1] == 'original':
                coco_data = json.load(open(orig_device_dir + '/original_obj_sem.json'))
            if os.path.split(orig_device_dir)[1] == 'val_easy_original':
                coco_data = json.load(open(orig_device_dir + '/val_easy_original_obj_sem.json'))
            
            # image lst needs to be restored every loop
            device_lst = os.listdir(orig_device_dir)
            device_img_lst = []
            # this list contains a json file
            for k in range (0, len(device_lst)):
                _, extension = os.path.splitext(device_lst[k])
                if extension == '.png':
                    device_img_lst.append(device_lst[k])
                    
            wire_lst = orig_wire_img_lst.copy()

            image = {"id": i, # this can be updated later in a for loop
                    "width": 1920,
                    "height": 1080,
                    "file_name": str(i) + '.png'}
            coco_new_obj_sem["images"].append(image)
            coco_new_obj_seg_sem["images"].append(image)
            
            # determine how many devices and how many cables in each image
            num_of_devices = int(random.random() * 2) + 1 # 1-2 devices
            num_of_wires = int(random.random() * 4) + 1 # 1-4 wires

            # arrays storing both wire and device info
            all_img_arr = []
            binary_mask_arr = []
            all_category_ids = []
            
            # get wire mask info
            for j in range (0, num_of_wires):
                index = int(len(wire_lst) * random.random())
                # print(index, len(wire_lst))
                wire = wire_lst[index]
                wire_img, wire_mask = self.get_wire_info(wire)
                
                # # check to see if out of frame
                # contour_count = count_contours(wire_img)
                # while contour_count > 1:
                    # index = int(len(wire_lst) * random.random())
                    # wire = wire_lst[index]
                    # wire_img, wire_mask = get_wire_info(wire_dir, wire)
                    # contour_count = count_contours(wire_img)
                
                wire_lst.pop(index)
                
                wire_mask = wire_mask/255
                wire_mask = wire_mask.astype('uint8')
                
                wire_msk = []
                wire_msk.append(wire_mask)
                
                # scaling
                if standard_scale_jittering == True or large_scale_jittering == True:
                    wire_img, wire_msk = self.scale_image(wire_img, wire_msk, scale)
                elif individual_standard_scale_jittering == True:
                    scale = scale_range_standard[0] + random.random()*(scale_range_standard[1] - scale_range_standard[0])
                    wire_img, wire_msk = self.scale_image(wire_img, wire_msk, scale)
                else:
                    scale = scale_range_large[0] + random.random()*(scale_range_large[1] - scale_range_large[0])
                    wire_img, wire_msk = self.scale_image(wire_img, wire_msk, scale)
                
                # color augmentation
                if individual_color_augmentation == True:
                    # only change hue for wires
                    wire_img = self.color_augmentation(wire_img, enhancer_range, change_brightness, change_contrast, change_saturation, change_hue)
                
                all_img_arr.append(wire_img)
                binary_mask_arr.append(wire_msk)
                all_category_ids.append(2)
        
            # get device info
            for j in range (0, num_of_devices):
                
                masks = []
                index = int(len(device_img_lst) * random.random())
                img = device_img_lst[index]
                device_img = cv2.imread(os.path.join(orig_device_dir, img), cv2.IMREAD_UNCHANGED)
                device_img_lst.pop(index)
                
                for annotations in coco_data['annotations'][:]:          
                    if annotations['image_name'] == img: 
                        masks.append(pycocomask.decode(annotations['segmentation']))
                
                # removed scaling for device 5/18/2022
                # # scaling
                # if standard_scale_jittering == True or large_scale_jittering == True:
                #     device_img, masks = scale_image(device_img, masks, scale)
                # elif individual_standard_scale_jittering == True:
                #     scale = scale_range_standard[0] + random.random()*(scale_range_standard[1] - scale_range_standard[0])
                #     device_img, masks = scale_image(device_img, masks, scale)
                # else:
                #     scale = scale_range_large[0] + random.random()*(scale_range_large[1] - scale_range_large[0])
                #     device_img, masks = scale_image(device_img, masks, scale)
                
                # color augmentation
                if individual_color_augmentation == True:
                    # don't change hue
                    device_img = self.color_augmentation(device_img, enhancer_range, change_brightness, change_contrast, change_saturation, False)
                
                all_img_arr.append(device_img)
                binary_mask_arr.append(masks)
                all_category_ids.append(class_dict["device"])

            # perform random flip
            for m in range (0, num_of_devices + num_of_wires):

                horizontal = int(2*random.random())
                vertical = int(2*random.random())
                
                if horizontal == 1:
                    all_img_arr[m] = cv2.flip(all_img_arr[m], 1)
                    for n in range (0, len(binary_mask_arr[m])):
                        binary_mask_arr[m][n] = np.flip(binary_mask_arr[m][n], 1)
                    
                if vertical == 1:
                    all_img_arr[m] = cv2.flip(all_img_arr[m], 0)
                    for n in range (0, len(binary_mask_arr[m])):
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
                
                msk1 = np.zeros((1080,1920,3)).astype('uint8')
                msk1[:,:,0] = mask1
                msk1[:,:,1] = mask1
                msk1[:,:,2] = mask1
                final_img = cv2.bitwise_or(msk1.astype('uint8'), all_img_arr[randint], mask = mask1)

                coco_new_obj_seg_sem, ann_id_seg_sem = self.object_segment_semantics(coco=coco_new_obj_seg_sem,ann_id=ann_id_seg_sem,img_id=img_id,file_name=None,label=None,class_dict=None,final_img=final_img,category_id=category_id)
                coco_new_obj_sem, ann_id_sem = self.object_semantics(coco=coco_new_obj_sem,ann_id=ann_id_sem,img_id=img_id,file_name=None,label=None,class_dict=None,mask=mask1,category_id=category_id)
                # print("Mask1 annid, line 321:",coco_new_o)
            # save mask1 info for later use
            mask_array_1 = np.dstack(binary_mask_arr[randint])
            mask_array_1 = np.max(mask_array_1, axis=2).astype('uint8')
            
            # prevent repetition
            all_img_arr.pop(randint)
            binary_mask_arr.pop(randint)
            all_category_ids.pop(randint)

            # combine the rest
            for j in range (1, num_of_devices + num_of_wires):
                
                # choose the second image
                randint2 = int(random.random()*len(all_img_arr))
                msk2 = np.zeros((1080,1920,3)).astype('uint8')   
                mask_array_2 = []                                
                
                category_id = all_category_ids[randint2]

                # update coco info
                for mask2 in binary_mask_arr[randint2]:
                    
                    # mask2 = binary_mask_arr[randint2][k]s
                    mask2 = mask2 & (~ mask_array_1)
                    mask_array_2.append(mask2)

                    msk2 = np.zeros((1080,1920,3)).astype('uint8')
                    msk2[:,:,0] = mask2
                    msk2[:,:,1] = mask2
                    msk2[:,:,2] = mask2
                    masked_layer = cv2.bitwise_or(msk2.astype('uint8'), all_img_arr[randint2], mask = mask2)
                    final_img = cv2.add(final_img, masked_layer)
                    
                    coco_new_obj_seg_sem, ann_id_seg_sem = self.object_segment_semantics(coco_new_obj_seg_sem,ann_id_seg_sem,img_id,final_img=masked_layer,category_id=category_id) 
                    coco_new_obj_sem, ann_id_sem = self.object_semantics(coco_new_obj_sem,ann_id_sem,img_id,mask=mask2,category_id=category_id)
                    # print("Mask2 annid, line 358:",ann_id_sem)

                mask_array_2 = np.dstack(mask_array_2)
                mask_array_2 = np.max(mask_array_2, axis=2).astype('uint8')
                
                # update mask_array_1 to be the combination of img1 and img2
                mask_array_1 = mask_array_1 | mask_array_2
                
                # prevent repetition
                all_img_arr.pop(randint2)
                binary_mask_arr.pop(randint2)
                all_category_ids.pop(randint2)
                
                # enter the next loop
            
            # add a random background
            randbg = int(random.random()*len(background_img_lst))
            bg = background_img_lst[randbg]
            # Download background image from azure
            src = self.download_background_image_from_azure(bg=bg)
            bg_arr = src.astype('uint8')

            # background random operations
            bg_rot = random.random() * 360
            bg_flip_h = bool(int(random.random()/0.5))
            bg_flip_v = bool(int(random.random()/0.5))
            bg_arr = self.random_operations(bg_arr, bg_rot, bg_flip_h, bg_flip_v)

            mask_bg = np.ones((1080,1920)).astype('uint8')
            msk_bg = np.zeros((1080,1920,3)).astype('uint8')
            mask_bg = mask_bg & (~ mask_array_1) # & (~ mask_array_2)

            msk_bg[:,:,0] = mask_bg
            msk_bg[:,:,1] = mask_bg
            msk_bg[:,:,2] = mask_bg

            masked_bg = cv2.bitwise_or(msk_bg.astype('uint8'), bg_arr, mask = mask_bg)

            final_img = cv2.add(final_img, masked_bg)
            
            # color augmentation
            final_img = self.color_augmentation(final_img, enhancer_range, change_brightness, change_contrast, change_saturation, False)
            
            cv2.imwrite(os.path.join(target_dir, str(i)+'.png'), final_img)
            
        return coco_new_obj_sem, coco_new_obj_seg_sem

    # Generating training dataset
    def generate_train_data(self) -> None:
        '''
        Generating training dataset
        '''
        # generate train
        print("Generating 'train' data")
        coco_new_obj_sem, coco_new_obj_seg_sem = self.combine(dataset_dir = self.dataset_dir, orig_device_dir = self.original, orig_wire_img_lst = self.train_wire_lst, background_img_lst = self.train_backgrounds_lst, target_dir = self.train, num_of_images = self.num_of_train_images, categories = self.categories, class_dict=self.class_dict, single_wire_container_client=self.single_wire_container_client, background_container_client=self.background_container_client)
        f1 = open(self.train + '/train_obj_sem.json', 'w')
        json.dump(coco_new_obj_sem, f1, sort_keys=True, indent=4)
        f1.close()
        f2 = open(self.train + '/train_obj_seg_sem.json', 'w' )
        json.dump(coco_new_obj_seg_sem, f2, sort_keys=True, indent=4)
        f2.close()

    # Generating val_easy dataset
    def generate_val_easy_data(self) -> None:
        '''
        Generating val_easy dataset
        '''
        print("Generating 'val_easy' data")
        coco_new_obj_sem, coco_new_obj_seg_sem = self.combine(dataset_dir = self.dataset_dir, orig_device_dir = self.val_easy_original, orig_wire_img_lst = self.val_easy_wire_lst, background_img_lst = self.val_easy_backgrounds_lst, target_dir = self.val_easy, num_of_images = self.num_of_val_easy_images, categories = self.categories, class_dict=self.class_dict, single_wire_container_client=self.single_wire_container_client, background_container_client=self.background_container_client)
        f1 = open(self.val_easy + '/val_easy_obj_sem.json', 'w')
        json.dump(coco_new_obj_sem, f1, sort_keys=True, indent=4)
        f1.close()
        f2 = open(self.val_easy + '/val_easy_obj_seg_sem.json', 'w' )
        json.dump(coco_new_obj_seg_sem, f2, sort_keys=True, indent=4)
        f2.close()

    # Delete certain directories
    def delete_dirs(self, root_dir: str) -> None:
        shutil.rmtree(self.dataset_dir + '/original')
        shutil.rmtree(self.dataset_dir + '/val_easy_original')
        os.remove(root_dir + '/image')
        os.remove(root_dir + '/label')

    # Creating zip file of the dataset
    def zip(self, base_name: str, root_dir: str, base_dir: str, format="zip") -> None:
        '''
        Function which zips all files in a directory
        '''
        shutil.make_archive(format=format, base_name=base_name, root_dir=root_dir, base_dir=base_dir)
        print(self.dataset_directory_name + '.zip file successfully created!')
    
