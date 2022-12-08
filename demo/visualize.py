import sys
import pycocotools.mask as pycocomask

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import os

import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

name = "cocopen-dataset-4"
path = f"./datasets/{name}"



class_map = {0: "device",
             1: "cable"}

dir = "train"
register_coco_instances("train", {}, f"{path}/{dir}/{dir}_obj_sem.json", f"{path}/{dir}/")
print(f"{path}/{dir}/{dir}_obj_sem.json")
print(f"{path}/{dir}/")
dicts = DatasetCatalog.get("train")
metadata = MetadataCatalog.get("train")
for d in dicts[:10]:
  img = cv2.imread(d["file_name"])
  annos = d["annotations"]
  _, name = os.path.split(d["file_name"])
  print("Image Name: ", "train")
  fig, axs = plt.subplots(1, len(annos), squeeze = False)
  if len(annos) > 1:
    fig.set_size_inches(30, 20, forward=True)
  else:
    fig.set_size_inches(8, 5, forward=True)
  for i, anno in enumerate(annos):
    encoded = anno["segmentation"]
    mask = pycocomask.decode(encoded)
    x,y,w,h = anno["bbox"]
    rect = Rectangle((x,y), w, h, linewidth=1, edgecolor='w', facecolor='None')
    cat = anno["category_id"]
    axs[0,i].imshow(mask, cmap=plt.get_cmap('Greys_r'))
    axs[0,i].add_patch(rect)  
    try:
      axs[0,i].title.set_text(class_map[cat])
    except:
      print("class_map and category are mismatched - class_map dictionary in this notebook may need modification")
    axs[0,i].get_xaxis().set_visible(False)
    axs[0,i].get_yaxis().set_visible(False)

  plt.show()
  visualizer = Visualizer(img, metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE)
  out = visualizer.draw_dataset_dict(d)
  cv2.imshow(out.get_image())
  cv2.imshow(img)