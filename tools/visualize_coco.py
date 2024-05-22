import cv2
import matplotlib.pyplot as plt
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from tqdm import tqdm
import pickle 
import numpy as np


# Register the COCO dataset
# register_coco_instances("my_dataset_train", {}, "path/to/train/annotations.json", "/mnt/e/datasets/sugarbeet_syn_v1/sugarbeet_syn_v4")
register_coco_instances("SugarbeetSyn23", {}, 
                        "/mnt/e/datasets/sugarbeet_syn_v1/sugarbeet_syn_v4/coco_annotations/instances_train.json", 
                        "/mnt/e/datasets/sugarbeet_syn_v1/sugarbeet_syn_v4/images_lis_directional")

register_coco_instances("Phenobench23", {}, 
                        "/mnt/e/datasets/phenobench/coco_annotations/coco_plants_panoptic_train.json", 
                        "/mnt/e/datasets/phenobench/train/")

# Create a predictor using the trained model

# Get the dataset dictionary
# my_dataset_metadata = MetadataCatalog.get("Phenobench23")
# dataset_dicts = DatasetCatalog.get("Phenobench23")
# cv2.namedWindow('Images')
# real_boxes = {}
# real_boxes['sugarbeet'] =[]
# real_boxes['weeds'] =[]
# for d in tqdm(dataset_dicts):
#     # img = cv2.imread(d["file_name"])
#     # visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_metadata, scale=1.0)
#     for img in dataset_dicts:
#         for ann in img['annotations']:
#             if ann['category_id'] == 1:
#                 real_boxes['sugarbeet'].append(ann['bbox'][2] * ann['bbox'][3])
#             else:
#                 real_boxes['weeds'].append(ann['bbox'][2] * ann['bbox'][3])
    # vis = visualizer.draw_dataset_dict(d)
    # cv2.imshow('Images', vis.get_image()[:, :, ::-1])
    # cv2.waitKey()
# cv2.destroyAllWindows()

# with open('real_boxes.pkl', 'wb') as f:
#     pickle.dump(real_boxes, f)

with open('real_boxes.pkl', 'rb') as f:
    real_boxes = pickle.load(f)
# 
# Plot histogram for real_boxes
plt.figure(figsize=(10, 5))
bins = np.arange(0,250000,10000)
plt.hist(np.clip(real_boxes['weeds'], bins[0], bins[4]), bins=10, alpha=0.5, label='weeds')
# plt.hist(real_boxes['sugarbeet'], bins=bins, alpha=0.5, label='Sugarbeet')
# plt.hist(real_boxes['weeds'], bins=50, alpha=0.5, label='Weeds')
plt.xlabel('Box Area')
plt.ylabel('Frequency')
plt.title('Real Boxes')
plt.legend()
plt.show() 

# dataset_dicts = DatasetCatalog.get("SugarbeetSyn23")
# syn_boxes = {}
# syn_boxes['sugarbeet'] =[]
# syn_boxes['weeds'] =[]
# for d in tqdm(dataset_dicts):
#     # img = cv2.imread(d["file_name"])
#     # visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_metadata, scale=1.0)
#     for img in dataset_dicts:
#         for ann in img['annotations']:
#             if ann['category_id'] == 1:
#                 syn_boxes['sugarbeet'].append(ann['bbox'][2] * ann['bbox'][3])
#             else:
#                 syn_boxes['weeds'].append(ann['bbox'][2] * ann['bbox'][3])
# with open('syn_boxes.pkl', 'wb') as f:
#     pickle.dump(syn_boxes, f)

with open('syn_boxes.pkl', 'rb') as f:
    syn_boxes = pickle.load(f)

# Plot histogram for syn_boxes
plt.figure(figsize=(10, 5))
# plt.hist(syn_boxes['sugarbeet'], bins=50, alpha=0.5, label='Sugarbeet')
bins = np.arange(0,250000,10000)
plt.hist(np.clip(syn_boxes['weeds'], bins[0], bins[6]), bins=10, alpha=0.5, label='weeds')
# plt.hist(syn_boxes['weeds'], bins=50, alpha=0.5, label='Weeds')
plt.xlabel('Box Area')
plt.ylabel('Frequency')
plt.title('Synthetic Boxes')
plt.legend()
plt.show()
print('hold')