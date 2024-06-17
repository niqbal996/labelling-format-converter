import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
with open('/mnt/e/datasets/phenobench/coco_annotations/coco_plants_panoptic_val.json', 'r') as f:
    data = json.load(f)
data1 = {}
data1['categories'] = data['categories']
tmp = {}
tmp['annotations'] = []
tmp['images'] = []
for img in tqdm(data['images']):
    if '06-05' in img['file_name']:
        img_id = img['id']
        tmp['images'].append(img)
        for ann, i in zip(data['annotations'], range(len(data['annotations']))):
            if ann['image_id'] == img['id']:
                tmp['annotations'].append(ann)
data1['annotations'] = tmp['annotations']
data1['images'] = tmp['images']
print(len(tmp['images']))
with open('/mnt/e/datasets/phenobench/coco_annotations/coco_plants_panoptic_val-day3.json', 'w+') as f:
    json.dump(data1, f)

print('hold')