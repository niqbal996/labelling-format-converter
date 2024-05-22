import json
import os
with open('/mnt/d/datasets/sugarbeet_syn_v1/coco_annotations/instances_2023_train_improved_boxes.json', 'r') as f:
    data = json.load(f)

# for idx in range(len(data['images'])):
#     filepath = data['images'][idx]['file_name']
#     basename = os.path.basename(filepath)
#     data['images'][idx]['file_name'] = basename


for idx, ann in zip(range(len(data['annotations'])), 
                    data['annotations']):
    data['annotations'][idx]['bbox'] = [int(item) for item in ann['bbox']]
    data['annotations'][idx]['segmentation'][0] = [int(item) for item in data['annotations'][idx]['segmentation'][0]]

with open('/mnt/d/datasets/sugarbeet_syn_v1/coco_annotations/instances_2023_train_jan15.json', 'w+') as f:
    json.dump(data, f, default=float, indent=4)