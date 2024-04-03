import json
import os
import numpy as np
import matplotlib.pyplot as plt
with open('/mnt/d/datasets/phenobench/coco_annotations/coco_plants_panoptic_val.json', 'r') as f:
    data = json.load(f)
areas = np.zeros(len(data['annotations']), dtype=np.uint32)
for ann, idx in zip(data['annotations'], range(len(data['annotations']))):
    areas[idx] = ann['area']

plt.hist(areas, bins=30)
plt.show()
print('hold')