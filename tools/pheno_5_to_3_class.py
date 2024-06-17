import json
import os
import numpy as np
from PIL import ImageTk, Image

from glob import glob
import matplotlib.pyplot as plt
base_dir = '/mnt/e/datasets/phenobench/'
splits = ['train', 'val']
os.makedirs(os.path.join(base_dir, 'train', 'semantics_3c'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'train', 'semantics_3c'), exist_ok=True)
for split in splits:
    os.makedirs(os.path.join(base_dir, split, 'semantics_3c'), exist_ok=True)
    anns = glob(os.path.join(base_dir, split, 'semantics/*.png'))
    for ann in anns:
        mask = Image.open(ann)
        if 3 or 4 in np.unique(mask):
            tmp = np.array(mask).copy()
            tmp[tmp == 3] = 1
            tmp[tmp == 4] = 2
            tmp = Image.fromarray(tmp)
            tmp.save(ann.replace('semantics', 'semantics_3c'))
        else:
            mask.save(ann.replace('semantics', 'semantics_3c'))
        # print(np.unique(mask))