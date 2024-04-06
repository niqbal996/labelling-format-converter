from glob import glob
import cv2
import albumentations as A
import numpy as np
import os
# /home/niqbal/git/syclops/output/2024_04_06_14_00_06/main_camera_annotations/semantic_segmentation
input_dir = '/home/niqbal/git/syclops/output/2024_04_06_14_00_06/main_camera_annotations/semantic_segmentation'
out_dir = os.path.join(input_dir, 'semantics')
os.makedirs(out_dir, exist_ok=True)
input_images = glob(input_dir + '/*.npz')
#cv2.namedWindow('transformed image', cv2.WINDOW_NORMAL)
for image_path in input_images:
    arr = np.load(image_path)['array'].astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, os.path.basename(image_path[:-3]+'png')), arr)
