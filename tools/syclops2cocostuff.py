from glob import glob
import cv2
import albumentations as A
import numpy as np
import os
color_map = {
    0: [0, 0, 0],
    1: [0, 255, 0], # green BGR
    2: [0, 0, 255], # red
}
# /home/niqbal/git/syclops/output/2024_04_06_14_00_06/main_camera_annotations/semantic_segmentation
input_dir = '/mnt/e/datasets/sugarbeet_syn_v1/sugarbeet_syn_v2/main_camera_annotations/semantic_segmentation'
# out_dir = os.path.join(input_dir, 'semantics')
# os.makedirs(out_dir, exist_ok=True)
input_images = glob(input_dir + '/*.npz')
#cv2.namedWindow('transformed image', cv2.WINDOW_NORMAL)
# for image_path in input_images:
#     arr = np.load(image_path)['array'].astype(np.uint8)
#     cv2.imwrite(os.path.join(out_dir, os.path.basename(image_path[:-3]+'png')), arr)


for image_path in input_images:
    arr = np.load(image_path)['array'].astype(np.uint8)
    color = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        color[arr == i] = color_map[i]
    cv2.imshow(os.path.basename(image_path), color)
    cv2.waitKey(0)


