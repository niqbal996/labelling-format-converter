from glob import glob
import cv2
import albumentations as A
import numpy as np
import os
import random
random.seed(2)

input_dir = '/home/niqbal/git/syclops/output/sugarbeet_syn_v2/main_camera/rect'
out_dir = os.path.join(input_dir, 'augmented')
os.makedirs(out_dir, exist_ok=True)
input_images = glob(f'{input_dir}/*.png')
#cv2.namedWindow('transformed image', cv2.WINDOW_NORMAL)
for image_path in input_images:
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        # A.AdvancedBlur(
        #             blur_limit=9,
        #             # sigma_x_limit=(2),
        #             # sigma_y_limit=(2),
        #             always_apply=True, 
        #             p=1.0),
        A.ImageCompression(quality_lower=60, quality_upper=70, p=1.0),
        # A.Defocus(radius=1, alias_blur=0.2, p=1.0),
        A.ChromaticAberration(p=1.0),   
        # A.ColorJitter(p=1.0), # not working
        A.CLAHE(p=0.5),
        A.MotionBlur(blur_limit=3, p=1.0),
    ])

    transformed = transform(image=image)
    transformed_image = transformed['image']
    combined_img = np.concatenate((image, transformed_image), axis=1)
    cv2.imwrite(os.path.join(out_dir, os.path.basename(image_path)), transformed_image)
    # cv2.imshow('transformed image', combined_img)
    # cv2.waitKey(0)
