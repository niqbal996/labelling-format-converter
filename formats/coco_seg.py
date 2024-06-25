
import sys
import json
import numpy as np
import cv2
import skimage 
from random import randint

from os.path import join, basename
from os import makedirs
from glob import glob
from tqdm import tqdm
from skimage import measure
import imutils 
from shapely.geometry import Polygon, MultiPolygon

class COCO_Instance_segmentation(object):
    def __init__(self, root_path='/media/naeem/T7/datasets/Corn_syn_dataset/',
                 anns_dir=None,
                 anns_file=None,
                 subset=False):
        self.root_path = root_path
        self.anns_dir = anns_dir
        self.anns_file = anns_file
        self.subset = subset
        self.boxes = None
        self.image_root = join(self.root_path, 'main_camera', 'rect')
        self.source_folder = join(self.root_path, 'main_camera_annotations', 'instance_segmentation')
        # NOTE need semantic segmentation to get the class label in a specified pixel area by the instance segmentation mask. 
        self.source_seg_folder = join(self.root_path, 'main_camera_annotations', 'semantic_segmentation')
        self.source_boxes = join(self.root_path, 'main_camera_annotations', 'bounding_box')
        self.new_Anns = {}
        self.new_Anns['info'] = {
                        'description': f'Synthetic Sugarbeets dataset v2.0',
                        'url': 'None',
                        'version': '1.0',
                        'year': '2024',
                        'contributor': 'LALWECO',
                        'date_created': '03 April,2024'}

        self.new_Anns['licenses'] = [{'url': 'None', 'id': 1, 'name': 'None'}]

        self.new_Anns['categories'] = [{'supercategory': 'plants', 'id': 1, 'name': 'sugarbeets'},
                                {'supercategory': 'plants', 'id': 2, 'name': 'weeds'},
                                ]

        self.new_Anns['images'] = []
        self.new_Anns['annotations'] = []    

    def mask2polygons(self, image_filename):
        file = join(self.source_folder, f'{basename(image_filename)[:-4]}.npz')
        seg_file = join(self.source_seg_folder, f'{basename(image_filename)[:-4]}.npz')
        
        instance_segmentation_mask = np.load(file)['array'].astype(np.uint16) + 1
        semantic_segmentation_mask = np.load(seg_file)['array']
        
        instance_ids = np.unique(instance_segmentation_mask)
        instance_ids = instance_ids[instance_ids != 0]

        all_segmentations, all_boxes, all_areas, all_ids = [], [], [], []
        kernel = np.ones((2,2), np.uint8)

        for id in instance_ids:
            instance_mask = (instance_segmentation_mask == id).astype(np.uint8)
            semantic_id = np.bincount(semantic_segmentation_mask[instance_mask == 1]).argmax()
            
            if semantic_id == 0:
                continue

            dilated_mask = cv2.dilate(instance_mask, kernel, iterations=1)
            contours = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            
            if area < 60:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.002 * peri, True)
            segmentation = approx.flatten().tolist()

            all_segmentations.append(segmentation)
            all_boxes.append([x, y, w, h])
            all_areas.append(area)
            all_ids.append(semantic_id)

        return all_segmentations, all_boxes, all_areas, all_ids

    def show_image(self, img, window_name):
        cv2.namedWindow("window_name", cv2.WINDOW_NORMAL)
        cv2.moveWindow("window_name", 40,30)
        cv2.resizeWindow("window_name", 1200, 1400)
        cv2.imshow('window_name', img)
        cv2.waitKey(0)
        cv2.destroyWindow('window_name')

    def toCOCO(self):
        annsId = 0
        cat_id = 1
        images = sorted(glob(join(self.image_root, '*.png')))
        tmp_image = cv2.imread(images[0])
        self.h, self.w = tmp_image.shape[0], tmp_image.shape[1]
        if len(images) != 0:
            print('[INFO] Found {} images in the given folder'.format(len(images)))
        else:
            print('[INFO] Found No images in the given folder. Check your path and subdirectory names. Exiting now!!!!')
            sys.exit()
        count = 0 
        for imageFile in tqdm(images, desc='Processing images'):
            imageId = int(basename(imageFile.split('.')[0]))
            img_data = cv2.imread(imageFile)
            width, height = img_data.shape[1], img_data.shape[0]
            self.new_Anns['images'].append({'license': 1,
                                    'file_name': basename(imageFile),
                                    'coco_url': 'None',
                                    'height': height,
                                    'width': width,
                                    'date_captured': 'None',
                                    'flickr_url': 'None',
                                    'id': imageId})
            segmentations, bboxes, areas, category_ids = self.mask2polygons(imageFile)
            count +=1
            assert len(segmentations) == len(bboxes) == len(areas) == len(category_ids)
            for seg, bbox, area, cat_id in zip(segmentations, bboxes, areas, category_ids):
                self.new_Anns['annotations'].append({
                                            'segmentation': [seg],
                                            'area': area,
                                            'iscrowd': 0,
                                            'image_id': imageId,
                                            'bbox': bbox,
                                            'category_id': cat_id,
                                            'id': annsId})
                annsId += 1

    def save_json(self, anns_dir, anns_file):
        self.anns_dir = anns_dir
        self.anns_file = anns_file
        makedirs(anns_dir, exist_ok=True)
        with open(join(anns_dir, anns_file), 'w+') as f:
            json.dump(self.new_Anns, f, default=float, indent=4)

    def toCityScape(self):
        mask_folder = join(self.source_seg_folder, 'gtFine')
        makedirs(mask_folder, exist_ok=True)
        mask_array = glob(join(self.source_seg_folder, '*.npz'))
        count = 0
        for mask in mask_array:
            mask_data = np.load(mask)
            mask_numpy_data = mask_data.f.array
            mask_image = np.zeros_like(mask_numpy_data, dtype=np.uint8)
            indices = np.unique(mask_numpy_data)
            # https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/docs/tutorials/customize_datasets.md
            # class index has to be between [0, numclasses - 1]            
            for index in indices:
                if index == 99.0:
                    pass
                else:
                    if int(index) == 1:
                        mask_image[np.where(mask_numpy_data==index)] = 0
                    elif int(index) == 2 or int(index) == 3:    # merge both weed classes into one 
                        mask_image[np.where(mask_numpy_data==index)] = 1
            cv2.imwrite(join(mask_folder, basename(mask)[:-4]+'.png'), mask_image)

    def visualize_coco(self):
        import os
        from detectron2.data import MetadataCatalog
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data.catalog import DatasetCatalog
        from detectron2.data.datasets import register_coco_instances
        register_coco_instances("SugarbeetSyn23", {}, join(self.anns_dir, self.anns_file), self.image_root)
        my_dataset_metadata = MetadataCatalog.get("SugarbeetSyn23")
        dataset_dicts = DatasetCatalog.get("SugarbeetSyn23")

        for d in dataset_dicts:
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_metadata, scale=1.0)
            vis = visualizer.draw_dataset_dict(d)
            cv2.imshow('{}'.format(d["file_name"]), vis.get_image()[:, :, ::-1])
            cv2.waitKey()
            cv2.destroyAllWindows()
                