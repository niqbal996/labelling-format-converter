
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
from shapely.geometry import Polygon, MultiPolygon

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances


class COCO_Instance_segmentation(object):
    def __init__(self, root_path='/media/naeem/T7/datasets/Corn_syn_dataset/',
                 subset=False):
        self.root_path = root_path
        self.subset = subset
        self.boxes = None
        # self.image_root = join(self.root_path, 'main_camera', 'rect')
        # self.source_folder = join(self.root_path, 'main_camera_annotations', 'instance_segmentation')
        # # Note! need semantic segmentation to get the class label in a specified pixel area by the instance segmentation mask. 
        # self.source_seg_folder = join(self.root_path, 'main_camera_annotations', 'semantic_segmentation')
        # self.source_boxes = join(self.root_path, 'main_camera_annotations', 'bounding_box')
        self.image_root = join(self.root_path, 'images')
        self.source_folder = join(self.root_path, 'annotations', 'instance_seg')
        # Note! need semantic segmentation to get the class label in a specified pixel area by the instance segmentation mask. 
        self.source_seg_folder = join(self.root_path, 'annotations', 'semantic_seg')
        self.source_boxes = join(self.root_path, 'annotations', 'bbx')
        self.new_Anns = {}
        self.new_Anns['info'] = {
                        'description': f'Synthetic Beets dataset v2.0',
                        'url': 'None',
                        'version': '1.0',
                        'year': '2023',
                        'contributor': 'AGRIGAIA',
                        'date_created': '9th June,2023'}

        self.new_Anns['licenses'] = [{'url': 'None', 'id': 1, 'name': 'None'}]

        self.new_Anns['categories'] = [{'supercategory': 'plants', 'id': 1, 'name': 'sugarbeets'},
                                {'supercategory': 'plants', 'id': 2, 'name': 'weeds'},
                                ]

        self.new_Anns['images'] = []
        self.new_Anns['annotations'] = []    

    def mask2polygons(self, image_filename):
        # Get the source label file name
        file = join(self.source_folder, '{}.npz'.format(basename(image_filename)[:-4]))
        seg_file = join(self.source_seg_folder, '{}.npz'.format(basename(image_filename)[:-4]))
        # tmp = cv2.imread(image_filename)
        label_image = np.zeros((self.h, self.w), dtype=np.uint16)
        # assumes max objects per image = 1000
        labels = [i  for i in reversed(range(5000))]
        kernel = np.ones((2,2), np.uint8)
        numpy_data = np.load(file)
        seg_data = np.load(seg_file)
        instance_segmentation_mask = np.array(numpy_data.f.array).astype(np.uint16)
        semantic_segmentation_mask = np.array(seg_data.f.array)
        class_ids, _ = np.unique(semantic_segmentation_mask, return_counts=True)         # unique classes
        instance_segmentation_mask += 1     # start indices from 1 
        instance_ids, counts = np.unique(instance_segmentation_mask, return_counts=True) # unique instances

        # # background value
        # background_value = instance_values[np.argmax(counts)]
        # instance_segmentation_mask[np.where(instance_segmentation_mask==0)] = len(instance_values)+1
        # # set 0 as the background index
        # instance_segmentation_mask[np.where(instance_segmentation_mask==background_value)] = 0
        # instance_values = np.delete(instance_values, np.argmax(counts))
        instance_count = len(instance_ids)
        # color_dict = {}
        # r = np.array([randint(0, 255) for p in range(0, instance_count)])
        # g = np.array([randint(0, 255) for p in range(0, instance_count)])
        # b = np.array([randint(0, 255) for p in range(0, instance_count)])
        # for idx in range(instance_count):
        #     color_dict[idx] = [r[idx], g[idx], b[idx]]
        # Since we can only put 
        assert len(labels) > instance_count
        # for value in instance_values:
        #     idx = np.where(instance_segmentation_mask == value)
        #     label_image[idx] = labels[instance_count]
        #     instance_count -= 1
        label_image_color = np.repeat(label_image[:, :, np.newaxis], 3, axis=2)
        all_segmentations = []
        all_boxes = []
        all_areas = []
        all_ids = []    # class IDs
        id_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        instance_mask = np.zeros((self.h, self.w), dtype=np.uint16)
        for id in instance_ids:
            # Fetch semantic ID
            id_mask[np.where(instance_segmentation_mask==id)] = semantic_segmentation_mask[np.where(instance_segmentation_mask==id)]
            val, counts = np.unique(id_mask, return_counts=True)
            ind = np.argmin(counts) # minimum value should be the index of the class ID
            if ind == 0:
                pass
            else:
                semantic_id = int(val[ind])  # class ID of current instance 
                instance_mask[np.where(instance_segmentation_mask==id)] = instance_segmentation_mask[np.where(instance_segmentation_mask==id)]
                instance_mask[np.where(instance_mask > 0)] = 1
                # fill small holes in masks 
                dilated_img = cv2.dilate(instance_mask, kernel, iterations=1) * 255
                dilated_img = dilated_img.astype(np.uint8)
                # contours, _ = cv2.findContours(instance_segmentation_mask.astype(np.int32), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
                contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = np.vstack(contours)              # TODO better contour merging. This one leaves artifacts. 
                approx = cv2.approxPolyDP(contours, 0.002 * cv2.arcLength(contours, True), True).astype(np.int32)
                # approx = cv2.approxPolyDP(contours[0], 0.002 * cv2.arcLength(contours[0], True), True).astype(float)
                for cnt in contours:
                    segmentation = []
                    for point in range(approx.shape[0]):
                        segmentation.append(approx[point, 0, 0])
                        segmentation.append(approx[point, 0, 1])
                x,y,w,h = cv2.boundingRect(approx.astype(int))
                cv2.rectangle(label_image_color,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.drawContours(label_image_color, approx, -1, (0,255, 0), 3)
                area = w*h
                all_segmentations.append(segmentation)
                all_boxes.append([x,y,w,h])
                all_areas.append(area)
                all_ids.append(semantic_id)
                # self.show_image(dilated_img, 'instance mask')
                label_image_color = np.repeat(label_image[:, :, np.newaxis], 3, axis=2)
                # flush old values
                instance_mask = np.zeros((self.h, self.w), dtype=np.uint16)
                id_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        # self.show_image(label_image_color, 'full_mask')

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
        images = glob(join(self.image_root, '*.png'))
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
            json.dump(self.new_Anns, f, default=str)

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
        import random
        register_coco_instances("SugarbeetSyn23", {}, join(self.anns_dir, self.anns_file), self.image_root)
        my_dataset_metadata = MetadataCatalog.get("SugarbeetSyn23")
        dataset_dicts = DatasetCatalog.get("SugarbeetSyn23")

        for d in dataset_dicts:
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_metadata, scale=0.5)
            vis = visualizer.draw_dataset_dict(d)
            cv2.imshow('image', vis.get_image()[:, :, ::-1])
            cv2.waitKey()
            cv2.destroyAllWindows()
                