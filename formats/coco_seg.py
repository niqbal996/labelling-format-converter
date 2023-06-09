from os.path import join, basename
from os import makedirs
import json
from glob import glob
import numpy as np
import cv2
import skimage 
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
        self.image_root = join(self.root_path, 'camera_main_camera', 'rect')
        self.source_folder = join(self.root_path, 'camera_main_camera_annotations', 'instance_segmentation')
        self.new_Anns = {}
        self.new_Anns['info'] = {
                        'description': f'Synthetic dataset v2.0',
                        'url': 'None',
                        'version': '2.0',
                        'year': '2023',
                        'contributor': 'AGRIGAIA',
                        'date_created': '9th June,2023'}

        self.new_Anns['licenses'] = [{'url': 'None', 'id': 1, 'name': 'None'}]

        self.new_Anns['categories'] = [{'supercategory': 'plants', 'id': 1, 'name': 'weed'},
                                {'supercategory': 'plants', 'id': 2, 'name': 'maize'},
                                {'supercategory': 'residue', 'id': 3, 'name': 'bark'}]

        self.new_Anns['images'] = []
        self.new_Anns['annotations'] = []    

    def mask2polygons(self, image_filename):
        # Get the source label file name
        file = join(self.source_folder, '{}.npz'.format(basename(image_filename)[:-4]))
        label_image = np.zeros((1536, 2048), dtype=np.uint8)
        labels = [i  for i in reversed(range(255))]
        numpy_data = np.load(file)
        img = np.array(numpy_data.f.array)
        # unique indices
        instance_values, counts = np.unique(img, return_counts=True)
        # # background value
        background_value = instance_values[np.argmax(counts)]
        instance_values = np.delete(instance_values, np.argmax(counts))
        instance_count = len(instance_values)
        assert len(labels) > instance_count
        for value in instance_values:
            label_image[np.where(img == value)] = labels[instance_count]
            instance_count -= 1
        label_image_color = np.repeat(label_image[:, :, np.newaxis], 3, axis=2)

        for value in instance_values:
            label_image = np.zeros((1536, 2048), dtype=np.uint8)
            label_image[np.where(img == value)] = labels[instance_count]
            kernel = np.ones((3,3), np.uint8)
            dilated_img = cv2.dilate(label_image, kernel, iterations=2)
            color_tmp = np.repeat(label_image[:, :, np.newaxis], 3, axis=2)
            contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contours = np.vstack(contours)
            approx = cv2.approxPolyDP(contours, 0.004 * cv2.arcLength(contours, True), True)
            segmentation = []
            for point in range(approx.shape[0]):
                segmentation.append(approx[point, 0, 0])
                segmentation.append(approx[point,0,1])
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(label_image_color,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.drawContours(label_image_color, [approx], -1, (0,255, 0), 3)
            area = w*h
            # cv2.imshow('mask', color_tmp)
            # cv2.waitKey(0)
            # cv2.destroyWindow('mask')
            # label_image_color = np.repeat(label_image[:, :, np.newaxis], 3, axis=2)
        # cv2.imshow('mask', label_image_color)
        # cv2.waitKey(0)
        # cv2.destroyWindow('mask')
        # cv2.imshow('mask', label_image)
        # cv2.waitKey(0)
        # cv2.destroyWindow('mask')

        return segmentation, [x,y,w,h], area
            
    def toCOCO(self):
        annsId = 0
        cat_id = 0
        images = glob(join(self.image_root, '*.png'))
        for imageFile in images:
            imageId = int(basename(imageFile.split('.')[0]))

            img_data = cv2.imread(imageFile)
            width, height = img_data.shape[1], img_data.shape[0]
            self.new_Anns['images'].append({'license': 1,
                                    'file_name': imageFile,
                                    'coco_url': 'None',
                                    'height': height,
                                    'width': width,
                                    'date_captured': 'None',
                                    'flickr_url': 'None',
                                    'id': imageId})
            segmentation, bbox, area = self.mask2polygons(imageFile)
            self.new_Anns['annotations'].append({
                                        'segmentation': segmentation,
                                        'area': area,  # Not sure about this
                                        'iscrowd': 0,
                                        'image_id': imageId,
                                        'bbox': bbox,
                                        'category_id': cat_id,
                                        'id': annsId})
            annsId += 1

    def create_sub_masks(self, img):
        labels = [i  for i in reversed(range(255))]
        instance_values, counts = np.unique(img, return_counts=True)
        # # background value
        background_value = instance_values[np.argmax(counts)]
        instance_values = np.delete(instance_values, np.argmax(counts))
        instance_count = len(instance_values)
        sub_masks = []
        for value in instance_values:
            sub_mask = np.zeros((1536, 2048), dtype=np.uint8)
            sub_mask[np.where(img == value)] = labels[instance_count]
            sub_masks.append(sub_mask)
        return sub_masks

    def save_json(self, anns_dir, anns_file):
        self.anns_dir = anns_dir
        self.anns_file = anns_file
        makedirs(anns_dir, exist_ok=True)
        with open(join(anns_dir, '{}.json'.format(anns_file)), 'w+') as f:
            json.dump(self.newAnns, f)

    def visualize_coco(self):
        import random
        register_coco_instances("CornSyn23", {}, join(self.anns_dir, self.anns_file), self.image_root)
        my_dataset_metadata = MetadataCatalog.get("CornSyn23")
        dataset_dicts = DatasetCatalog.get("CornSyn23")

        for d in random.sample(dataset_dicts, 3):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_metadata, scale=2)
            vis = visualizer.draw_dataset_dict(d)
            cv2.imshow('image', vis.get_image()[:, :, ::-1])
            cv2.waitKey()
            cv2.destroyAllWindows()
                