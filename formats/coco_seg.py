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
        all_segmentations = []
        all_boxes = []
        all_areas = []
        for value in instance_values:
            label_image = np.zeros((1536, 2048), dtype=np.uint8)
            label_image[np.where(img == value)] = labels[instance_count]
            kernel = np.ones((4,4), np.uint8)
            dilated_img = cv2.dilate(label_image, kernel, iterations=2)
            color_tmp = np.repeat(label_image[:, :, np.newaxis], 3, axis=2)
            contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contours = np.vstack(contours)              # TODO use all contours as list of polygons for disconnected polygons for same object
            approx = cv2.approxPolyDP(contours, 0.002 * cv2.arcLength(contours, True), True).astype(float)
            for cnt in contours:
                segmentation = []
                for point in range(approx.shape[0]):
                    segmentation.append(approx[point, 0, 0])
                    segmentation.append(approx[point,0,1])
            x,y,w,h = cv2.boundingRect(approx.astype(int))
            # cv2.rectangle(label_image_color,(x,y),(x+w,y+h),(0,255,0),2)
            # cv2.drawContours(label_image_color, [approx], -1, (0,255, 0), 3)
            area = w*h
            all_segmentations.append(segmentation)
            all_boxes.append([x,y,w,h])
            all_areas.append(area)
            # self.show_image(dilated_img, 'instance mask')
            # label_image_color = np.repeat(label_image[:, :, np.newaxis], 3, axis=2)
        # self.show_image(label_image_color, 'full_mask')

        return [all_segmentations], all_boxes, all_areas
    
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
        count = 0 
        for imageFile in images[0:2]:
            imageId = int(basename(imageFile.split('.')[0]))
            print('INFO:::: Processing image number {} / {}'.format(count, len(images)))
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
            segmentations, bboxes, areas = self.mask2polygons(imageFile)
            count +=1
            # assert len(segmentations) == len(bboxes) == len(areas)
            for seg, bbox, area in zip(segmentations, bboxes, areas):
                self.new_Anns['annotations'].append({
                                            'segmentation': seg,
                                            'area': area,
                                            'iscrowd': 0,
                                            'image_id': imageId,
                                            'bbox': bbox,
                                            'category_id': cat_id,      # TODO dummy category IDs
                                            'id': annsId})
                annsId += 1

    def save_json(self, anns_dir, anns_file):
        self.anns_dir = anns_dir
        self.anns_file = anns_file
        makedirs(anns_dir, exist_ok=True)
        with open(join(anns_dir, anns_file), 'w+') as f:
            json.dump(self.new_Anns, f, default=str)

    def visualize_coco(self):
        import random
        register_coco_instances("CornSyn23", {}, join(self.anns_dir, self.anns_file), self.image_root)
        my_dataset_metadata = MetadataCatalog.get("CornSyn23")
        dataset_dicts = DatasetCatalog.get("CornSyn23")

        for d in random.sample(dataset_dicts,2):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_metadata, scale=0.5)
            vis = visualizer.draw_dataset_dict(d)
            cv2.imshow('image', vis.get_image()[:, :, ::-1])
            cv2.waitKey()
            cv2.destroyAllWindows()
                