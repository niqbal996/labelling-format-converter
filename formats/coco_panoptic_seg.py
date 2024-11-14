
import sys
import json
import numpy as np
import cv2
import skimage 
import concurrent.futures
from random import randint

from os.path import join, basename
from os import makedirs
from glob import glob
from tqdm import tqdm
from skimage import measure
import imutils 
from shapely.geometry import Polygon, MultiPolygon
from panopticapi.utils import rgb2id, id2rgb
from PIL import Image

ids = []

def process_image(filepath, idx):
    return {
        "filename": basename(filepath),
        "height": 1024,
        "width": 1024,
        "id": str(basename(filepath[:-4])),
    }

class COCO_Panoptic_segmentation(object):
    def __init__(self, root_path='/media/naeem/T7/datasets/Corn_syn_dataset/',
                 anns_dir=None,
                 output_dir=None,
                 subset_size=10, 
                 split='train'):
        self.root_path = root_path
        self.anns_dir = anns_dir
        self.out_dir = output_dir
        self.split = split
        self.boxes = None
        self.segment_counter = 0
        self.subset_size = subset_size
        self.image_root = join(self.root_path, 'main_camera', 'rect')
        self.source_instance_segmentation = join(self.root_path, 'main_camera_annotations', 'instance_segmentation')
        self.source_segmentation_folder = join(self.root_path, 'main_camera_annotations', 'semantics')
        self.source_boxes = join(self.root_path, 'main_camera_annotations', 'bounding_box')
        self.panoptic_labels= join(self.root_path, 'plants_panoptic_{}'.format(split))
        self.semantic_labels= join(self.root_path, 'plants_panoptic_semseg_{}'.format(split))
        self.background_category_id = 3
        makedirs(self.panoptic_labels, exist_ok=True)
        makedirs(self.semantic_labels, exist_ok=True)
        self.new_Anns = {}
        self.new_Anns['info'] = {
                        'description': f'Synthetic Sugarbeets dataset v2.0',
                        'url': 'None',
                        'version': '1.0',
                        'year': '2024',
                        'contributor': 'LALWECO',
                        'date_created': '03 April,2024'}

        self.new_Anns['licenses'] = [{'url': 'None', 'id': 1, 'name': 'None'}]

        self.new_Anns['categories'] = [
                                {'supercategory': 'background', 'id': 3, "isthing": 0, 'name': 'soil', 'color': [10, 10, 10]},
                                {'supercategory': 'plants', 'id': 1, "isthing": 1, 'name': 'sugarbeet', 'color': [111, 74, 0]},
                                {'supercategory': 'plants', 'id': 2, "isthing": 1,'name': 'weed', 'color': [230, 150, 140]},
                                ]   
        
    def images2json(self):
        if self.subset_size != 0:
            image_files = sorted(glob(join(self.image_root, '*.png')))[0:self.subset_size]
        else:
            image_files = sorted(glob(join(self.image_root, '*.png')))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create a list of future objects
            futures = [executor.submit(process_image, filepath, idx) 
                    for idx, filepath in enumerate(image_files)]
            # Collect results as they complete
            images = [future.result() for future in concurrent.futures.as_completed(futures)]
        return images
    
    @staticmethod
    def normalize_uint16_to_uint8(input_array):
        """
        Normalize and scale a uint16 array to uint8.
        
        Parameters:
        input_array (numpy.ndarray): Input array of type uint16
        
        Returns:
        numpy.ndarray: Normalized and scaled array of type uint8
        """
        if input_array.dtype != np.uint16:
            raise ValueError("Input array must be of type uint16")
        
        # Ensure the input is a numpy array
        input_array = np.asarray(input_array)
        
        # Find the minimum and maximum values in the input array
        min_val = np.min(input_array)
        max_val = np.max(input_array)
        
        # Normalize the array to float in range [0, 1]
        normalized = (input_array - min_val) / (max_val - min_val)
        
        # Scale to uint8 range [0, 255] and convert
        scaled = (normalized * 255).astype(np.uint8)
        
        return scaled
        
    @staticmethod
    def process_label_file(args):
        label_file_path, idx, self = args
        numpy_data = np.load(label_file_path)
        image_id = str(basename(label_file_path)[:-4])
        semantic_mask = np.asarray(Image.open(join(self.source_segmentation_folder, 
                                                basename(label_file_path)[:-3]+'png')))
        label_array = numpy_data['array'].astype(np.uint16)
        label_array = self.normalize_uint16_to_uint8(label_array)
        segmentIds, counts = np.unique(label_array, return_counts=True)
        background_idx = segmentIds[counts.argmax()]
        pan_format = np.zeros(
            (semantic_mask.shape[0], semantic_mask.shape[1], 3), dtype=np.uint8
        )
        segmInfo = []
        for segment_id in segmentIds:
            mask = label_array == segment_id
            semantic_ID = semantic_mask[mask][0]
            segment_color = id2rgb(segment_id)
            pan_format[mask] = segment_color
            if semantic_ID == 0 and segment_id == background_idx:
                # black color for background in both semantic and panoptic masks
                segment_color = [0, 0, 0]
                pan_format[mask] = segment_color
                segmInfo.append({"id": int(0),
                                "category_id": int(self.background_category_id),
                                "area": int(np.sum(mask)),
                                "bbox": [0, 0, 1024, 1024],
                                "iscrowd": 0})
            else:
                area = np.sum(mask)
                hor = np.sum(mask, axis=0)
                hor_idx = np.nonzero(hor)[0]
                x = hor_idx[0]
                width = hor_idx[-1] - x + 1
                vert = np.sum(mask, axis=1)
                vert_idx = np.nonzero(vert)[0]
                y = vert_idx[0]
                height = vert_idx[-1] - y + 1
                bbox = [int(x), int(y), int(width), int(height)]
                segmInfo.append({"id": int(segment_id) if segment_id != 0 else int(1),
                                "category_id": int(semantic_ID),
                                "area": int(area),
                                "bbox": bbox,
                                "iscrowd": 0})
            self.segment_counter += 1
        
        annotation = {'image_id': image_id,
                    'file_name': image_id+'_panoptic.png',
                    "segments_info": segmInfo}
        
        Image.fromarray(pan_format).save(join(self.panoptic_labels, basename(label_file_path)[:-4]+'_panoptic.png'))
        # Change background to 3 in semantic mask
        mask = semantic_mask == 0
        semantic_new = semantic_mask.copy()
        semantic_new[mask] = self.background_category_id
        Image.fromarray(semantic_mask).save(join(self.semantic_labels, basename(label_file_path)[:-4]+'_semantic.png'))
        return annotation
    
    def syclops_panoptic2coco_panoptic(self):
        if self.subset_size != 0:
            panoptic_syclops_labels = sorted(glob(join(self.source_instance_segmentation, '*.npz')))[0:self.subset_size]
        else:
            panoptic_syclops_labels = sorted(glob(join(self.source_instance_segmentation, '*.npz')))
    
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_label_file, (label_file_path, idx, self)) 
                    for idx, label_file_path in enumerate(panoptic_syclops_labels)]
            
            annotations = []
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(panoptic_syclops_labels), desc="Processing labels"):
                annotations.append(future.result())
        
        return annotations
            
    def start_processing(self):
        self.new_Anns['images'] = self.images2json()
        self.new_Anns['annotations'] = self.syclops_panoptic2coco_panoptic()
        self.save_json(anns_dir=self.root_path, anns_file='plants_panoptic_{}_{}.json'.format(self.split, self.subset_size))
        
    def save_json(self, anns_dir, anns_file):
        self.anns_dir = anns_dir
        self.anns_file = anns_file
        makedirs(anns_dir, exist_ok=True)
        with open(join(anns_dir, anns_file), 'w+') as f:
            json.dump(self.new_Anns, f, default=float)
            
    def show_image(self, img, window_name):
        cv2.namedWindow("window_name", cv2.WINDOW_NORMAL)
        cv2.moveWindow("window_name", 40,30)
        cv2.resizeWindow("window_name", 1200, 1400)
        cv2.imshow('window_name', img)
        cv2.waitKey(0)
        cv2.destroyWindow('window_name')

    
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
                