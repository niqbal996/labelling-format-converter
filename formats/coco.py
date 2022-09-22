import os
import sys
import json
import glob
from posixpath import splitdrive
import imagesize
import cv2, json, random, os, PIL
# from detectron2.data import MetadataCatalog
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data.catalog import DatasetCatalog
# from detectron2.data.datasets import register_coco_instances

class CoCo_converter(object):
    def __init__(self, root_path='/media/naeem/T7/datasets/Corn_syn_dataset/',
                 subset=False):
        self.root_path = root_path
        self.subset = subset
        self.boxes = None

    def read_box_labels(self):
        boxes_path = os.path.join(self.root_path, '*_annotations', 'bounding_box', '*.txt')
        self.box_files = glob.glob(boxes_path)
        self.read_boxes()
        self.read_images()
        print('hold')

    def read_boxes(self):
        # read all the files containing box labels
        for file_name in self.box_files:
            # read all the boxes from a single file
            with open(file_name) as file:
                boxes = [line.rstrip() for line in file]
                # convert each box to a list
                for box, i in zip(boxes, range(len(boxes))):
                    boxes[i] = list(map(float, box.split()))
                self.boxes = boxes

    def read_images(self):
        images_path = os.path.join(self.root_path, 'camera_main_camera', 'rect', '*.png')
        self.images = glob.glob(images_path)
        self.base_names = [os.path.basename(image_path)[:-4] for image_path in self.images]

        print('hold')

    def YOLO2COCO(self, rootDir, split='train.txt'):

            imagesDir = os.path.join(rootDir, 'data')
            ogAnnsDir = os.path.join(rootDir, 'data')
            cocoAnnsDir = os.path.join(rootDir, 'coco_annotations')
            os.makedirs(cocoAnnsDir, exist_ok=True)
            yoloAnns = os.path.join(rootDir, split)
            with open(yoloAnns, 'r') as f:
                files = f.readlines()
            # Make the new annotations directory
            cocoAnns = {}
            cocoAnns['info'] = {'description': f'Maize dataset',
                            'url': 'None',
                            'version': '1.0',
                            'year': '2022',
                            'contributor': 'None',
                            'date_created': 'None'}

            cocoAnns['licenses'] = [{'url': 'None', 'id': 1, 'name': 'None'}]

            cocoAnns['categories'] = [{'supercategory': 'plants', 'id': 1, 'name': 'Weeds'},
                                    {'supercategory': 'plants', 'id': 2, 'name': 'Maize'}]

            cocoAnns['images'] = []
            cocoAnns['annotations'] = []
            annsId = 0
            cat_id = 0
            for filename in files:
                if split=='train.txt':
                    imageId = files.index(filename) # Unique from validation split
                elif split=='valid.txt':
                    imageId = files.index(filename) + 2862 # size of train split hard coded for now.
                else:
                    print('[ERROR] Image ID not specified.')
                    sys.exit(-1)
                width, height = imagesize.get(os.path.join(imagesDir, os.path.basename(filename)[:-1]))
                cocoAnns['images'].append({'license': 1,
                                        'file_name': os.path.basename(filename)[:-1],
                                        'coco_url': 'None',
                                        'height': height,
                                        'width': width,
                                        'date_captured': 'None',
                                        'flickr_url': 'None',
                                        'id': imageId})

                with open(os.path.join(ogAnnsDir, os.path.basename(filename)[:-4]+'txt'), 'r') as f:
                    for line in f.readlines():
                        newBbox = [float(i) for i in line[:-1].split(' ')[1:]]
                        newBbox[0] *= width
                        newBbox[1] *= height
                        newBbox[2] *= width
                        newBbox[3] *= height
                        newBbox[0] = newBbox[0] - newBbox[2] / 2
                        newBbox[1] = newBbox[1] - newBbox[3] / 2
                        newBbox = [round(i, 2) for i in newBbox]
                        if int(float(line.split(' ')[0])) == 0:
                            cat_id = 1
                        elif int(float(line.split(' ')[0])) == 1:
                            cat_id = 2
                        else:
                            cat_id = 0
                            continue #skip bark ID
                        cocoAnns['annotations'].append({'segmentation': [],
                                                    'area': newBbox[2] * newBbox[3],
                                                    'iscrowd': 0,
                                                    'image_id': imageId,
                                                    'bbox': newBbox,
                                                    'category_id': cat_id,
                                                    'id': annsId})
                        annsId += 1
            with open(os.path.join(cocoAnnsDir, '{}_2022.json'.format(os.path.splitext(split)[0])), 'w+') as f:
                json.dump(cocoAnns, f)

            # register_coco_instances("new_dataset", {}, os.path.join(cocoAnnsDir, f'{datasetName}NewAnnotations.json'), imagesDir)
            # my_dataset_metadata = MetadataCatalog.get("new_dataset")
            # dataset_dicts = DatasetCatalog.get("new_dataset")

            # for d in random.sample(dataset_dicts, 3):
            #     img = cv2.imread(d["file_name"])
            #     visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_metadata, scale=2)
            #     vis = visualizer.draw_dataset_dict(d)
            #     cv2.imshow('image', vis.get_image()[:, :, ::-1])
            #     cv2.waitKey()
            #     cv2.destroyAllWindows()


