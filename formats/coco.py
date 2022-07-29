import os
import json
import glob

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



