import os
import json
import glob
import random

import cv2


class YOLO_converter(object):
    def __init__(self, root_path='/media/naeem/T7/datasets/Maize_dataset_backup',
                 subset=False, remove_class_id=None):
        self.root_path = root_path
        self.subset = subset
        self.boxes = None
        self.remove_id = remove_class_id

    def read_box_labels(self):
        boxes_path = os.path.join(self.root_path, '*_annotations', 'bounding_box', '*.txt')
        self.box_files = glob.glob(boxes_path)
        self.read_boxes()
        # adjusted_labels = self.write_adjusted_boxes()
        # self.read_images()

    def read_boxes(self):
        # read all the files containing box labels
        root_dir = os.path.split(os.path.split(self.box_files[0])[0])[0]
        new_label_dir = os.path.join(root_dir, 'adjusted_bounding_box_remove_class')
        os.makedirs(new_label_dir, exist_ok=True)
        for file_name in self.box_files:
            # read all the boxes from a single file
            with open(file_name) as file:
                boxes = [line.rstrip() for line in file]
                # convert each box to a list
                for box, i in zip(boxes, range(len(boxes))):
                    boxes[i] = list(map(float, box.split()))
                self.boxes = boxes
            # Change the box label IDs from single txt file

            tmp = []
            for box in self.boxes:
                if box[0] == 2.0:  # Corn
                    box[0] = 1
                    tmp.append(box)
                elif box[0] == 3.0:  # Bark
                    box[0] = 2
                elif box[0] == 4.0:
                    box[0] = 0
                    tmp.append(box)
                else:
                    print('Found unexpected labels')

            basename = os.path.basename(file_name)
            with open(os.path.join(new_label_dir, basename), 'w') as file:
                for box in tmp:
                    c=0
                    for item in box:
                        if c == 4:
                            file.write("{}\n".format(item))
                            c=0
                        else:
                            file.write("{} ".format(item))
                            c+=1

    def split_data_by_resolution(self):
        image_path = os.path.join(self.root_path, 'obj_train_data', '*.png')
        data_folder = os.path.join('./obj_train_data')
        image_list = glob.glob(image_path)
        image_subset1 = []
        image_subset2 = []
        for image_path in image_list:
            image = cv2.imread(image_path)
            if image.shape[0:2] == (720, 1280): # Filter by resolution 1280 x 720
                # print(image.shape[0:2])
                image_subset_path = os.path.join(data_folder, os.path.basename(image_path))
                image_subset1.append(image_subset_path)
            else:                               # Filter by other resolution atm 640 x 480
                # print(image.shape[0:2])
                image_subset_path = os.path.join(data_folder, os.path.basename(image_path))
                image_subset2.append(image_subset_path)

        with open(os.path.join(self.root_path, 'val_res1.txt'), 'w') as file:
            for item in image_subset1:
                file.write("{}\n".format(item))

        with open(os.path.join(self.root_path, 'val_res2.txt'), 'w') as file:
            for item in image_subset2:
                file.write("{}\n".format(item))

        print('[INFO] Created new validation splits based on resolutions present in the dataset \n with {} and {} entries in each file'.format(len(image_subset1), len(image_subset2)))

    def fix_labels(self):
        label_path = os.path.join(self.root_path, 'obj_train_data', '*.txt')
        tmp = '/home/naeem/git/labelling-format-converter/test'
        label_list = glob.glob(label_path)
        for file in label_list:
            with open(file, 'r') as f:
                new_labels = []
                labels = f.read().splitlines()
                for label in labels:
                    x = list(map(float, label.split(' ')))
                    x[0] = int(x[0])
                    x[0] = x[0] - 1     # zero indexing
                    if x[0] == 2:       # skip bark
                        continue
                    else:
                        new_labels.append(x)

            new_file = os.path.join(tmp, os.path.split(file)[1])
            with open(new_file, 'w') as f:
                # f.write('\n'.join([' '.join(str(i)) for i in label]))
                for label in new_labels:
                    for item, i in zip(label, range(len(new_labels))):
                        if i+1 == 5:
                            f.write('{}\n'.format(item))
                        else:
                            f.write('{} '.format(item))
                print('hold')

    def split_set(self, train_sz=2500):
        random.seed(20)
        all_files = os.path.join(self.root_path, 'train.txt')
        with open(all_files, 'r') as f:
            all_paths = f.readlines()
            print('hold')
        rand_idx = list(range(0, len(all_paths)))
        random.shuffle(rand_idx)

        with open('new_train.txt', 'w') as f:
            c = 0
            for idx in rand_idx:
                f.write('{}'.format(all_paths[idx]))
                c = c+1
                if c == train_sz:
                    break

        with open('valid.txt', 'w') as f:
            for idx in range(train_sz, len(all_paths)):
                f.write('{}'.format(all_paths[rand_idx[idx]]))

