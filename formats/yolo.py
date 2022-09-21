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

    # def split_data_by_resolution(self, paths):
    #     # image_path = os.path.join(self.root_path, 'obj_train_data', '*.png')
    #     # data_folder = os.path.join('./obj_train_data')
    #     image_list = glob.glob(paths)
    #     image_subset1 = []
    #     image_subset2 = []
    #     for image_path in image_list:
    #         image = cv2.imread(image_path)
    #         if image.shape[0:2] == (720, 1280): # Filter by resolution 1280 x 720
    #             # print(image.shape[0:2])
    #             image_subset_path = os.path.join(data_folder, os.path.basename(image_path))
    #             image_subset1.append(image_subset_path)
    #         else:                               # Filter by other resolution atm 640 x 480
    #             # print(image.shape[0:2])
    #             image_subset_path = os.path.join(data_folder, os.path.basename(image_path))
    #             image_subset2.append(image_subset_path)

    #     # with open(os.path.join(self.root_path, 'val_res1.txt'), 'w') as file:
    #     #     for item in image_subset1:
    #     #         file.write("{}\n".format(item))

    #     # with open(os.path.join(self.root_path, 'val_res2.txt'), 'w') as file:
    #     #     for item in image_subset2:
    #     #         file.write("{}\n".format(item))

    #     print('[INFO] Created new validation splits based on resolutions present in the dataset \n with {} and {} entries in each file'.format(len(image_subset1), len(image_subset2)))

    def adjust_splits(self, paths, split='train'):
        image_paths = os.path.join(self.root1, paths)
        print('hold')

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

    def write_split_file(self, files, root_path, name='test.txt'):
        with open(os.path.join(root_path, '{}'.format(name)), 'w') as file:
            for file in files:
                file.write("{}".format(file))

    def remove_empty(self, list_of_files, root_path):
        with_labels = []
        without_labels = []
        for file in list_of_files:
            if file[0] == '/':
                path_ = os.path.join(root_path, file[1:-4]+'txt')
            else:
                path_ = os.path.join(root_path, file[0:-4]+'txt')
            with open(path_, 'r') as f:
                boxes = f.readlines()
            if len(boxes) > 0: 
                with_labels.append(file)
            else:
                without_labels.append(file)
        print('[INFO] Kept {} files with labels!'.format(len(with_labels)))
        print('[INFO] Kept {} files without any labels!'.format(len(without_labels)))   
        print('[INFO] TOTAL =======> {} + {} = {}'.format(len(with_labels), 
        len(without_labels), 
        len(with_labels)+len(without_labels)))     

        return with_labels, without_labels

    def keep_labels_with_more_entries(self, files):
        full_paths_1 = [os.path.join(self.root1, 'obj_train_data', 'top_down_maize_filtered', i) for i in files]
        full_paths_2 = [os.path.join(self.root2, 'obj_train_data', 'top_down_maize', i) for i in files]

        keep = []
        for item1, item2 in zip(full_paths_1, full_paths_2):
            with open(item1[:-4]+'txt', 'r') as f:
                labels1 = f.readlines()
            with open(item2[:-4]+'txt', 'r') as f:
                labels2 = f.readlines()

            if labels1 > labels2:
                keep.append(item1)
            else:
                keep.append(item2)

        return keep
    
    def merge(self, p1, p2):
        self.root1 = p1
        self.root2 = p2
        files1 = os.path.join(p1, 'train.txt')
        files2 = os.path.join(p2, 'train.txt')
        with open(files1, 'r') as f:
            labels1 = f.readlines()
            print(len(labels1))
        with open(files2, 'r') as f:
            labels2 = f.readlines()
            print(len(labels2))
        # combi = labels1 + labels2

        label_split1, unlabelled_split1 = self.remove_empty(labels1, p1)
        label_split2, unlabelled_split2 = self.remove_empty(labels2, p2)

        label_split1 = [os.path.basename(i) for i in label_split1]
        label_split2 = [os.path.basename(i) for i in label_split2]
        unlabelled_split1 = [os.path.basename(i) for i in unlabelled_split1]
        unlabelled_split2 = [os.path.basename(i) for i in unlabelled_split2]
        tmp1 = set(label_split1)
        tmp2 = set(label_split2)
        common = tmp2 & tmp1 
        common_tmp = self.keep_labels_with_more_entries(list(common))
        tmpxx = tmp1 - common
        tmpyy = tmp2 - common


        combi = list(tmpxx) + list(tmpyy) 

        # all_labelled = label_split1 + label_split2
        # all_unlabelled = unlabelled_split1 + unlabelled_split2

        # self.write_split_file(files=all_labelled, root_path=p1, name='train.txt')
        self.adjust_splits(label_split1, p1, split='train')

        self.write_split_file(files=all_unlabelled, root_path=p1, name='test.txt')
        self.write_split_file(files=all_labelled, root_path=p1, name='valid.txt')


        print('hold')