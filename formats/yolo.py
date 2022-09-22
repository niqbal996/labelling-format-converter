import os
import glob
import random
import shutil
import imagesize
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

    def make_splits(self, files, ratio=0.8):

        res1 = []
        res2 = []
        for file in files:
            if file[-1:] == '\n':
                img_path = os.path.join(self.combi_path, file)[:-1]
            else:
                img_path = os.path.join(self.combi_path, file)
               # image = cv2.imread(img_path)
            # reso = image.shape[0:2]
            reso = imagesize.get(img_path)
            if reso == (640, 480):
                res1.append(file)
            else:
                assert reso == (1280, 720)
                res2.append(file)
        res1.sort()
        res2.sort()
        factor = round(1 / (1 - ratio)) 
        len_res1 = round(len(res1) / factor)
        len_res2 = round(len(res2) / factor)
        valid = res1[0: round(len_res1/2)] + res1[-round(len_res1/2):-1]    # extract half from both ends
        valid = valid + res2[0: round(len_res2/2)] + res2[-round(len_res2/2):-1]

        valid = set(valid)
        train = list(set(res1 + res2) - valid)
        valid = list(valid)

        print('[INFO] Dataset has been divided into TRAIN={} and VALIDATION={} splits'.format(len(train), len(valid)))
        return train, valid

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
        with open(os.path.join(root_path, name), 'w') as f:
            for item in files:
                f.write("{}".format(item))

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
            assert len(labels1) != 0
            assert len(labels2) != 0
            if labels1 > labels2:
                keep.append(item1)
            else:
                keep.append(item2)
        
        print('[INFO] Copying files from the source to target new combined dataset path')
        new_paths = []
        for file in keep:
            name = file[:-4]
            shutil.copy(name+'txt', os.path.join(self.combi_path))
            shutil.copy(name+'png', os.path.join(self.combi_path))
            new_paths.append(os.path.join(self.combi_path, os.path.basename(name)+'png'))

        return new_paths

    def copy_unlabelled_files_to_dest(self, files1, files2, common):
        full_paths_1 = [os.path.join(self.root1, 'obj_train_data', 'top_down_maize_filtered', i) for i in files1]
        full_paths_2 = [os.path.join(self.root2, 'obj_train_data', 'top_down_maize', i) for i in files2]
        full_paths_common = [os.path.join(self.root2, 'obj_train_data', 'top_down_maize', i) for i in common]

        for file in full_paths_1:
            shutil.copy(file[:-1], '/home/naeem/combi/testing')

        for file in full_paths_2:
            shutil.copy(file[:-1], '/home/naeem/combi/testing')

        for file in full_paths_common:
            shutil.copy(file[:-1], '/home/naeem/combi/testing')

        print('[INFO] Copied all unlabelled files to the target folder')

    def copy_labelled_files_to_dest(self, files1, files2, dest):
        full_paths_1 = [os.path.join(self.root1, 'obj_train_data', 'top_down_maize_filtered', i) for i in files1]
        full_paths_2 = [os.path.join(self.root2, 'obj_train_data', 'top_down_maize', i) for i in files2]

        for file in full_paths_1:
            shutil.copy(file[:-4]+'txt', dest)
            shutil.copy(file[:-4]+'png', dest)
        
        for file in full_paths_2:
            shutil.copy(file[:-4]+'txt', dest)
            shutil.copy(file[:-4]+'png', dest)

    
    def merge(self, p1, p2):
        self.root1 = p1
        self.root2 = p2
        files1 = os.path.join(p1, 'train.txt')
        files2 = os.path.join(p2, 'train.txt')
        with open(files1, 'r') as f:
            labels1 = f.readlines()
            print('[INFO] Found {} number of images and labels'.format(len(labels1)))
        with open(files2, 'r') as f:
            labels2 = f.readlines()
            print('[INFO] Found {} number of images and labels'.format(len(labels2)))
        # combi = labels1 + labels2

        label_split1, unlabelled_split1 = self.remove_empty(labels1, p1)
        label_split2, unlabelled_split2 = self.remove_empty(labels2, p2)

        label_split1 = [os.path.basename(i) for i in label_split1]
        label_split2 = [os.path.basename(i) for i in label_split2]
        unlabelled_split1 = [os.path.basename(i) for i in unlabelled_split1]
        unlabelled_split2 = [os.path.basename(i) for i in unlabelled_split2]
        tmp1 = set(label_split1)
        tmp2 = set(label_split2)

        # Process the images and labels common across two tasks and create a new cleaned dataset
        common = tmp2 & tmp1 
        self.combi_path = '/home/naeem/combi/data/'
        new_common_paths = self.keep_labels_with_more_entries(list(common))

        tmpxx = tmp1 - common
        tmpyy = tmp2 - common

        combi_unique = list(tmpxx) + list(tmpyy) 

        tmp1 = set(unlabelled_split1)
        tmp2 = set(unlabelled_split2)
        unlabelled_common = tmp1 & tmp2
        tmp1 = tmp1 - unlabelled_common
        tmp2 = tmp2 - unlabelled_common
        # convert back to lists
        tmp1, tmp2, unlabelled_common = list(tmp1), list(tmp2), list(unlabelled_common)
        # unlabelled_unique = list(tmp1) + list(tmp2)
        self.copy_unlabelled_files_to_dest(tmp1, tmp2, unlabelled_common)
        self.copy_labelled_files_to_dest(tmpxx, tmpyy, self.combi_path)

        train, valid = self.make_splits(combi_unique+list(common))

        # append root directory path to the filenames

        train = [os.path.join('./data/', file) for file in train]
        valid = [os.path.join('./data/', file) for file in valid]
        test = tmp1 + tmp2 + unlabelled_common
        test = [os.path.join('./testing/', file) for file in test]
        train.sort()
        valid.sort()
        test.sort()
        self.write_split_file(train, '/home/naeem/combi/', 'train_sorted.txt')
        self.write_split_file(valid, '/home/naeem/combi/', 'valid_sorted.txt')
        self.write_split_file(test, '/home/naeem/combi/', 'test_sorted.txt')

        random.seed(20)
        random.shuffle(train)
        random.shuffle(valid)
        random.shuffle(test)
        self.write_split_file(train, '/home/naeem/combi/', 'train.txt')
        self.write_split_file(valid, '/home/naeem/combi/', 'valid.txt')
        self.write_split_file(test, '/home/naeem/combi/', 'test.txt')

        print('[INFO] Merged the two datasets and created new train, valid and test YOLO formatted files.')