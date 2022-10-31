import os
import glob
import argparse


# root = "/media/naeem/T7/datasets/Corn_syn_dataset/"
# root = "/media/naeem/T7/datasets/Corn_syn_dataset/"
root = '/media/naeem/T7/datasets/Maize_dataset_backup'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-folder', type=str, default='/media/naeem/T7/datasets/Corn_syn_dataset/', help='simulation output root folder path')
    parser.add_argument('--output', type=str, default='yolo', choices=['coco', 'yolo'], help='output labelling format')

    flags = parser.parse_args()

    if flags.output == 'coco':
        from formats.coco import CoCo_converter
        format = CoCo_converter(root_path=flags.root_folder, subset=False)
        format.YOLO2COCO(rootDir=flags.root_folder, split='train_mini.txt')
        format.read_box_labels()
    if flags.output == 'yolo':
        from formats.yolo import YOLO_converter
        # format1 = YOLO_converter(root_path=flags.root_folder, subset=False, remove_class_id=3)
        format = YOLO_converter(root_path=flags.root_folder, subset=False, remove_class_id=3)
        # format.read_box_labels()
        # format.split_data_by_resolution()
        # format.merge(p1, p2)
        format.fix_labels()

if __name__ == '__main__':
    main()