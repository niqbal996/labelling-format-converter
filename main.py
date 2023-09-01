import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-folder', type=str, default='/mnt/d/datasets/Corn_syn_dataset/2022_GIL_Paper_Dataset_V2', help='simulation output root folder path')
    parser.add_argument('--output', type=str, default='yolo', choices=['coco', 'yolo', 'coco_seg'], help='output labelling format')

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
    if flags.output == 'coco_seg':
        from formats.coco_seg import COCO_Instance_segmentation
        format = COCO_Instance_segmentation(root_path=flags.root_folder)
        # format.toCityScape()
        format.toCOCO()
        format.save_json(anns_dir='/mnt/d/datasets/sugarbeet_syn_v1/coco_annotations',
                         anns_file='instances_2023_train.json')
        format.visualize_coco()

if __name__ == '__main__':
    main()