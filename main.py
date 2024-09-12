import os
import glob
import argparse
# from datumaro.components.dataset import Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-folder', type=str, default='/mnt/d/datasets/Corn_syn_dataset/2022_GIL_Paper_Dataset_V2', help='simulation output root folder path')
    parser.add_argument('--output', type=str, default='coco_seg', choices=['yolo', 'coco_seg', 'coco_panoptic'], help='output labelling format')
    parser.add_argument('--visualize', action='store_true', help='Visualize with detectron API')

    flags = parser.parse_args()

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
        format = COCO_Instance_segmentation(root_path=flags.root_folder,
                                            anns_dir=os.path.join(flags.root_folder, 'main_camera_annotations'),
                                            anns_file='instances_train.json'
                                            )
        format.toCOCO()
        format.save_json(
                        anns_dir=os.path.join(flags.root_folder, 'coco_annotations'),
                        anns_file='instances_train.json')
        if flags.visualize:
            format.visualize_coco()
    if flags.output == 'coco_panoptic':
        from formats.coco_panoptic_seg import COCO_Panoptic_segmentation
        format = COCO_Panoptic_segmentation(root_path=flags.root_folder,
                                            anns_dir=os.path.join(flags.root_folder, 'main_camera_annotations'),
                                            output_dir=flags.root_folder,
                                            )
        format.start_processing()

if __name__ == '__main__':
    main()