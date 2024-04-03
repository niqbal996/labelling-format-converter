from datumaro.components.dataset import Dataset

coco_path = "/mnt/d/datasets/sugarbeet_syn_datumaro"
output_path = "/mnt/d/datasets/sugarbeet_syn_datumaro"
coco_dataset = Dataset.import_from(coco_path, "coco")

# coco_dataset.export(output_path, format='yolo', save_media=False)
coco_dataset.export(output_path, format='cvat', save_media=False)
print(coco_dataset)