from glob import glob
import os 
files = glob('/home/niqbal/git/syclops/output/iso_barrel_v1/main_camera_annotations/bounding_box/*.txt')

tmp = []
for file in sorted(files):
    with open(file, 'r+') as f:
        labels = f.readlines()
        for label in labels:
            tmp.append(label.replace('5 ', '0 ')) 
        print('hold')
    basename = os.path.basename(file)
    with open('/home/niqbal/git/syclops/output/iso_barrel_v1/main_camera_annotations/bounding_box/tmp/{}'.format(basename), 'w+') as f:
        for line in tmp:
            f.write('{}'.format(line))
