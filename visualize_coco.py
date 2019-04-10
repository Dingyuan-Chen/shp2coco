from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os

ROOT_DIR = './example_data/dataset'
image_directory = os.path.join(ROOT_DIR, "greenhouse_2019")
annotation_file = os.path.join(ROOT_DIR, "instances_greenhouse_train2019.json")

example_coco = COCO(annotation_file)

category_ids = example_coco.getCatIds(catNms=['square'])
image_ids = example_coco.getImgIds(catIds=category_ids)
image_data = example_coco.loadImgs(image_ids[2])[0]

image = io.imread(image_directory + '/' + image_data['file_name'])
plt.imshow(image); plt.axis('off')
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
annotations = example_coco.loadAnns(annotation_ids)
example_coco.showAnns(annotations)
plt.show()