#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from tif_process import *
from slice_dataset import slice

# root path for saving the tif and shp file.
ROOT = r'./example_data/original_data'
img_path = 'img'
shp_path = 'shp'
# root path for saving the mask.
ROOT_DIR = ROOT + '/dataset'
IMAGE_DIR = os.path.join(ROOT_DIR, "greenhouse_2019")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")

clip_size = 512

INFO = {
    "description": "Greenhouse Dataset",
    "url": "",
    "version": "0.1.0",
    "year": 2019,
    "contributor": "DuncanChen",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "",
        "url": ""
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'greenhouse',
        'supercategory': 'building',
    },
]

def filter_for_jpeg(root, files):
    # file_types = ['*.jpeg', '*.jpg']
    file_types = ['*.tiff', '*.tif']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_annotations(root, files, image_filename):
    # file_types = ['*.png']
    file_types = ['*.tif']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    # file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    # files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    files = [f for f in files if basename_no_extension == os.path.splitext(os.path.basename(f))[0].split('_', 1)[0]]

    return files

def from_mask_to_coco(root, MARK, IMAGE, ANNOTATION):
    ROOT_DIR = root + '/' + MARK
    IMAGE_DIR = ROOT_DIR + '/' + IMAGE
    ANNOTATION_DIR = ROOT_DIR + '/' + ANNOTATION
    if os.path.exists(ROOT_DIR):
        coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }

        image_id = 1
        segmentation_id = 1

        # filter for jpeg images
        for root, _, files in os.walk(IMAGE_DIR):
            image_files = filter_for_jpeg(root, files)

            # go through each image
            for image_filename in image_files:
                image = Image.open(image_filename)
                image_info = pycococreatortools.create_image_info(
                    image_id, os.path.basename(image_filename), image.size)
                coco_output["images"].append(image_info)

                # filter for associated png annotations
                for root, _, files in os.walk(ANNOTATION_DIR):
                    annotation_files = filter_for_annotations(root, files, image_filename)

                    # go through each associated annotation
                    for annotation_filename in annotation_files:

                        print(annotation_filename)
                        class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                        binary_mask = np.asarray(Image.open(annotation_filename)
                                                 .convert('1')).astype(np.uint8)

                        annotation_info = pycococreatortools.create_annotation_info(
                            segmentation_id, image_id, category_info, binary_mask,
                            image.size, tolerance=2)

                        if annotation_info is not None:
                            coco_output["annotations"].append(annotation_info)

                        segmentation_id = segmentation_id + 1

                image_id = image_id + 1

        with open('{}/instances_greenhouse_{}2019.json'.format(ROOT_DIR, MARK), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)
    else:
        print(ROOT_DIR + ' does not exit!')

def main():
    clip_from_file(clip_size, ROOT, img_path, shp_path)
    slice(ROOT_DIR, train=0.6, eval=0.2, test=0.2)
    from_mask_to_coco(ROOT_DIR, 'train', "greenhouse_2019", "annotations")
    from_mask_to_coco(ROOT_DIR, 'eval', "greenhouse_2019", "annotations")
    from_mask_to_coco(ROOT_DIR, 'test', "greenhouse_2019", "annotations")

if __name__ == "__main__":
    main()
