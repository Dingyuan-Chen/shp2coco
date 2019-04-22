import os
import numpy as np
import shutil
import re
import fnmatch

ann_path = 'annotations'
img_path = 'greenhouse_2019'

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

def copy_data(input_path, id, num, mark = 'train'):
    if num != 0:
        list = os.listdir(input_path + '/' + img_path)
        ann_list = os.listdir(input_path + '/' + ann_path)
        if not os.path.isdir(input_path + '/' + mark + '/' + img_path):
            os.makedirs(input_path + '/' + mark + '/' + img_path)
        if not os.path.isdir(input_path + '/' + mark + '/' + ann_path):
            os.makedirs(input_path + '/' + mark + '/' + ann_path)

        for i in range(num):
            shutil.copy(input_path + '/' + img_path + '/' + list[id[i]], input_path + '/' + mark + '/' + img_path
                        + '/' + list[id[i]])
            print('From src: ' + img_path + '/' + list[id[i]] + ' =>dst:' + '/' + mark + '/' + img_path
                  + '/' + list[id[i]])
            annotation_files = filter_for_annotations(input_path, ann_list, list[id[i]])
            for j in range(len(annotation_files)):
                shutil.copy(input_path + '/' + ann_path + '/' + os.path.basename(annotation_files[j]),
                            input_path + '/' + mark + '/' + ann_path + '/' + os.path.basename(annotation_files[j]))

        f = open(input_path + '/' + mark + '/' + mark + '.txt', 'w')
        f.write(str(id))
        f.close()

def slice(input_path, train=0.8, eval=0.2, test=0.0):
    """
    slice the dataset into training, eval and test sub_dataset.
    :param input_path:  path to the original dataset.
    :param train:  the ratio of the training subset.
    :param eval: the ratio of the eval subset.
    :param test: the ratio of the test subset.
    """
    list = os.listdir(input_path + '/' + img_path)
    ann_list = os.listdir(input_path + '/' + ann_path)
    num_list = len(list)
    n_train = int(num_list * train)
    if test == 0:
        n_eval = num_list - n_train
        n_test = 0
    else:
        n_eval = int(num_list * eval)
        n_test = num_list - n_train - n_eval

    img_id = np.arange(num_list)
    np.random.shuffle(img_id)
    train_id, eval_id, test_id = img_id[:n_train], img_id[n_train: n_train+n_eval], img_id[n_train+n_eval:]
    copy_data(input_path, train_id, n_train, 'train')
    copy_data(input_path, eval_id, n_eval, 'eval')
    copy_data(input_path, test_id, n_test, 'test')

if __name__ == '__main__':
    input_path = r'./example_data/original_data/dataset'
    # slice(input_path, train=0.6, eval=0.2, test=0.2)
    slice(input_path)
