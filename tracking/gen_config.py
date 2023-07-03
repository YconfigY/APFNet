import os
import json
import numpy as np
from data.generate_dataset_list import dataset_dict


def gen_config(seq_path, dataset):
    """generate dataset's sequence config such as visible and infrared images list and groundtruth bounding box

    Args:
        seq_path (str): dataset's sequence path
        dataset (str): dateset name

    Returns:
        img_list_visible (list[str]): all visible image files' path in current sequence form a list
        img_list_infrared (list[str]): all infrared image files' path in current sequence form a list
        gt (list[np.ndarray]): all target groundtruth in current sequence form a list
    """
    path, seqname = os.path.split(seq_path)
    img_list_visible = sorted([os.path.join(seq_path, dataset_dict[dataset]['visible'], img_name)
                               for img_name in os.listdir(os.path.join(seq_path, dataset_dict[dataset]['visible']))
                               if os.path.splitext(img_name)[1] in ['.jpg', '.png', '.bmp']])
    img_list_infrared = sorted([os.path.join(seq_path, dataset_dict[dataset]['infrared'], img_name)
                                for img_name in os.listdir(os.path.join(seq_path, dataset_dict[dataset]['infrared']))
                                if os.path.splitext(img_name)[1] in ['.jpg', '.png', '.bmp']])
    if dataset == 'GTOT':
        gt = np.loadtxt(seq_path + '/init.txt', delimiter='	')
    else:
        gt = np.loadtxt(seq_path + '/init.txt', delimiter=',')
    
    if gt.shape[1] == 8:
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)
    return img_list_visible, img_list_infrared, gt
