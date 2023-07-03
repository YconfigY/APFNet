# from ast import main
import os
import numpy as np
import pickle
from collections import OrderedDict

from utils.config import cfg
# def construct_db(dataset_dir, challenge_type):
    
dataset_dict = {'RGBT234':{'visible': 'visible','infrared':'infrared'},
                'GTOT':{'visible': 'v','infrared':'i'},
                'LasHeR':{'visible': 'visible','infrared':'infrared'}}
#     for dataset_name in dataset_dict:
#         dataset_path = os.path.join(dataset_dir, dataset_name)
        
#         seq_list_flie = os.path.join(dataset_path, dataset_name) + '.txt'  # modify it to yourself
        
#         with open(seq_list_flie, 'r') as fp:
#             seq_list = fp.read().splitlines()
            
#         # Construct db
#         data = OrderedDict()
#         for i, seq_name in enumerate(seq_list):
            
#             seq_abs_path_v = os.path.join(dataset_path, seq_name, dataset_dict[dataset_name]['visible'])
#             seq_abs_path_i = os.path.join(dataset_path, seq_name, dataset_dict[dataset_name]['infrared'])
            
#             seq_path_v = os.path.join(dataset_name, seq_name, dataset_dict[dataset_name]['visible'])
#             seq_path_i = os.path.join(dataset_name, seq_name, dataset_dict[dataset_name]['infrared'])
            
#             # get img list(filename + name extension)
#             img_list_v = sorted(
#                 [p for p in os.listdir(seq_abs_path_v)
#                 if os.path.splitext(p)[1] in ['.jpg', '.bmp', '.png']])
#             img_list_i = sorted(
#                 [p for p in os.listdir(seq_abs_path_i)
#                 if os.path.splitext(p)[1] in ['.jpg', '.bmp', '.png']])
            
#             # get img abs pth list(path + filename + name extension)
#             img_list_v = [os.path.join(seq_path_v, img) for img in img_list_v]
#             img_list_i = [os.path.join(seq_path_i, img) for img in img_list_i]
            
#             # load seq gt list from txt file
#             if dataset_name == 'GTOT':
#                 gt = np.loadtxt(os.path.join(dataset_path, seq_name + '/init.txt'))
#             elif dataset_name == 'RGBT234':
#                 gt = np.loadtxt(os.path.join(dataset_path, seq_name + '/init.txt'), delimiter=',')
                
#             assert len(img_list_v) == len(gt) == len(img_list_i), "Lengths do not match!!"
#             if gt.shape[1] == 8:
#                 x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
#                 y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
#                 x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
#                 y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
#                 gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)
            
#             if challenge_dict.__contains__(challenge_type):
#                 challenge_inf = challenge_dict[challenge_type] + '.tag'
#                 try:
#                     challenge_label = np.loadtxt(os.path.join(dataset_path, seq_name, challenge_inf)).tolist()
#                     assert len(challenge_label) == len(img_list_v), 'len(challenge_label)!=len(img_list_v):'
#                     challenge_label = np.array(challenge_label)
#                     idx = np.ones(len(img_list_v), dtype=bool)
#                     idx *= (challenge_label > 0)
#                     img_list_v = np.array(img_list_v)
#                     img_list_i = np.array(img_list_i)
#                     gt = gt[idx, :]
#                     img_list_v = img_list_v[idx].tolist()
#                     img_list_i = img_list_i[idx].tolist()
#                     print(seq_name, challenge_type, len(img_list_v), len(gt))  # modify py3 to py2
#                     if len(img_list_v) > 0:
#                         data[seq_name] = {'images_v': img_list_v, 'images_i': img_list_i, 'gt': gt}
#                     else:
#                         print(seq_name, 'length not enough!')  # modify py3 to py2
#                 except:
#                     print(seq_name, 'no', challenge_type)  # modify py3 to py2
                    
#             if challenge_type == 'ALL':
#                 data[seq_name] = {'images_v': img_list_v, 'images_i': img_list_i, 'gt': gt}
                
#         with open(data_list, 'ab') as fp:
#             print('generate', dataset_name, challenge_type, 'data_list, and write in', data_list)  # modify py3 to py2
#             pickle.dump(data, fp)


def construct_db(challenge_type):
    dataset_path = os.path.join(cfg.DATA.DATA_PATH, cfg.DATA.DATASET)
    seq_list_flie = os.path.join(dataset_path, cfg.DATA.DATASET) + '.txt'  # modify it to yourself
        
    with open(seq_list_flie, 'r') as fp:
        seq_list = fp.read().splitlines()
            
    # Construct db
    data = OrderedDict()
    for i, seq_name in enumerate(seq_list):
        img_dir_name = []
        if 'v' in os.listdir(os.path.join(dataset_path, seq_name)):
            img_dir_name = ['v', 'i']
            # load seq gt list from txt file
            gt = np.loadtxt(os.path.join(dataset_path, seq_name) + '/init.txt')
        elif 'visible' in os.listdir(os.path.join(dataset_path, seq_name)):
            img_dir_name = ['visible', 'infrared']
            gt = np.loadtxt(os.path.join(dataset_path, seq_name) + '/init.txt', delimiter=',')
        
        seq_abs_path_v = os.path.join(dataset_path, seq_name, img_dir_name[0])
        seq_abs_path_i = os.path.join(dataset_path, seq_name, img_dir_name[1])
            
        seq_path_v = os.path.join(seq_name, img_dir_name[0])
        seq_path_i = os.path.join(seq_name, img_dir_name[1])
            
        # get img list(filename + name extension)
        img_list_v = sorted(
            [p for p in os.listdir(seq_abs_path_v)
            if os.path.splitext(p)[1] in ['.jpg', '.bmp', '.png']])
        img_list_i = sorted(
            [p for p in os.listdir(seq_abs_path_i)
            if os.path.splitext(p)[1] in ['.jpg', '.bmp', '.png']])
        
        # get img abs pth list(path + filename + name extension)
        img_list_v = [os.path.join(seq_path_v, img) for img in img_list_v]
        img_list_i = [os.path.join(seq_path_i, img) for img in img_list_i]

        assert len(img_list_v) == len(gt) == len(img_list_i), "Lengths do not match!!"
        if gt.shape[1] == 8:
            x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
            y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
            x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
            y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
            gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)
            
        if challenge_dict.__contains__(challenge_type):
            challenge_inf = challenge_dict[challenge_type] + '.tag'
            try:
                challenge_label = np.loadtxt(os.path.join(dataset_path, seq_name, challenge_inf)).tolist()
                assert len(challenge_label) == len(img_list_v), 'len(challenge_label)!=len(img_list_v):'
                challenge_label = np.array(challenge_label)
                idx = np.ones(len(img_list_v), dtype=bool)
                idx *= (challenge_label > 0)
                img_list_v = np.array(img_list_v)
                img_list_i = np.array(img_list_i)
                gt = gt[idx, :]
                img_list_v = img_list_v[idx].tolist()
                img_list_i = img_list_i[idx].tolist()
                print(seq_name, challenge_type, len(img_list_v), len(gt))  # modify py3 to py2
                if len(img_list_v) > 0:
                    data[seq_name] = {'images_v': img_list_v, 'images_i': img_list_i, 'gt': gt}
                else:
                    print(seq_name, 'length not enough!')  # modify py3 to py2
            except:
                print(seq_name, 'no', challenge_type)  # modify py3 to py2
                    
        if challenge_type == 'ALL':
            data[seq_name] = {'images_v': img_list_v, 'images_i': img_list_i, 'gt': gt}
                
    with open(data_list, 'ab') as fp:
        print('generate', challenge_type, 'data_list, and write in', data_list)  # modify py3 to py2
        pickle.dump(data, fp)


if __name__ == "__main__":
    # Save db
    challenge_dict = {'FM': 'fast_motion', 'SV': 'size_change', 
                      'OCC': 'occlusion', 'ILL': 'illum_change', 'TC': 'thermal_crossover'}
    # ALL~Whole dataset
    # construct_db(challenge)
    challenges = ['ALL']
    
    # set challenge type. 
    # challenges = cfg.DATA.CHALLENGE
    for challenge in challenges:
        data_list = os.path.join('./data/dataset_list', cfg.DATA.DATASET, challenge + '.pkl')
        construct_db(challenge)