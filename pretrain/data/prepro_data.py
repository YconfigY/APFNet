# from ast import main
import os
import numpy as np
import pickle
from collections import OrderedDict


def construct_db(dataset_name, seq_list):
    # Construct db
    data = OrderedDict()
    for i, seq_name in enumerate(seq_list):
        
        seq_path_v = os.path.join(dataset_path, seq_name, dataset_dict[dataset_name]['visible'])
        seq_path_i = os.path.join(dataset_path, seq_name, dataset_dict[dataset_name]['infrared'])
        
        # get img list(filename + name extension)
        img_list_v = sorted(
            [p for p in os.listdir(seq_path_v)
             if os.path.splitext(p)[1] in ['.jpg', '.bmp', '.png']])
        img_list_i = sorted(
            [p for p in os.listdir(seq_path_i)
             if os.path.splitext(p)[1] in ['.jpg', '.bmp', '.png']])
        
        # get img abs pth list(path + filename + name extension)
        img_list_v = [os.path.join(seq_path_v, img) for img in img_list_v]
        img_list_i = [os.path.join(seq_path_i, img) for img in img_list_i]
        
        # load seq gt list from txt file
        if dataset_name == 'GTOT':
            gt = np.loadtxt(dataset_path, seq_name + '/init.txt')
        elif dataset_name == 'RGBT234':
            gt = np.loadtxt(dataset_path, seq_name + '/init.txt', delimiter=',')
            
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
                
    return data, fp

if __name__ == "__main__":
    # Save db
    dataset_dir = ''
    dataset_dict = {'GTOT':{'visible': 'v','infrared':'i'}, 
                    'RGBT234':{'visible': 'visible','infrared':'infrared'}}
    challenge_dict = {'FM': 'fast_motion', 'SC': 'size_change', 
                      'OCC': 'occlusion', 'ILL': 'illum_change', 'TC': 'thermal_crossover'}
    
    for dataset_name in dataset_dict:
        dataset_path = os.path.join(dataset_dir, dataset_name)
        # FM~Fast Motion; 
        # OCC~Occlusion; 
        # SC~Scale Variation; 
        # ILL~Illumination Variation; 
        # TC~Thermal Crossover; 
        # ALL~Whole dataset
        challenge_type = 'ALL'  # set challenge type. 
        
        seq_list_flie = os.path.join(dataset_path, dataset_name) + '.txt'  # modify it to yourself
        data_list = './pretrain/data/' + dataset_name + '_' + challenge_type + '.pkl'  # modify it to yourself
        with open(seq_list_flie, 'r') as fp:
            seq_list = fp.read().splitlines()
            
        data, fp = construct_db(dataset_name, seq_list)
        
        with open(data_list, 'wb') as fp:
            print('data_list', data_list)  # modify py3 to py2
            pickle.dump(data, fp)
