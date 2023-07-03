import os
import numpy as np
import torch
import scipy.io

from tqdm import tqdm

from utils.config import cfg


def load_model(model):
    pretrained_model, resume, stage = cfg.MODEL.PRETRAINED, cfg.MODEL.RESUME, cfg.MODEL.STAGE_TYPE
    if stage <= 3:
        # try:
        # checkpoint = torch.load(pretrained_model)
        # if 'state_dict' in checkpoint:
        #     checkpoint = checkpoint['state_dict']
        # ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        # for _conv, _stage in (['conv1','stage0'],['conv2','stage1'],['conv3','stage2'],['conv4','stage3']):
        #     model.layers_v[_conv].load_state_dict(ckpt[_stage])
        #     model.layers_i[_conv].load_state_dict(ckpt[_stage])
        print('Load RepVGG A2 pretrained model successfully!')
        # except:
            # raise RuntimeError('Unknown model format: {:s}'.format(pretrained_model))
    if resume is not None:
        resume_file = os.path.join('./resume/stage_' + stage, cfg.DATA.TRAIN_DATASET, cfg.TRAIN.CHALLENGE, 
                                   cfg.TRAIN.CHALLENGE + '_' + str(cfg.TRAIN.START_EPOCH) + '.pth')
        try:
            states = torch.load(resume_file)
            for _layer_type in model.blockes.values():
                for _layer_name, _layer in _layer_type.items():
                    _layer.load_state_dict(states[_layer_name]) # parallel1(2/3)_skconv...
            print('Resume stage_{:d} model successfully.'.format(stage))
        except:
            raise RuntimeError('Resume stage_{:d} model {:s} error!'.format(stage, resume_file))
    else:
        if stage == 2:
            try:
                for i, challenge in tqdm(enumerate(cfg.DATA.CHALLENGE), desc='Loading stage1 trained model',
                                            total=len(cfg.DATA.CHALLENGE), leave=True, ncols=100):
                    trained_model_file = os.path.join('./resume/stage_1', cfg.DATA.TRAIN_DATASET, challenge, 
                                                        challenge + '.pth')
                    # load parallele1 branches
                    model.parallel1[i].load_state_dict(torch.load(trained_model_file)['parallel1'])
                    model.parallel1_skconv[i].load_state_dict(torch.load(trained_model_file)['parallel1_skconv'])
                    # load parallele2 branches
                    model.parallel2[i].load_state_dict(torch.load(trained_model_file)['parallel2'])
                    model.parallel2_skconv[i].load_state_dict(torch.load(trained_model_file)['parallel2_skconv'])
                    # load parallele3 branches
                    model.parallel3[i].load_state_dict(torch.load(trained_model_file)['parallel3'])
                    model.parallel3_skconv[i].load_state_dict(torch.load(trained_model_file)['parallel3_skconv'])
                print('Load stage1 trained model successfully.')
            except:
                raise RuntimeError('Load stage1 trained model error!')
        elif stage == 3:
            states = torch.load(os.path.join('./resume/stage_2', cfg.DATA.TRAIN_DATASET, 
                                             cfg.TRAIN.CHALLENGE, cfg.TRAIN.CHALLENGE + '.pth'))
            try:
                model.layers_v.load_state_dict(states['layers_v'])
                model.layers_i.load_state_dict(states['layers_i'])
                model.fc.load_state_dict(states['fc'])
                for _layer_type in model.blockes.values():
                    for _layer_name, _layer in _layer_type.items():
                        _layer.load_state_dict(states[_layer_name]) # parallel1(2/3)_skconv...
                print('Loading stage2 trained model successfully.')
            except:
                raise RuntimeError('Loading stage2 trained model error!')
            
    if stage == 4:
        states = torch.load(os.path.join('./resume/stage_3', cfg.DATA.TRAIN_DATASET, 
                                         cfg.TRAIN.CHALLENGE, cfg.TRAIN.CHALLENGE + '.pth'))
        try:
            print('loading stage3 model.')
            model.layers_v.load_state_dict(states['layers_v'])
            model.layers_i.load_state_dict(states['layers_i'])
            model.fc.load_state_dict(states['fc'])
            for _layer_type in model.blockes.values():
                for _layer_name, _layer in _layer_type.items():
                    _layer.load_state_dict(states[_layer_name]) # parallel1(2/3)_skconv...
            _module_index = 0
            for _module in model.transformer.values():
                _module_index += 1
                _units = ['encoder1', 'encoder2', 'encoder3', 'decoder1', 'decoder2']
                _units_index = 0
                for _layer in _module.values():
                    _layer.load_state_dict(states[f'transformer{_module_index}_{_units[_units_index]}']) # transformer1(2/3)_encoder1(2/3)...
                    _units_index += 1
            print('loading model successfully.')
        except:
            raise RuntimeError('Loading stage3 model error!')
