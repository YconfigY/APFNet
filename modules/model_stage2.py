# encoding=utf-8
import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

from utils.utils import append_params

'''
   model stage2 is MDNet(backbone)+ (Five challenge branches + SKNet ensemble)  
'''


challenge_list = ['FM', 'SC', 'OCC', 'ILL', 'TC']


class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet, self).__init__()
        self.K = K
        # backbone
        self.layers_v, self.layers_i = [nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2), 
                nn.ReLU(inplace=True), 
                nn.LocalResponseNorm(2),
                nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv2', nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=5, stride=2),
                nn.ReLU(inplace=True), nn.LocalResponseNorm(2),
                nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv3', nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1), 
                nn.ReLU()))])) for _ in range(2)]
        
        self.fc = nn.Sequential(OrderedDict([
            ('fc4', nn.Sequential(
                nn.Linear(512 * 3 * 3 * 2, 512), 
                nn.ReLU(inplace=True))),
            
            ('fc5', nn.Sequential(
                nn.Dropout(0.5), 
                nn.Linear(512, 512), 
                nn.ReLU(inplace=True)))]))
        
        # the first branch to fuse
        self.parallel1 = nn.ModuleList([nn.Sequential(OrderedDict([  # 0:FM 1:OCC 2:SC 3:TC 4:ILL
            ('parallel1_conv1', nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2), nn.ReLU())),
            ('parallel1_conv2', nn.Sequential(
                nn.Conv2d(32, 96, kernel_size=4, stride=2)))])) for _ in range(5)])

        self.parallel2 = nn.ModuleList([nn.Sequential(OrderedDict([
            ('parallel2_conv1', nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=3, stride=2), 
                nn.MaxPool2d(kernel_size=8, stride=1)))])) for _ in range(5)])
        
        self.parallel3 = nn.ModuleList([nn.Sequential(OrderedDict([
            ('parallel3_conv1', nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=1), 
                nn.MaxPool2d(kernel_size=3, stride=1)))])) for _ in range(5)])
        
        self.parallel1_skconv, self.parallel2_skconv, self.parallel3_skconv = \
            [nn.ModuleList([nn.Sequential(OrderedDict([
            (f'parallel{_layer_indx+1}_skconv_global_pool', nn.AdaptiveAvgPool2d(1)),
            (f'parallel{_layer_indx+1}_skconv_fc1', nn.Sequential(
                nn.Conv2d(_channel_b, _channel_a, 1, bias=False),
                nn.ReLU(inplace=True))),
            (f'parallel{_layer_indx+1}_skconv_fc2', nn.Sequential(
                nn.Conv2d(_channel_a, _channel_b * 2, 1, 1, bias=False)))])) for _ in range(5)])
             for _layer_indx, (_channel_a, _channel_b) in enumerate([[32,96],[32,256],[64,512]])]
        
        # filter the five challenge information
        self.ensemble1_skconv, self.ensemble2_skconv, self.ensemble3_skconv = [
            nn.Sequential(OrderedDict([
                (f'ensemble{_layer_indx+1}_skconv_global_pool', nn.AdaptiveAvgPool2d(1)),
                (f'ensemble{_layer_indx+1}_skconv_fc1', nn.Sequential(
                    nn.Conv2d(_channel_b, _channel_a * 5, 1, bias=False), 
                    nn.ReLU(inplace=True))),
                (f'ensemble{_layer_indx+1}_skconv_fc2', nn.Sequential(
                    nn.Conv2d(_channel_a * 5, _channel_b * 5, 1, 1, bias=False)))]))
            for _layer_indx, (_channel_a, _channel_b) in enumerate([[32,96],[64,256],[128,512]])]

        # fc6
        self.branches = nn.ModuleList([nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, 2)) for _ in range(K)])

        self.blockes = [self.parallel1, self.parallel2, self.parallel3,
                        self.parallel1_skconv, self.parallel2_skconv, self.parallel3_skconv,
                        self.ensemble1_skconv, self.ensemble2_skconv, self.ensemble3_skconv]

        # initial parameters
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
        for m in self.branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        # add new branch
        for i, block in enumerate(self.blockes):
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if i < 3:
                        nn.init.constant_(m.bias, 0.1)
        # end the model and load the parameters
        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError('Unkown model format: {:s}'.format(model_path))
        self.build_param_dict()


    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers_v.named_children():
            append_params(self.params, module, 'layers_v' + name)
        for name, module in self.layers_i.named_children():
            append_params(self.params, module, 'layers_i' + name)
        for name, module in self.fc.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_%d' % (k))
        # add new architecher
        for i, block in enumerate(self.blockes):
            for name, module in block.named_children():
                if i > 2 and 'pool' in name:
                    continue
                append_params(self.params, module, name)


    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False


    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        print('get_learnable_params', params.keys())
        return params


    def get_all_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            params[k] = p
        return params


    def forward(self, img_v, img_i, k=0, in_layer='conv1', out_layer='fc6'):
        # forward model from in_layer to out_layer
        run = False
        img = img_v
        parallel_dict = {'conv1': self.parallel1,
                         'conv2': self.parallel2,
                         'conv3': self.parallel3}
        parallel_skconv_dict = {'conv1': self.parallel1_skconv,
                                'conv2': self.parallel2_skconv,
                                'conv3': self.parallel3_skconv}
        ensemble_skconv_dict = {'conv1': self.ensemble1_skconv,
                                'conv2': self.ensemble2_skconv,
                                'conv3': self.ensemble3_skconv}
        channel_dict = {'conv1': 96, 'conv2': 256, 'conv3': 512}
        for (name_v, module_v), (name_i, module_i) in zip(
            self.layers_v.named_children(),self.layers_i.named_children()):
            if name_v == in_layer:
                run = True
            if run:
                if name_v in ['conv1', 'conv2', 'conv3']:
                    V_out =[]
                    for i in range(5):  # 0:FM 1:OCC 2:SC 3:TC 4:ILL
                        output = []
                        img_v_parallel = parallel_dict[name_v][i](img_v)  #107*107->25*25->5*5->3*3
                        img_i_parallel = parallel_dict[name_v][i](img_i)
                        batch_size = img_v.size(0)
                        output.append(img_v_parallel)
                        output.append(img_i_parallel)  # [[N,96/256/512,25,25],[N,96/256/512,25,25]]
                        # Operation of element-wise addition for output
                        U = reduce(lambda x, y: x + y, output)  # [N,96/256/512,25,25]
                        a_b = parallel_skconv_dict[name_v][i](U)
                        a_b = a_b.reshape(batch_size, 2, channel_dict[name_v], -1)  # [N,(96/256/512)*2,1,1]->[N,2,96/256/512,1]
                        a_b = nn.Softmax(dim=1)(a_b)
                        # Splits a_b into 2 chunks, [N,2,96/256/512,1]->[[N,1,96/256/512,1], [N,1,96/256/512,1]]
                        a_b = list(a_b.chunk(2, dim=1))
                        a_b = list(map(lambda x: x.reshape(
                            batch_size, channel_dict[name_v], 1, 1), a_b))  # [N,1,96/256/512,1]->[N,96/256/512,1,1]
                        # Operation of element-wise multiplication between output and a_b
                        V = list(map(lambda x, y: x * y, output, a_b))
                        # operation of element-wise addition for V
                        V = reduce(lambda x, y: x + y, V)  # [N,96/256/512,25/5/3,25/5/3]
                        V_out.append(V)
                    # input to ensemble for x1: ALL1
                    batch_size = img_v.size(0)
                    U = reduce(lambda x, y: x + y, V_out)
                    a_b = ensemble_skconv_dict[name_v](U)
                    a_b = a_b.reshape(batch_size, 5, channel_dict[name_v], -1)
                    a_b = nn.Softmax(dim=1)(a_b)
                    a_b = list(a_b.chunk(5, dim=1))
                    a_b = list(map(lambda x: x.reshape(batch_size, channel_dict[name_v], 1, 1), a_b))
                    V = list(map(lambda x, y: x * y, V_out, a_b))
                    V = reduce(lambda x, y: x + y, V)
                    torch.cuda.empty_cache()

                img_v = module_v(img_v)
                img_i = module_i(img_i)
                img_v = img_v + V
                img_i = img_i + V
                
                if name_v == 'conv3':
                    img = torch.cat((img_v, img_i), 1)
                    img = img.contiguous().view(img.size(0), -1)
                if name_v == out_layer:
                    return img
        img = self.fc(img)
        img = self.branches[k](img)
        if out_layer == 'fc6':
            return img
        elif out_layer == 'fc6_softmax':
            return F.softmax(img, dim=1)


    def load_model(self, model_path):
        states = torch.load(model_path)
        print('load LanYangYang model.')
        self.layers_v.load_state_dict(states['layers_v'])
        self.layers_i.load_state_dict(states['layers_i'])
        self.fc.load_state_dict(states['fc'])
        
        dataset_name = 'GTOT'

        for i in range(len(challenge_list)):
            state_dict_path = './models/' + dataset_name + '_' + challenge_list[i] + '.pth'
            # load parallele1 branches
            self.parallel1[i].load_state_dict(torch.load(state_dict_path)['parallel1'])
            self.parallel1_skconv[i].load_state_dict(torch.load(state_dict_path)['parallel1_skconv'])
            # load parallele2 branches
            self.parallel2[i].load_state_dict(torch.load(state_dict_path)['parallel2'])
            self.parallel2_skconv[i].load_state_dict(torch.load(state_dict_path)['parallel2_skconv'])
            # load parallele3 branches
            self.parallel3[i].load_state_dict(torch.load(state_dict_path)['parallel3'])
            self.parallel3_skconv[i].load_state_dict(torch.load(state_dict_path)['parallel3_skconv'])
            
        print('load LanYangYang model end.')


    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.layers_v[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers_v[i][0].bias.data = torch.from_numpy(bias[:, 0])
            self.layers_i[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers_i[i][0].bias.data = torch.from_numpy(bias[:, 0])