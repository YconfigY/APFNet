# encoding=utf-8
import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce

from utils.utils import append_params

'''
   Model stage1 is MDNet(RGBT)+each challenge branch 
   We need add each attribute branch one by one 
'''        


class MDNet(nn.Module):
    def __init__(self, model_path=None, num_branches=1):
        """_summary_

        Args:
            model_path (_type_, optional): _description_. Defaults to None.
            num_branches (int, optional): _description_. Defaults to 1.

        Raises:
            RuntimeError: _description_
        """
        super(MDNet, self).__init__()
        self.num_branches = num_branches
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

        self.parallel1 = nn.Sequential(OrderedDict([
            # feature extracter
            ('parallel1_conv1', nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2), nn.ReLU(),)),  # 5*5 kernel  
            # 107*107 -> 52*52,(107-5)/2+1=52
            ('parallel1_conv2', nn.Sequential(
                nn.Conv2d(32, 96, kernel_size=4, stride=2)))]))  # 4*4 kernel
            # 52*52 -> 25*25,(52-4)/2+1=25

        self.parallel2 = nn.Sequential(OrderedDict([
            ('parallel2_conv1', nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=3, stride=2),  # 25->12,(25-3)/2+1=12
                nn.MaxPool2d(kernel_size=8, stride=1)))]))  # 12->5,(12-8)/1+1=->5

        self.parallel3 = nn.Sequential(OrderedDict([
            ('parallel3_conv1', nn.Sequential(
                 nn.Conv2d(256, 512, kernel_size=1, stride=1),  #5,(5-1)/1+1=5
                 nn.MaxPool2d(kernel_size=3, stride=1)))  # 5->3,(5-3)/1+1=3
            ]))
        
        self.parallel1_skconv, self.parallel2_skconv, self.parallel3_skconv = \
            [nn.ModuleList([nn.Sequential(OrderedDict([
            (f'parallel{_layer_indx+1}_skconv_global_pool', nn.AdaptiveAvgPool2d(1)),  # output=1
            (f'parallel{_layer_indx+1}_skconv_fc1', nn.Sequential(
                nn.Conv2d(_channel_b, _channel_a, 1, bias=False),
                nn.ReLU(inplace=True))),
            (f'parallel{_layer_indx+1}_skconv_fc2', nn.Sequential(
                nn.Conv2d(_channel_a, _channel_b * 2, 1, 1, bias=False)))])) for _ in range(5)])
             for _layer_indx, (_channel_a, _channel_b) in enumerate([[32,96],[32,256],[64,512]])]
        
        
        self.paralle_blockes = [self.parallel1, self.parallel2, self.parallel3,
                               self.parallel1_skconv, self.parallel2_skconv, self.parallel3_skconv]

        self.branches = nn.ModuleList([nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, 2)) for _ in range(num_branches)])

        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
        for m in self.branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        for i, paralle_block in enumerate(self.paralle_blockes):
            for m in paralle_block.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if i < 3:
                        nn.init.constant_(m.bias, 0.1)

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
        
        for i, paralle_block in enumerate(self.paralle_blockes):
            for name, module in paralle_block.named_children():
                if i > 2 and 'pool' in name:
                    continue
                append_params(self.params, module, name)


    def set_learnable_params(self, layers):
        """Set learnable params in selected layers before training

        Args:
            layers (_type_): selected layers that usually belong to backbone
        """
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False


    def get_learnable_params(self):
        """Get learnable params in every layer before training

        Returns:
            params (list[]): learnable params in every layer
        """
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
        """Forward model from in_layer to out_layer

        Args:
            img_v (Tensor): Visible image
            img_i (Tensor): Infrared image
            k (int, optional): Branch number. Defaults to 0.
            in_layer (str, optional): _description_. Defaults to 'conv1'.
            out_layer (str, optional): _description_. Defaults to 'fc6'.

        Returns:
            score (Tensor): The socre of visible and infrared images
        """
        run = False
        img = img_v
        for (name_v, module_v), (name_i, module_i) in zip(
            self.layers_v.named_children(), self.layers_i.named_children()):
            output = []
            if name_v == in_layer:
                run = True
            if run and name_v in ['conv1', 'conv2', 'conv3']:
                parallel_dict = {'conv1': self.parallel1,
                                 'conv2': self.parallel2,
                                 'conv3': self.parallel3}
                parallel_skconv_dict = {'conv1': self.parallel1_skconv,
                                        'conv2': self.parallel2_skconv,
                                        'conv3': self.parallel3_skconv}
                channel_dict = {'conv1': 96, 'conv2': 256, 'conv3': 512}
                img_v_parallel = parallel_dict[name_v](img_v)  #107*107->25*25->5*5->3*3
                img_i_parallel = parallel_dict[name_v](img_i)
                batch_size = img_v.size(0)
                output.append(img_v_parallel)
                output.append(img_i_parallel)  # [[N,96/256/512,25,25],[N,96/256/512,25,25]]
                # Operation of element-wise addition for output
                U = reduce(lambda x, y: x + y, output)  # [N,96/256/512,25,25]
                a_b = parallel_skconv_dict[name_v](U)
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

                img_v = module_v(img_v)  # VGG-M, [N,3,107,107]->[N,96,25,25]->[N,256,5,5]->[N,512,3,3]
                img_v = img_v + V
                img_i = module_i(img_i)
                img_i = img_i + V
                if name_v == 'conv3':
                    x = torch.cat((img_v, img_i), 1)  # [N,512,3,3]->[N,1024,3,3]
                    x = x.contiguous().view(x.size(0), -1)  # [N,1024,3,3]->[N,1024*3*3]
                if name_v == out_layer:
                    return x
        x = self.fc(x)  # [N, 512*3*3*2]->[N,512]
        x = self.branches[k](x)  # kth branch, [N,512]->[N,2]
        if out_layer == 'fc6':
            return x
        elif out_layer == 'fc6_softmax':
            return F.softmax(x, dim=1)


    def load_model(self, model_path):
        states = torch.load(model_path)
        try:
            print('load LanYangYang model.')
            self.layers_v.load_state_dict(states['layers_v'])
            self.layers_i.load_state_dict(states['layers_i'])
            self.fc.load_state_dict(states['fc'])
            print('load LanYangYang model end.')
        except:
            print('load LanYangYang model error!')
            print('load VID model.')
            shared_layers = states['shared_layers']
            pretrain_parm = OrderedDict()
            pretrain_parm['layers_v'] = OrderedDict()
            pretrain_parm['layers_i'] = OrderedDict()
            pretrain_parm['fc'] = OrderedDict()
            for k, v in shared_layers.items():
                if 'conv' in k:
                    pretrain_parm['layers_v'][k] = v
                    pretrain_parm['layers_i'][k] = v
                elif k == 'fc4.0.weight':
                    pretrain_parm['fc'][k] = torch.cat((v, v), 1)
                else:
                    pretrain_parm['fc'][k] = v
            self.layers_v.load_state_dict(pretrain_parm['layers_v'])
            self.layers_i.load_state_dict(pretrain_parm['layers_i'])
            self.fc.load_state_dict(pretrain_parm['fc'])
            print('load VID model end.')


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
