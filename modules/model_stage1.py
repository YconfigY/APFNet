# encoding=utf-8
import os
from joblib import parallel_backend
import scipy.io
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce

'''
   Model stage1 is MDNet(RGBT)+each challenge branch 
   We need add each attribute branch one by one 
'''


def append_params(params, module, prefix):
    for child in module.children():
        for k, p in child._parameters.items():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError('Duplicated param name: {:s}'.format(name))


def set_optimizer(model, lr_base, lr_mult, train_all=False, momentum=0.9, w_decay=0.0005):
    if train_all:
        params = model.get_all_params()
    else:
        params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


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
        self.layers_v = nn.Sequential(OrderedDict([
            ('conv1',nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2), 
                nn.ReLU(inplace=True), 
                nn.LocalResponseNorm(2),
                nn.MaxPool2d(kernel_size=3, stride=2))),
            
            ('conv2',nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=5, stride=2), 
                nn.ReLU(inplace=True), 
                nn.LocalResponseNorm(2),
                nn.MaxPool2d(kernel_size=3, stride=2))),
            
            ('conv3', nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1), 
                nn.ReLU()))
            ]))
        
        self.layers_i = nn.Sequential(OrderedDict([
            ('conv1',nn.Sequential(
                 nn.Conv2d(3, 96, kernel_size=7, stride=2), 
                 nn.ReLU(inplace=True), 
                 nn.LocalResponseNorm(2),
                 nn.MaxPool2d(kernel_size=3, stride=2))),
            
            ('conv2',nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=5, stride=2), 
                nn.ReLU(inplace=True), 
                nn.LocalResponseNorm(2),
                nn.MaxPool2d(kernel_size=3, stride=2))),
            
            ('conv3', nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1), 
                nn.ReLU()))
            ]))
        
        self.fc = nn.Sequential(OrderedDict([
            ('fc4', nn.Sequential(
                nn.Linear(512 * 3 * 3 * 2, 512), 
                nn.ReLU(inplace=True))),
            
            ('fc5', nn.Sequential(
                nn.Dropout(0.5), 
                nn.Linear(512, 512), 
                nn.ReLU(inplace=True)))]))

        self.parallel1 = nn.Sequential(OrderedDict([
            ('parallel1_conv1', nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2), 
                nn.ReLU(), )),  # 5*5 kernel
            
            ('parallel1_conv2', nn.Sequential(
                nn.Conv2d(32, 96, kernel_size=4, stride=2)))]))  # 4*4 kernel

        self.paralle1_skconv = nn.Sequential(OrderedDict([
            ('parallel1_skconv_global_pool', nn.AdaptiveAvgPool2d(1)),
            
            ('parallel1_skconv_fc1', nn.Sequential(
                nn.Conv2d(96, 32, 1, bias=False),
                nn.ReLU(inplace=True))),
            
            ('parallel1_skconv_fc2', nn.Sequential(
                nn.Conv2d(32, 96 * 2, 1, 1, bias=False)))]))

        self.parallel2 = nn.Sequential(OrderedDict([
            ('parallel2_conv1', nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=8, stride=1)))]))

        self.paralle2_skconv = nn.Sequential(OrderedDict([
            ('parallel2_skconv_global_pool', nn.AdaptiveAvgPool2d(1)),
            
            ('parallel2_skconv_fc1', nn.Sequential(
                nn.Conv2d(256, 32, 1, bias=False),
                nn.ReLU(inplace=True))),
            ('parallel2_skconv_fc2', nn.Sequential(
                nn.Conv2d(32, 256 * 2, 1, 1, bias=False)))]))

        self.parallel3 = nn.Sequential(OrderedDict([
            ('parallel3_conv1', nn.Sequential(
                 nn.Conv2d(256, 512, kernel_size=1, stride=1), 
                 nn.MaxPool2d(kernel_size=3, stride=1)))]))

        self.paralle3_skconv = nn.Sequential(OrderedDict([
            ('parallel3_skconv_global_pool', nn.AdaptiveAvgPool2d(1)),
            
            ('parallel3_skconv_fc1', nn.Sequential(
                nn.Conv2d(512, 64, 1, bias=False),
                nn.ReLU(inplace=True))),
            ('parallel3_skconv_fc2', nn.Sequential(
                nn.Conv2d(64, 512 * 2, 1, 1, bias=False)))]))
        
        self.paralle_layers = [self.parallel1, self.parallel2, self.parallel3,
                               self.paralle1_skconv, self.paralle2_skconv, self.paralle3_skconv]

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

        for i, layer in enumerate(self.paralle_layers):
            for m in layer.modules():
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
        
        for i, layer in enumerate(self.paralle_layers):
            for name, module in layer.named_children():
                if i > 2 and 'pool' in name:
                    continue
                append_params(self.params, module, name)


    def set_learnable_params(self, layers):
        """set learnable params in selected layers before training

        Args:
            layers (_type_): selected layers that usually belong to backbone
        """
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        """get learnable params in every layer before training

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
        # forward model from in_layer to out_layer
        run = False
        img = img_v
        for (name_v, module_v), (name_i, module_i) in zip(self.layers_v.named_children(),
                                                          self.layers_i.named_children()):
            output = []
            if name_v == in_layer:
                run = True
            if run:
                if name_v in ['conv1', 'conv2', 'conv3']:
                    if name_v == 'conv1':
                        img_v_parallel, img_i_parallel = self.parallel1(img_v), self.parallel1(img_i)
                        batch_size = img_v.size(0)
                        output.append(img_v_parallel)
                        output.append(img_i_parallel)
                        # Operation of element-wise addition for output
                        U = reduce(lambda x, y: x + y, output)
                        a_b = self.paralle1_skconv(U)
                        a_b = a_b.reshape(batch_size, 2, 96, -1)  # [N,192,1,1]->[N,2,96,1]
                        a_b = nn.Softmax(dim=1)(a_b)
                        # Splits a_b into 2 chunks, [N,2,96,1]->[[N,1,96,1], [N,1,96,1]]
                        a_b = list(a_b.chunk(2, dim=1))
                        a_b = list(map(lambda x: x.reshape(batch_size, 96, 1, 1), a_b))  # [N,1,96,1]->[N,96,1,1]
                        # Operation of element-wise multiplication between output and a_b
                        V = list(map(lambda x, y: x * y, output, a_b))  
                        V = reduce(lambda x, y: x + y, V)  # operation of element-wise addition for V
                    elif name_v == 'conv2':
                        img_v_paralle, img_i_paralle = self.parallel2(img_v), self.parallel2(img_i)
                        batch_size = img_v.size(0)
                        output.append(img_v_paralle)
                        output.append(img_i_paralle)
                        U = reduce(lambda x, y: x + y, output)
                        a_b = self.paralle2_skconv(U)
                        a_b = a_b.reshape(batch_size, 2, 256, -1)
                        a_b = nn.Softmax(dim=1)(a_b)

                        a_b = list(a_b.chunk(2, dim=1))
                        a_b = list(map(lambda x: x.reshape(batch_size, 256, 1, 1), a_b))
                        V = list(map(lambda x, y: x * y, output, a_b))
                        V = reduce(lambda x, y: x + y, V)
                    else:
                        img_v_paralle, img_i_paralle = self.parallel3(img_v), self.parallel3(img_i)
                        batch_size = img_v.size(0)
                        output.append(img_v_paralle)
                        output.append(img_i_paralle)
                        U = reduce(lambda x, y: x + y, output)
                        a_b = self.paralle3_skconv(U)
                        a_b = a_b.reshape(batch_size, 2, 512, -1)
                        a_b = nn.Softmax(dim=1)(a_b)

                        a_b = list(a_b.chunk(2, dim=1))
                        a_b = list(map(lambda x: x.reshape(batch_size, 512, 1, 1), a_b))
                        V = list(map(lambda x, y: x * y, output, a_b))
                        V = reduce(lambda x, y: x + y, V)
                img_v = module_v(img_v)
                img_v = img_v + V
                img_i = module_i(img_i)
                img_i = img_i + V
                if name_v == 'conv3':
                    x = torch.cat((img_v, img_i), 1)
                    x = x.contiguous().view(x.size(0), -1)
                if name_v == out_layer:
                    return x
        x = self.fc(x)
        x = self.branches[k](x)
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


class BCELoss(nn.Module):
    def forward(self, pos_score, neg_score, average=True):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
        neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]

        loss = pos_loss.sum() + neg_loss.sum()
        if average:
            loss /= (pos_loss.size(0) + neg_loss.size(0))
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float()
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float()
        acc = (pos_correct + neg_correct) / (pos_score.size(0) + neg_score.size(0) + 1e-8)
        return acc.item()


class Precision():
    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)
        return prec.item()
