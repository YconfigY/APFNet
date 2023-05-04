# encoding=utf-8
import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import math


'''
   Complete network structure: Encoder and Decoder were added on a two-phase basis
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


class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet, self).__init__()
        self.K = K
        # backbone
        self.layers_v = nn.Sequential(OrderedDict([
            ('conv1',
             nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2), nn.ReLU(inplace=True), nn.LocalResponseNorm(2),
                           nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv2',
             nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2), nn.ReLU(inplace=True), nn.LocalResponseNorm(2),
                           nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1), nn.ReLU()))]))
        self.layers_i = nn.Sequential(OrderedDict([
            ('conv1',
             nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2), nn.ReLU(inplace=True), nn.LocalResponseNorm(2),
                           nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv2',
             nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2), nn.ReLU(inplace=True), nn.LocalResponseNorm(2),
                           nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1), nn.ReLU()))]))
        self.fc = nn.Sequential(OrderedDict([
            ('fc4', nn.Sequential(nn.Linear(512 * 3 * 3 * 2, 512), nn.ReLU(inplace=True))),
            ('fc5', nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 512), nn.ReLU(inplace=True)))]))

        self.parallel1 = nn.ModuleList([nn.Sequential(OrderedDict([  # 0:FM 1:OCC 2:SC 3:TC 4:ILL
            ('parallel1_conv1', nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, stride=2), nn.ReLU())),
            ('parallel1_conv2', nn.Sequential(nn.Conv2d(32, 96, kernel_size=4, stride=2)))])) for _ in range(5)])

        self.parallel1_skconv = nn.ModuleList([nn.Sequential(OrderedDict([
            ('parallel1_skconv_global_pool', nn.AdaptiveAvgPool2d(1)),
            ('parallel1_skconv_fc1', nn.Sequential(nn.Conv2d(96, 32, 1, bias=False),
                                                   nn.ReLU(inplace=True))),
            ('parallel1_skconv_fc2', nn.Sequential(nn.Conv2d(32, 96 * 2, 1, 1, bias=False)))])) for _ in range(5)])

        self.parallel2 = nn.ModuleList([nn.Sequential(OrderedDict([
            ('parallel2_conv1',
             nn.Sequential(nn.Conv2d(96, 256, kernel_size=3, stride=2), nn.MaxPool2d(kernel_size=8, stride=1)))])) for _
            in range(5)])

        self.parallel2_skconv = nn.ModuleList([nn.Sequential(OrderedDict([
            ('parallel2_skconv_global_pool', nn.AdaptiveAvgPool2d(1)),
            ('parallel2_skconv_fc1', nn.Sequential(nn.Conv2d(256, 32, 1, bias=False),
                                                   nn.ReLU(inplace=True))),
            ('parallel2_skconv_fc2', nn.Sequential(nn.Conv2d(32, 256 * 2, 1, 1, bias=False)))])) for _ in range(5)])
        self.parallel3 = nn.ModuleList([nn.Sequential(OrderedDict([
            ('parallel3_conv1',
             nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=1), nn.MaxPool2d(kernel_size=3, stride=1)))])) for
            _ in range(5)])

        self.parallel3_skconv = nn.ModuleList([nn.Sequential(OrderedDict([
            ('parallel3_skconv_global_pool', nn.AdaptiveAvgPool2d(1)),
            ('parallel3_skconv_fc1', nn.Sequential(nn.Conv2d(512, 64, 1, bias=False),
                                                   nn.ReLU(inplace=True))),
            ('parallel3_skconv_fc2', nn.Sequential(nn.Conv2d(64, 512 * 2, 1, 1, bias=False)))])) for _ in range(5)])

        self.ensemble1_skconv = nn.Sequential(OrderedDict([
            ('ensemble1_skconv_global_pool', nn.AdaptiveAvgPool2d(1)),
            ('ensemble1_skconv_fc1', nn.Sequential(nn.Conv2d(96, 32 * 5, 1, bias=False),
                                                   nn.ReLU(inplace=True))),
            ('ensemble1_skconv_fc2', nn.Sequential(nn.Conv2d(32 * 5, 96 * 5, 1, 1, bias=False)))]))

        self.ensemble2_skconv = nn.Sequential(OrderedDict([
            ('ensemble2_skconv_global_pool', nn.AdaptiveAvgPool2d(1)),
            ('ensemble2_skconv_fc1', nn.Sequential(nn.Conv2d(256, 64 * 5, 1, bias=False),
                                                   nn.ReLU(inplace=True))),
            ('ensemble2_skconv_fc2', nn.Sequential(nn.Conv2d(64 * 5, 256 * 5, 1, 1, bias=False)))]))

        self.ensemble3_skconv = nn.Sequential(OrderedDict([
            ('ensemble3_skconv_global_pool', nn.AdaptiveAvgPool2d(1)),
            ('ensemble3_skconv_fc1', nn.Sequential(nn.Conv2d(512, 128 * 5, 1, bias=False),
                                                   nn.ReLU(inplace=True))),
            ('ensemble3_skconv_fc2', nn.Sequential(nn.Conv2d(128 * 5, 512 * 5, 1, 1, bias=False)))]))

        # We add Encoders and Decoders here.
        # a layer has there encoders and decoders.
        #####################################
        self.transformer1_encoder1 = nn.Sequential(OrderedDict([
            ('transformer1_encoder1_WK', nn.Sequential(nn.Linear(32, 32))),
            ('transformer1_encoder1_WV', nn.Sequential(nn.Linear(32, 32))),
            ('transformer1_encoder1_fc_reduce', nn.Sequential(nn.Conv2d(96, 32, 1, 1, bias=False))),
            ('transformer1_encoder1_fc_rise', nn.Sequential(nn.Conv2d(32, 96, 1)))
        ]))
        self.transformer1_encoder2 = nn.Sequential(OrderedDict([
            ('transformer1_encoder2_WK', nn.Sequential(nn.Linear(32, 32))),
            ('transformer1_encoder2_WV', nn.Sequential(nn.Linear(32, 32))),
            ('transformer1_encoder2_fc_reduce', nn.Sequential(nn.Conv2d(96, 32, 1, 1, bias=False))),
            ('transformer1_encoder2_fc_rise', nn.Sequential(nn.Conv2d(32, 96, 1)))
        ]))
        self.transformer1_encoder3 = nn.Sequential(OrderedDict([
            ('transformer1_encoder3_WK', nn.Sequential(nn.Linear(32, 32))),
            ('transformer1_encoder3_WV', nn.Sequential(nn.Linear(32, 32))),
            ('transformer1_encoder3_fc_reduce', nn.Sequential(nn.Conv2d(96, 32, 1, 1, bias=False))),
            ('transformer1_encoder3_fc_rise', nn.Sequential(nn.Conv2d(32, 96, 1)))
        ]))
        ###################################################
        self.transformer1_decoder1 = nn.Sequential(OrderedDict([
            ('transformer1_decoder1_WK', nn.Sequential(nn.Linear(32, 32))),
            ('transformer1_decoder1_WV', nn.Sequential(nn.Linear(32, 32))),
            ('transformer1_decoder1_fc_reduce', nn.Sequential(nn.Conv2d(96, 32, 1, 1, bias=False))),
            ('transformer1_decoder1_fc_rise', nn.Sequential(nn.Conv2d(32, 96, 1)))
        ]))
        self.transformer1_decoder2 = nn.Sequential(OrderedDict([
            ('transformer1_decoder2_WK', nn.Sequential(nn.Linear(32, 32))),
            ('transformer1_decoder2_WV', nn.Sequential(nn.Linear(32, 32))),
            ('transformer1_decoder2_fc_reduce', nn.Sequential(nn.Conv2d(96, 32, 1, 1, bias=False))),
            ('transformer1_decoder2_fc_rise', nn.Sequential(nn.Conv2d(32, 96, 1)))
        ]))
        ############################################################

        ######################################
        self.transformer2_encoder1 = nn.Sequential(OrderedDict([
            ('transformer2_encoder1_WK', nn.Sequential(nn.Linear(64, 64))),
            ('transformer2_encoder1_WV', nn.Sequential(nn.Linear(64, 64))),
            ('transformer2_encoder1_fc_reduce', nn.Sequential(nn.Conv2d(256, 64, 1, 1, bias=False))),
            ('transformer2_encoder1_fc_rise', nn.Sequential(nn.Conv2d(64, 256, 1)))
        ]))
        self.transformer2_encoder2 = nn.Sequential(OrderedDict([
            ('transformer2_encoder2_WK', nn.Sequential(nn.Linear(64, 64))),
            ('transformer2_encoder2_WV', nn.Sequential(nn.Linear(64, 64))),
            ('transformer2_encoder2_fc_reduce', nn.Sequential(nn.Conv2d(256, 64, 1, 1, bias=False))),
            ('transformer2_encoder2_fc_rise', nn.Sequential(nn.Conv2d(64, 256, 1)))
        ]))
        self.transformer2_encoder3 = nn.Sequential(OrderedDict([
            ('transformer2_encoder3_WK', nn.Sequential(nn.Linear(64, 64))),
            ('transformer2_encoder3_WV', nn.Sequential(nn.Linear(64, 64))),
            ('transformer2_encoder3_fc_reduce', nn.Sequential(nn.Conv2d(256, 64, 1, 1, bias=False))),
            ('transformer2_encoder3_fc_rise', nn.Sequential(nn.Conv2d(64, 256, 1)))
        ]))

        self.transformer2_decoder1 = nn.Sequential(OrderedDict([
            ('transformer2_decoder1_WK', nn.Sequential(nn.Linear(64, 64))),
            ('transformer2_decoder1_WV', nn.Sequential(nn.Linear(64, 64))),
            ('transformer2_decoder1_fc_reduce', nn.Sequential(nn.Conv2d(256, 64, 1, 1, bias=False))),
            ('transformer2_decoder1_fc_rise', nn.Sequential(nn.Conv2d(64, 256, 1)))
        ]))
        self.transformer2_decoder2 = nn.Sequential(OrderedDict([
            ('transformer2_decoder2_WK', nn.Sequential(nn.Linear(64, 64))),
            ('transformer2_decoder2_WV', nn.Sequential(nn.Linear(64, 64))),
            ('transformer2_decoder2_fc_reduce', nn.Sequential(nn.Conv2d(256, 64, 1, 1, bias=False))),
            ('transformer2_decoder2_fc_rise', nn.Sequential(nn.Conv2d(64, 256, 1)))
        ]))

        ################################################
        self.transformer3_encoder1 = nn.Sequential(OrderedDict([
            ('transformer3_encoder1_WK', nn.Sequential(nn.Linear(128, 128))),
            ('transformer3_encoder1_WV', nn.Sequential(nn.Linear(128, 128))),
            ('transformer3_encoder1_fc_reduce', nn.Sequential(nn.Conv2d(512, 128, 1, 1, bias=False))),
            ('transformer3_encoder1_fc_rise', nn.Sequential(nn.Conv2d(128, 512, 1)))
        ]))
        self.transformer3_encoder2 = nn.Sequential(OrderedDict([
            ('transformer3_encoder2_WK', nn.Sequential(nn.Linear(128, 128))),
            ('transformer3_encoder2_WV', nn.Sequential(nn.Linear(128, 128))),
            ('transformer3_encoder2_fc_reduce', nn.Sequential(nn.Conv2d(512, 128, 1, 1, bias=False))),
            ('transformer3_encoder2_fc_rise', nn.Sequential(nn.Conv2d(128, 512, 1)))
        ]))
        self.transformer3_encoder3 = nn.Sequential(OrderedDict([
            ('transformer3_encoder3_WK', nn.Sequential(nn.Linear(128, 128))),
            ('transformer3_encoder3_WV', nn.Sequential(nn.Linear(128, 128))),
            ('transformer3_encoder3_fc_reduce', nn.Sequential(nn.Conv2d(512, 128, 1, 1, bias=False))),
            ('transformer3_encoder3_fc_rise', nn.Sequential(nn.Conv2d(128, 512, 1)))
        ]))

        self.transformer3_decoder1 = nn.Sequential(OrderedDict([
            ('transformer3_decoder1_WK', nn.Sequential(nn.Linear(128, 128))),
            ('transformer3_decoder1_WV', nn.Sequential(nn.Linear(128, 128))),
            ('transformer3_decoder1_fc_reduce', nn.Sequential(nn.Conv2d(512, 128, 1, 1, bias=False))),
            ('transformer3_decoder1_fc_rise', nn.Sequential(nn.Conv2d(128, 512, 1)))
        ]))
        self.transformer3_decoder2 = nn.Sequential(OrderedDict([
            ('transformer3_decoder2_WK', nn.Sequential(nn.Linear(128, 128))),
            ('transformer3_decoder2_WV', nn.Sequential(nn.Linear(128, 128))),
            ('transformer3_decoder2_fc_reduce', nn.Sequential(nn.Conv2d(512, 128, 1, 1, bias=False))),
            ('transformer3_decoder2_fc_rise', nn.Sequential(nn.Conv2d(128, 512, 1)))
        ]))

        # multi branch
        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 2)) for _ in range(K)])


        self.Transformer_Encoder = [[self.transformer1_encoder1, 
                                     self.transformer1_encoder2, 
                                     self.transformer1_encoder3],
                                    [self.transformer2_encoder1, 
                                     self.transformer2_encoder2, 
                                     self.transformer2_encoder3],
                                    [self.transformer3_encoder1, 
                                     self.transformer3_encoder2, 
                                     self.transformer3_encoder3]]

        self.Transformer_Decoder = [[self.transformer1_decoder1, self.transformer1_decoder2],
                                    [self.transformer2_decoder1, self.transformer2_decoder2]]

        for m in self.branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        for m in self.transformer1_encoder1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.transformer1_encoder2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.transformer1_encoder3.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.transformer1_decoder1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.transformer1_decoder2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()
        # for m in self.transformer1_FFN.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #     if isinstance(m, nn.Linear):
        #         m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
        #         if m.bias is not None:
        #            m.bias.data.zero_()

        # initialize the second layer
        for m in self.transformer2_encoder1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.transformer2_encoder2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.transformer2_encoder3.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.transformer2_decoder1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.transformer2_decoder2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()
        # for m in self.transformer2_FFN.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #     if isinstance(m, nn.Linear):
        #         m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
        #         if m.bias is not None:
        #            m.bias.data.zero_()

        # initialize the third layer
        for m in self.transformer3_encoder1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.transformer3_encoder2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.transformer3_encoder3.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()

        for m in self.transformer3_decoder1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.transformer3_decoder2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    m.bias.data.zero_()
        # for m in self.transformer3_FFN.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #     if isinstance(m, nn.Linear):
        #         m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
        #         if m.bias is not None:
        #            m.bias.data.zero_()

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

        for k, module in enumerate(self.parallel1):
            for name, model in module.named_children():
                if 'pool' in name:
                    continue
                append_params(self.params, model, name + '_%d' % (k))
        for k, module in enumerate(self.parallel2):
            for name, model in module.named_children():
                if 'pool' in name:
                    continue
                append_params(self.params, model, name + '_%d' % (k))
        for k, module in enumerate(self.parallel3):
            for name, model in module.named_children():
                if 'pool' in name:
                    continue
                append_params(self.params, model, name + '_%d' % (k))

        for k, module in enumerate(self.parallel1_skconv):
            for name, model in module.named_children():
                if 'pool' in name:
                    continue
                append_params(self.params, model, name + '_%d' % (k))

        for k, module in enumerate(self.parallel2_skconv):
            for name, model in module.named_children():
                if 'pool' in name:
                    continue
                append_params(self.params, model, name + '_%d' % (k))

        for k, module in enumerate(self.parallel3_skconv):
            for name, model in module.named_children():
                if 'pool' in name:
                    continue
                append_params(self.params, model, name + '_%d' % (k))
        #############################################################
        # the last ALL modeles
        for name, module in self.ensemble1_skconv.named_children():
            if 'pool' in name:
                continue
            append_params(self.params, module, name)
        for name, module in self.ensemble2_skconv.named_children():
            if 'pool' in name:
                continue
            append_params(self.params, module, name)
        for name, module in self.ensemble3_skconv.named_children():
            if 'pool' in name:
                continue
            append_params(self.params, module, name)

        # add Transformer
        for name, module in self.transformer1_encoder1.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer1_encoder2.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer1_encoder3.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer1_decoder1.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer1_decoder2.named_children():
            append_params(self.params, module, name)
        # for name, module in self.transformer1_FFN.named_children():
        #     append_params(self.params, module, name)
        for name, module in self.transformer2_encoder1.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer2_encoder2.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer2_encoder3.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer2_decoder1.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer2_decoder2.named_children():
            append_params(self.params, module, name)
        # for name, module in self.transformer2_FFN.named_children():
        #     append_params(self.params, module, name)
        for name, module in self.transformer3_encoder1.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer3_encoder2.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer3_encoder3.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer3_decoder1.named_children():
            append_params(self.params, module, name)
        for name, module in self.transformer3_decoder2.named_children():
            append_params(self.params, module, name)
        # for name, module in self.transformer3_FFN.named_children():
        #     append_params(self.params, module, name)

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


    # self-atttention
    def Self_Attention(self, layer_index, type_index, x):
        # layer_index = 0(layer1)、1(layer2)、2(layer3)
        # layer_index : vis:0  inf:1  agg:2
        dict = {'layer1':0, 'layer2':1, 'layer3':2,
                'vis':0, 'inf':1, 'agg':2}
        layer_i, type_i = dict[layer_index], dict[type_index]
        
        x_1 = self.Transformer_Encoder[layer_i][type_i][2](x)
        batch, dim, w, h = x_1.shape
        x_1 = x_1.permute(0, 2, 3, 1)
        x_k = x_1.reshape(batch, w * h, dim)
        x_v = x_1.reshape(batch, w * h, dim)
        x_q = x_1.reshape(batch, w * h, dim)

        w_k = self.Transformer_Encoder[layer_i][type_i][0](x_k)
        w_k = F.normalize(w_k, p=2, dim=-1)
        w_k = w_k.permute(0, 1, 2)

        w_q = self.Transformer_Encoder[layer_i][type_i][0](x_q)
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(0, 2, 1)

        dot_prod = torch.bmm(w_q, w_k)
        affinity = F.softmax(dot_prod * 30, dim=-1)

        w_v = self.Transformer_Encoder[layer_i][type_i][1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v = w_v.permute(0, 2, 1)

        output = torch.bmm(affinity, w_v)
        # output=output.permute(0,2,1)
        # output=self.transformer1_FFN[1](nn.Dropout(0.2)(F.relu(((self.transformer1_FFN[0](output))))))
        # output=output.permute(0,2,1)
        output = output.reshape(batch, dim, w, h)
        output = self.Transformer_Encoder[layer_i][type_i][3](output)
        x = x + output
        return x


    # Cross Attention
    def Cross_Attention(self, layer_index, type_index, x, V):
        dict = {'layer1':0, 'layer2':1, 'layer3':2,
                'visagg':0, 'infagg':1}
        layer_i, type_i = dict[layer_index], dict[type_index]
        
        x_1 = self.Transformer_Decoder[layer_i][type_i][2](x)
        batch, dim, w, h = x_1.shape
        V_1 = self.Transformer_Decoder[layer_i][type_i][2](V)
        x_1 = x_1.permute(0, 2, 3, 1)
        V_1 = V_1.permute(0, 2, 3, 1)
        x_k = x_1.reshape(batch, w * h, dim)
        x_v = x_1.reshape(batch, w * h, dim)
        V_q = V_1.reshape(batch, w * h, dim)

        w_k = self.Transformer_Decoder[layer_i][type_i][0](x_k)
        w_k = F.normalize(w_k, p=2, dim=-1)
        w_k = w_k.permute(0, 1, 2)

        w_q = self.Transformer_Decoder[layer_i][type_i][0](V_q)
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(0, 2, 1)

        dot_prod = torch.bmm(w_q, w_k)
        affinity = F.softmax(dot_prod * 30, dim=-1)

        w_v = self.Transformer_Decoder[layer_i][type_i][1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v = w_v.permute(0, 2, 1)

        output = torch.bmm(affinity, w_v)
        # output=output.permute(0,2,1)
        # output=self.transformer1_FFN[1](nn.Dropout(0.2)(F.relu(((self.transformer1_FFN[0](output))))))
        # output=output.permute(0,2,1)
        output = output.reshape(batch, dim, w, h)
        output = self.Transformer_Decoder[layer_i][type_i][3](output)
        x = x + output
        return x
    

    def forward(self, img_v, img_i, k=0, in_layer='conv1', out_layer='fc6'):
        # forward model from in_layer to out_layer
        run = False
        img = img_v
        parallel_dict = {'conv1': self.parallel1, 'conv2': self.parallel2, 'conv3': self.parallel3}
        parallel_skconv_dict = {'conv1': self.parallel1_skconv,
                                'conv2': self.parallel2_skconv,
                                'conv3': self.parallel3_skconv}
        layer_dict = {'conv1': 'layer1', 'conv2': 'layer2', 'conv3': 'layer3'}
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
                    a_b = self.ensemble1_skconv(U)
                    a_b = a_b.reshape(batch_size, 5, channel_dict[name_v], -1)
                    a_b = nn.Softmax(dim=1)(a_b)
                    a_b = list(a_b.chunk(5, dim=1))
                    a_b = list(map(lambda x: x.reshape(batch_size, channel_dict[name_v], 1, 1), a_b))
                    V = list(map(lambda x, y: x * y, V_out, a_b))
                    V = reduce(lambda x, y: x + y, V)
                    torch.cuda.empty_cache()
                
                    img_v = module_v(img_v)
                    img_i = module_i(img_i)
                    img_v = self.Self_Attention(layer_dict[name_v], 'vis', img_v)
                    img_i = self.Self_Attention(layer_dict[name_v], 'inf', img_i)
                    V = self.Self_Attention(layer_dict[name_v], 'agg', V)
                    img_v = self.Cross_Attention(layer_dict[name_v], 'visagg', img_v, V)
                    img_i = self.Cross_Attention(layer_dict[name_v], 'infagg', img_i, V)
                
                img = torch.cat((img_v, img_v), 1)
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
        try:
            print('load LanYangYang model.')
            self.layers_v.load_state_dict(states['layers_v'])
            self.layers_i.load_state_dict(states['layers_i'])
            self.fc.load_state_dict(states['fc'])
            self.parallel1.load_state_dict(states['parallel1'])
            self.parallel2.load_state_dict(states['parallel2'])
            self.parallel3.load_state_dict(states['parallel3'])
            self.parallel1_skconv.load_state_dict(states['parallel1_skcov'])
            self.parallel2_skconv.load_state_dict(states['parallel2_skcov'])
            self.parallel3_skconv.load_state_dict(states['parallel3_skcov'])
            self.ensemble1_skconv.load_state_dict(states['ensemble1_skcov'])
            self.ensemble2_skconv.load_state_dict(states['ensemble2_skcov'])
            self.ensemble3_skconv.load_state_dict(states['ensemble3_skcov'])
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