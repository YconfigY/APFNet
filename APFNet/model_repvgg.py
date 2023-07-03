# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict
from collections import OrderedDict
from functools import reduce

from APFNet.backbone.repvgg import get_RepVGG_func_by_name
from utils.config import cfg


'''
   Model stage1 is MDNet(RGBT)+each challenge branch 
   We need add each attribute branch one by one 
'''


class MDNet_REPVGG(nn.Module):
    def __init__(self, deploy=False, num_branches=1):
        super(MDNet_REPVGG, self).__init__()
        self.stage = cfg.MODEL.STAGE_TYPE
        if cfg.MODEL.STAGE_TYPE == 'test':
            self.stage = 3
            deploy = True
        self.num_branches = num_branches
        self.params = OrderedDict()
        # backbone
        
        RepVGG_A2 = get_RepVGG_func_by_name('RepVGG-A2')
        backbone = RepVGG_A2(deploy)
        if cfg.MODEL.PRETRAINED:
            checkpoint = torch.load('./data/models/RepVGG-A2-deploy.pth' if deploy else cfg.MODEL.PRETRAINED)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)
        
        self.layers_v, self.layers_i = [nn.Sequential(OrderedDict([
            ('conv1', backbone.stage0), ('conv2', backbone.stage1),
            ('conv3', backbone.stage2), ('conv4', backbone.stage3)])) for _ in range(2)]
        self.fc = nn.Sequential(OrderedDict([
            ('fc4', nn.Sequential(nn.Linear(384 * 14 * 14 * 2, 512), nn.ReLU(inplace=True))),
            ('fc5', nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 512),  nn.ReLU(inplace=True)))]))
        # feature extracter
        if self.stage == 1:
            self.parallel1, self.parallel2, self.parallel3, self.parallel4 = \
                [nn.Sequential(OrderedDict([(f'parallel{_idx+1}_conv', nn.Sequential(
                    nn.Conv2d(_cha, _chb, kernel_size=3, stride=2, padding=1), nn.ReLU()))])) 
                 for _idx, (_cha, _chb) in enumerate([[3,64],[64,96],[96,192],[192,384]])]  # 224->112,(224-3+2)/2+1=112

            self.parallel1_skconv, self.parallel2_skconv, self.parallel3_skconv, self.parallel4_skconv= \
            [nn.Sequential(OrderedDict([
                (f'parallel{_idx+1}_skconv_global_pool', nn.AdaptiveAvgPool2d(1)),  # output=1
                (f'parallel{_idx+1}_skconv_fc1', nn.Sequential(
                    nn.Conv2d(_chb, _cha, 1, bias=False), nn.ReLU(inplace=True))),
                (f'parallel{_idx+1}_skconv_fc2', nn.Sequential(
                     nn.Conv2d(_cha, _chb * 2, 1, 1, bias=False)))]))
             for _idx, (_cha, _chb) in enumerate([[32,64],[64,96],[64,192],[96,384]])]
        elif self.stage >= 2:
            # the first branch to fuse  # 0:FM 1:OCC 2:SV 3:TC 4:ILL
            self.parallel1 = nn.ModuleList([nn.Sequential(OrderedDict([
                ('parallel1_conv1', nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, stride=2), nn.ReLU())),
                ('parallel1_conv2', nn.Sequential(nn.Conv2d(32, 96, kernel_size=4, stride=2)))])) for _ in range(5)])

            self.parallel2 = nn.ModuleList([nn.Sequential(OrderedDict([
                ('parallel2_conv1', nn.Sequential(nn.Conv2d(96, 256, kernel_size=3, stride=2),  
                                                  nn.MaxPool2d(kernel_size=8, stride=1)))])) for _ in range(5)])
            
            self.parallel3 = nn.ModuleList([nn.Sequential(OrderedDict([
                ('parallel3_conv1', nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=1), 
                                                  nn.MaxPool2d(kernel_size=3, stride=1)))])) for _ in range(5)])
            # Squeeze operation see [SENet](https://arxiv.org/abs/1709.01507, code:https://github.com/miraclewkf/SENet-PyTorch)
            self.parallel1_skconv, self.parallel2_skconv, self.parallel3_skconv = \
            [nn.ModuleList([nn.Sequential(OrderedDict([
                (f'parallel{_idx+1}_skconv_global_pool', nn.AdaptiveAvgPool2d(1)),
                (f'parallel{_idx+1}_skconv_fc1', nn.Sequential(
                    nn.Conv2d(_channel_b, _channel_a, 1, bias=False), nn.ReLU(inplace=True))),
                (f'parallel{_idx+1}_skconv_fc2', nn.Sequential(
                    nn.Conv2d(_channel_a, _channel_b * 2, 1, 1, bias=False)))])) for _ in range(5)])
             for _idx, (_channel_a, _channel_b) in enumerate([[32,96],[32,256],[64,512]])]
            # filter the five challenge information
            # Squeeze operation see [SENet](https://arxiv.org/abs/1709.01507, code:https://github.com/miraclewkf/SENet-PyTorch)
            self.ensemble1_skconv, self.ensemble2_skconv, self.ensemble3_skconv = \
            [nn.Sequential(OrderedDict([
                (f'ensemble{_idx+1}_skconv_global_pool', nn.AdaptiveAvgPool2d(1)), 
                (f'ensemble{_idx+1}_skconv_fc1', nn.Sequential(
                    nn.Conv2d(_channel_b, _channel_a * 5, 1, bias=False), nn.ReLU(inplace=True))),
                (f'ensemble{_idx+1}_skconv_fc2', nn.Sequential(
                    nn.Conv2d(_channel_a * 5, _channel_b * 5, 1, 1, bias=False)))]))
             for _idx, (_channel_a, _channel_b) in enumerate([[32,96],[64,256],[128,512]])]
            
            if self.stage >= 3:
                # We add Encoders and Decoders here. And every layer has there encoders and two decoders.
                self.transformer1_encoder1, self.transformer1_encoder2, self.transformer1_encoder3, \
                    self.transformer1_decoder1, self.transformer1_decoder2, \
                    self.transformer2_encoder1, self.transformer2_encoder2, self.transformer2_encoder3, \
                    self.transformer2_decoder1, self.transformer2_decoder2, \
                    self.transformer3_encoder1, self.transformer3_encoder2, self.transformer3_encoder3, \
                    self.transformer3_decoder1, self.transformer3_decoder2 = [
                        nn.Sequential(OrderedDict([
                            (f'transformer{_idx+1}_{_name}_WK', nn.Sequential(nn.Linear(_channel_a, _channel_a))),
                            (f'transformer{_idx+1}_{_name}_WV', nn.Sequential(nn.Linear(_channel_a, _channel_a))),
                            (f'transformer{_idx+1}_{_name}_fc_reduce', nn.Sequential(
                                nn.Conv2d(_channel_b, _channel_a, 1, 1, bias=False))),
                            (f'transformer{_idx+1}_{_name}_fc_rise', nn.Sequential(
                                nn.Conv2d(_channel_a, _channel_b, 1)))]))
                        for _idx, (_channel_a, _channel_b) in enumerate([[32,96],[64,256],[128,512]])
                        for _name in ['encoder1', 'encoder2', 'encoder3', 'decoder1', 'decoder2']]
        # fc6
        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5), 
                                                     nn.Linear(512, 2)) for _ in range(num_branches)])
        
        self.blockes = {'parallel':{}, 'p_skconv':{}}
        for i, (parallel, p_skconv) in enumerate(zip(['parallel1','parallel2','parallel3', 'parallel4'], 
                                                     ['parallel1_skconv', 'parallel2_skconv', 
                                                      'parallel3_skconv', 'parallel3_skconv'])):
            self.blockes['parallel'][parallel] = eval('self.{}'.format(parallel))
            self.blockes['p_skconv'][p_skconv] = eval('self.{}'.format(p_skconv))

        if self.stage >= 2:
            self.blockes.update({'e_skconv':{}})
            for i, e_skconv in enumerate(['ensemble1_skconv', 'ensemble2_skconv', 'ensemble3_skconv']):
                self.blockes['e_skconv'][e_skconv] = eval('self.{}'.format(e_skconv))
            if self.stage >= 3:
                self.transformer = {'module_1':{}, 'module_2':{}, 'module_3':{}}
                self.transformer = EasyDict(self.transformer)
                for i, _module_ in enumerate(['module_1', 'module_2', 'module_3']):
                    for j, _type_ in enumerate(['vis', 'inf', 'agg', 'visagg', 'infagg']):
                        if j < 3:
                            self.transformer[_module_][_type_] = eval('self.transformer{}_encoder{}'.format(i+1, j+1))
                        else:
                            self.transformer[_module_][_type_] = eval('self.transformer{}_decoder{}'.format(i+1, j+1-3))
                
    
    def transformer_encoder(self, encoder, input):
        x = encoder[2](input)
        batch, dim, w, h = x.shape
        x = x.permute(0, 2, 3, 1)
        x_k = x.reshape(batch, w * h, dim)
        x_v = x.reshape(batch, w * h, dim)
        x_q = x.reshape(batch, w * h, dim)

        w_k = encoder[0](x_k)
        w_k = F.normalize(w_k, p=2, dim=-1)
        w_k = w_k.permute(0, 1, 2)

        w_q = encoder[0](x_q)
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(0, 2, 1)

        dot_prod = torch.bmm(w_q, w_k)
        affinity = F.softmax(dot_prod * 30, dim=-1)

        w_v = encoder[1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v = w_v.permute(0, 2, 1)

        output = torch.bmm(affinity, w_v)
        output = output.reshape(batch, dim, w, h)
        output = encoder[3](output)
        return input + output
    
    
    def transformer_decoder(self, decoder, input, outs_ABAF):
        x = decoder[2](input)
        batch, dim, w, h = x.shape
        out_ABAF = decoder[2](outs_ABAF)
        x = x.permute(0, 2, 3, 1)
        out_ABAF = out_ABAF.permute(0, 2, 3, 1)
        x_k = x.reshape(batch, w * h, dim)
        x_v = x.reshape(batch, w * h, dim)
        out_ABAF_q = out_ABAF.reshape(batch, w * h, dim)

        w_k = decoder[0](x_k)
        w_k = F.normalize(w_k, p=2, dim=-1)
        w_k = w_k.permute(0, 1, 2)

        w_q = decoder[0](out_ABAF_q)
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(0, 2, 1)

        dot_prod = torch.bmm(w_q, w_k)
        affinity = F.softmax(dot_prod * 30, dim=-1)

        w_v = decoder[1](x_v)
        w_v = F.normalize(w_v, p=2, dim=-1)
        w_v = w_v.permute(0, 2, 1)

        output = torch.bmm(affinity, w_v)
        output = output.reshape(batch, dim, w, h)
        output = decoder[3](output)
        return input + output
    

    def get_dict_stage(self):
        dict_stage = {'conv1':{}, 'conv2':{}, 'conv3':{}, 'conv4':{}}
        dict_stage = EasyDict(dict_stage)
        for i, channel in enumerate([64, 96, 192, 384]):
            dict_stage['conv{}'.format(i+1)] = {'parallel':eval('self.parallel{}'.format(i+1)), 
                                                'p_skconv':eval('self.parallel{}_skconv'.format(i+1)), 
                                                'channel':channel}
            if self.stage >= 2:
                dict_stage['conv{}'.format(i+1)]['e_skconv'] = eval('self.ensemble{}_skconv'.format(i+1))
                if self.stage >= 3:
                    dict_stage['conv{}'.format(i+1)]['layer'] = 'module_{}'.format(i+1)
        return dict_stage


    def forward(self, img_v, img_i, k=0, in_layer='conv1', out_layer='fc6'):
        """Forward model from in_layer to out_layer

        Args:
            img_v (Tensor): Visible image
            img_i (Tensor): Infrared image
            k (int, optional): Branch number. Defaults to 0.
            in_layer (str, optional): forward model start layer. Defaults to 'conv1'.
            out_layer (str, optional): forward model end layer. Defaults to 'fc6'.

        Returns:
            score (Tensor): The socre of visible and infrared images
        """
        run = False
        img = img_v
        batch_size = img_v.size(0)
        dict_ = self.get_dict_stage()
        
        for (name_v, module_v), (name_i, module_i) in zip(
            self.layers_v.named_children(), self.layers_i.named_children()):
            if name_v == in_layer:
                run = True
            if run:
                if name_v in ['conv1', 'conv2', 'conv3', 'conv4']:
                    outs_parallel = []
                    if self.stage == 1:
                        img_v_parallel = dict_[name_v]['parallel'](img_v)  #107*107->25*25->5*5->3*3
                        img_i_parallel = dict_[name_v]['parallel'](img_i)
                        outs_parallel.append(img_v_parallel)
                        outs_parallel.append(img_i_parallel)  # [[N,96/256/512,25,25],[N,96/256/512,25,25]]
                        # Operation of element-wise addition for outs_parallel
                        EwA_ASFB_1 = reduce(lambda x, y: x + y, outs_parallel)  # [N,96/256/512,25,25]
                        out_skconv = dict_[name_v]['p_skconv'](EwA_ASFB_1)
                        out_skconv = out_skconv.reshape(  # [N,(96/256/512)*2,1,1]->[N,2,96/256/512,1]
                            batch_size, 2, dict_[name_v]['channel'], -1)
                        out_skconv = nn.Softmax(dim=1)(out_skconv)
                        # Splits out_skconv into 2 chunks, [N,2,96/256/512,1]->[[N,1,96/256/512,1], [N,1,96/256/512,1]]
                        outs_skconv = list(out_skconv.chunk(2, dim=1))
                        outs_skconv = list(map(lambda x: x.reshape(  # [N,1,96/256/512,1]->[N,96/256/512,1,1]
                            batch_size, dict_[name_v]['channel'], 1, 1), outs_skconv))
                        # Operation of element-wise multiplication between outs_parallel and outs_skconv
                        EwA_ASFB_2 = list(map(lambda x, y: x * y, outs_parallel, outs_skconv))
                        # operation of element-wise addition for EwA_ASFB_2
                        outs_ASFB = reduce(lambda x, y: x + y, EwA_ASFB_2)  # [N,96/256/512,25/5/3,25/5/3]
                        out_stage = outs_ASFB  # as input to the next layer(conv2、3)
                    if self.stage >= 2:
                        outs_ASFB =[]
                        for j in range(len(cfg.DATA.CHALLENGE)):  # 0:FM 1:OCC 2:SV 3:TC 4:ILL
                            img_v_parallel = dict_[name_v]['parallel'][j](img_v)  #107*107->25*25->5*5->3*3
                            img_i_parallel = dict_[name_v]['parallel'][j](img_i)
                            outs_parallel.append(img_v_parallel)
                            outs_parallel.append(img_i_parallel)  # [[N,96/256/512,25/5/3,25/5/3], [..]]
                            # Operation of element-wise addition for 'outs_parallel'
                            EwA_ASFB_1 = reduce(lambda x, y: x + y, outs_parallel)  # [N,96/256/512,25/5/3,25/5/3]
                            out_skconv = dict_[name_v]['p_skconv'][j](EwA_ASFB_1)
                            out_skconv = out_skconv.reshape(  # [N,(96/256/512)*2,1,1]->[N,2,96/256/512,1]
                                batch_size, 2, dict_[name_v]['channel'], -1)  
                            out_skconv = nn.Softmax(dim=1)(out_skconv)
                            # Splits 'out_skconv' into 2 chunks, [N,2,96/256/512,1]->[[N,1,96/256/512,1], [...]]
                            outs_skconv = list(out_skconv.chunk(2, dim=1))
                            outs_skconv = list(map(lambda x: x.reshape(  # [N,1,96/256/512,1]->[N,96/256/512,1,1]
                                batch_size, dict_[name_v]['channel'], 1, 1), outs_skconv))
                            # Operation of element-wise multiplication between 'outputs_parallel' and 'outs_skconv'
                            EwM_ASFB = list(map(lambda x, y: x * y, outs_parallel, outs_skconv))
                            # operation of element-wise addition for EwM_ASFB
                            EwA_ASFB_2 = reduce(lambda x, y: x + y, EwM_ASFB)  # [N, 96/256/512, 25/5/3, 25/5/3]
                            outs_ASFB.append(EwA_ASFB_2)  # append every attribute branch's output to 'outs_ASFB'
                            outs_parallel.clear()
                        # input to ensemble for img_v: ALL
                        EwA_ABAF_1 = reduce(lambda x, y: x + y, outs_ASFB)  # [N, 96/256/512, 25/5/3, 25/5/3]
                        out_e_skconv = dict_[name_v]['e_skconv'](EwA_ABAF_1)  # [N, 96/256/512*5, 1, 1]
                        out_e_skconv = out_e_skconv.reshape(
                            batch_size, 5, dict_[name_v]['channel'], -1)  # [N, 5, 96/256/512, 1]
                        out_e_skconv = nn.Softmax(dim=1)(out_e_skconv)
                        outs_e_skconv = list(out_e_skconv.chunk(5, dim=1))  # [[N, 1, 96/256/512, 1],[...]*4]
                        outs_e_skconv = list(map(lambda x: x.reshape(  # [[N, 96/256/512, 1, 1],[...]*4]
                            batch_size, dict_[name_v]['channel'], 1, 1), outs_e_skconv))
                        EwM_ABAF = list(map(lambda x, y: x * y, 
                                            outs_ASFB, outs_e_skconv))  # [[N, 96/256/512, 25/5/3, 25/5/3],[...]*4]
                        outs_ABAF = reduce(lambda x, y: x + y, EwM_ABAF)  # EwA_ABAF_2
                        out_stage = outs_ABAF  # as input to the next layer(conv2、3)
                        if self.stage >= 3:
                            in_top_encoder = module_v(img_v)  # top encoder input
                            in_bottom_encoder = module_i(img_i)  # bottom encoder input
                            out_top_encoder = self.transformer_encoder(
                                self.transformer[dict_[name_v]['layer']].vis, in_top_encoder)
                            out_bottom_encoder = self.transformer_encoder(
                                self.transformer[dict_[name_v]['layer']].inf, in_bottom_encoder)
                            out_middle_encoder = self.transformer_encoder(
                                self.transformer[dict_[name_v]['layer']].agg, out_stage)
                            out_upper_decoder = self.transformer_decoder(
                                self.transformer[dict_[name_v]['layer']].visagg, out_top_encoder, out_middle_encoder)
                            out_lower_decoder = self.transformer_decoder(
                                self.transformer[dict_[name_v]['layer']].infagg, out_bottom_encoder, out_middle_encoder)
                            img_v, img_i = out_upper_decoder, out_lower_decoder  # as input to the next layer(conv2、3)
                if self.stage <= 2:
                    if name_v == 'conv1':
                        img_v = module_v(img_v)  # RepVGG A2, [N,3,224,224]->[N,64,112,112]
                        img_i = module_i(img_i)
                    else:
                        for _mo_v, _mo_i in zip(module_v, module_i):
                            img_v = _mo_v(img_v)  # RepVGG A2, [N,64,112,112]->[N,96,56,56]
                            img_i = _mo_i(img_i)  #            ->[N,192,28,28]->[N,384,14,14]
                    img_v = img_v + out_stage
                    img_i = img_i + out_stage
                if name_v == 'conv4' or self.stage >= 3:
                    img = torch.cat((img_v, img_i), 1)  # [N,512,3,3]->[N,1024,3,3]
                    img = img.contiguous().view(img.size(0), -1)  # [N,1024,3,3]->[N,1024*3*3]
                if name_v == out_layer:
                    return img
        img = self.fc(img)  # [N, 512*3*3*2]->[N,512]
        img = self.branches[k](img)  # k-th branch, [N,512]->[N,2]
        if out_layer == 'fc6':
            return img
        elif out_layer == 'fc6_softmax':
            return F.softmax(img, dim=1)