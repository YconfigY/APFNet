import math
import torch.nn as nn

from collections import OrderedDict

from utils.config import cfg
# from APFNet.utils.model_load import load_model
from APFNet.utils.model_load_repvgg import load_model


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


def init_modules(model):
    if cfg.MODEL.STAGE_TYPE < 3 : # stage=1 or stage=2
        # init fc weight and bias
        for m in model.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
                
        # init ASFB and ABAF weight and bias
        for _type in model.blockes.values():
            for _layer_name, _layer in _type.items():
                for m in _layer.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.normal_(m.weight, 0, 0.01)
                        if 'skconv' not in _layer_name:
                            nn.init.constant_(m.bias, 0.1)
    
    # init branches weight and bias
    for m in model.branches.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
            
    # init transformer encoder and decoder weight and bias
    if cfg.MODEL.STAGE_TYPE == 3:
        for _module in model.transformer.values():
            for _unit in _module.values():
                for m in _unit.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.normal_(m.weight, 0, 0.01)
                    if isinstance(m, nn.Linear):
                        m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
                        if m.bias is not None:
                            m.bias.data.zero_()


def build_param_dict(model):
    for name, module in model.layers_v.named_children():
        append_params(model.params, module, 'layers_v' + name)
    for name, module in model.layers_i.named_children():
        append_params(model.params, module, 'layers_i' + name)
    for name, module in model.fc.named_children():
        append_params(model.params, module, name)
    for k, module in enumerate(model.branches):
        append_params(model.params, module, 'fc6_%d' % (k))
    # add new architecher
    for _type in model.blockes.values():
        for _layer_name, _layer in _type.items():
            for name, module in _layer.named_children():
                if 'skconv' in _layer_name and 'pool' in name:
                    continue
                append_params(model.params, module, name)
    # add transformer
    if cfg.MODEL.STAGE_TYPE >= 3:
        for _units in model.transformer.values():
            for _unit in _units.values():
                for _name, _module in _unit.named_children():
                    append_params(model.params, _module, _name)


def set_learnable_params(model, layers):
    """set learnable params in selected layers before training

    Args:
        layers (_type_): selected layers that usually belong to backbone
    """
    for _params_name, _params in model.params.items():
        if any([_params_name.startswith(_layer) for _layer in layers]):
            _params.requires_grad = True
        else:
            _params.requires_grad = False


def get_learnable_params(model):
    """get learnable params in every layer before training

    Returns:
        params (list[]): learnable params in every layer
    """
    params = OrderedDict()
    for _params_name, _params in model.params.items():
        if _params.requires_grad:
            params[_params_name] = _params
    print('get_learnable_params', params.keys())
    return params


def get_all_params(model):
    params = OrderedDict()
    for _params_name, _params in model.params.items():
        params[_params_name] = _params
    return params


def init_model(model, cfgs):
    """Init model. To start with, init model's all modules' weights and bias.
    Secondly, building parameters dict. Thirdly, loading pretrained model or checkpointer. 
    Finally, setting leranabel parameters in the model on the training stage.

    Args:
        model (_type_): _description_
        cfgs (_type_): _description_
    """
    init_modules(model)
    build_param_dict(model)
    load_model(model)
    set_learnable_params(model, cfgs.MODEL.STAGE.LR_LAYER)
    