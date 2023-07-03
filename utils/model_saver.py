import torch
import os

from utils.config import cfg


def save_model(model, logger, epoch, isBest=False, isCheckpoint=False):
    """save best or excellent model checkpointer according to the stage.

    Args:
        model (_type_): training model.
        logger (_type_): 
        epoch (int): the number of current epoch.
        isBest (boolean): best training model.
        isCheckpoint (boolean): 
    """
    snapshot_path = os.path.join(cfg.TRAIN.SNAPSHOT + '_' + str(cfg.MODEL.STAGE_TYPE), 
                                 cfg.DATA.TRAIN_DATASET, cfg.TRAIN.CHALLENGE)
    checkpoint_path = os.path.join('./resume/stage_' + str(cfg.MODEL.STAGE_TYPE), 
                                   cfg.DATA.TRAIN_DATASET, cfg.TRAIN.CHALLENGE)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if cfg.MODEL.DEVICE:
        model = model.cpu()
    states = get_states(model)
    snapshot_file = os.path.join(snapshot_path, cfg.TRAIN.CHALLENGE + '_' + str(epoch+1) + '.pth')
    if isBest:
        snapshot_file = os.path.join(checkpoint_path, cfg.TRAIN.CHALLENGE +'.pth')
    if isCheckpoint: 
        snapshot_file = os.path.join(checkpoint_path, cfg.TRAIN.CHALLENGE + '_' + str(epoch+1) + '.pth')
    torch.save(states, snapshot_file)
    logger.info('Save model to {:s}'.format(snapshot_file))
    if cfg.MODEL.DEVICE:
        model = model.cuda()


def get_states(model):
    """get the 'state_dict' of the model to be saved according to the stage.

    Args:
        model (_type_): training model.

    Returns:
        state_dict : the 'state_dict' of the model to be saved
    """
    states = {}
    # save the SKNet's state.dict() of the ASFB or(stage1)/and(stage2、3) the ABAF
    for _type in model.blockes.values():
        for _layer_name, _layer in _type.items():
            states[_layer_name] = _layer.state_dict()

    if cfg.MODEL.STAGE_TYPE > 1:
        states['layers_v'] =  model.layers_v.state_dict()
        states['layers_i'] =  model.layers_i.state_dict()
        states['fc'] =  model.fc.state_dict()
        
        if cfg.MODEL.STAGE_TYPE == 3:
            # save the transformers' state.dict() of the ASEF
            _module_index = 0
            for _module in model.transformer.values():
                _module_index += 1
                _units = ['encoder1', 'encoder2', 'encoder3', 'decoder1', 'decoder2']
                _units_index = 0
                for _layer in _module.values():
                    states[f'transformer{_module_index}_{_units[_units_index]}'] = _layer.state_dict()
                    _units_index += 1

    return states