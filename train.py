# encoding=utf-8
import os, sys
import yaml
import time
import argparse
import numpy as np
import pdb
import torch

from concurrent import futures
from tqdm import tqdm

from data.dataset import init_datasets

# from APFNet.model_bulider import MDNet
from APFNet.model import MDNet
from APFNet.utils.model_params import init_model

from utils.metric import BCELoss, Precision
from utils.optimizer import set_optimizer
from utils.log import get_logger
from utils.model_saver import save_model

# Rememeber to change the cfg for stage
from utils.config import cfg, get_config


os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def get_data(k):
    """load a part of dataset or full dataset to generate positive and negative samples in advance

    Args:
        k (int): seq index

    Returns:
        seq
    """
    return dataset[k].next()

def process_train_data(k_list, epoch, start_index=0):
    data = []
    end_index = len(k_list)
    if start_index == 0 and cfg.TRAIN.STAGE_NUM > 1 and cfg.DATA.DATASET == 'RGBT234':
        end_index = len(k_list) // 2
    if start_index != 0:
        print('')
    print('Processing train_data from', start_index+1, 'to', end_index)
    tic_process = time.time()
    
    with futures.ProcessPoolExecutor(max_workers=cfg.DATA.NUM_WORKERS) as executor:
        for k, (pos_v, neg_v, pos_i, neg_i) in zip(
            k_list[start_index:end_index], executor.map(get_data, k_list[start_index:end_index])):
            # data.append([pos_v, neg_v, pos_i, neg_i])
            data.append([pos_v.cuda(), neg_v.cuda(), pos_i.cuda(), neg_i.cuda()])
    if start_index == 0:
        if end_index == len(k_list):
            logger.info('Process epoch:{} train_data, spend time:{:.3f}'.format(epoch+1, time.time()-tic_process))
        else:
            logger.info('Process epoch:{} first part of train_data, spend time:{:.3f}'
                        .format(epoch+1, time.time()-tic_process))
    else:
        logger.info('Process epoch:{} last part of train_data, spend time:{:.3f}'
                    .format(epoch+1, time.time()-tic_process))
    return data


def train_mdnet(dataset, model, criterion, evaluator, optimizer):
    # Main trainig loop
    best_prec = 0.8
    logger.info('start training APFNet, stage:{} challenge: {}'.format(cfg.TRAIN.STAGE_NUM, cfg.TRAIN.CHALLENGE))
    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.EPOCHS):
        print('==== Start Epoch {:d}/{:d} ===='.format(epoch+1, cfg.TRAIN.EPOCHS))
        if epoch in cfg.TRAIN.LR.DECAY:
            print('decay learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= cfg.TRAIN.LR.GAMMA
        # Training
        model.train()
        precision = np.zeros(len(dataset))
        seq_index_list = np.random.permutation(len(dataset))
        
        count = 0
        tic_epoch = time.time()
        train_data = process_train_data(seq_index_list, epoch)
        
        for i, seq_index in enumerate(seq_index_list):
            if i == len(seq_index_list)//2 and cfg.TRAIN.STAGE_NUM>1:
                count = i
                train_data = process_train_data(seq_index_list, epoch, i)
            tic = time.time()
            # training
            pos_regions_v, neg_regions_v, pos_regions_i, neg_regions_i = train_data[i-count]
            pos_score = model(pos_regions_v, pos_regions_i, seq_index)
            neg_score = model(neg_regions_v, neg_regions_i, seq_index)
            loss = criterion(pos_score, neg_score)
            if i % cfg.TRAIN.BATCH_ACCUM == 0:
                model.zero_grad()
            loss.backward()
            if i % cfg.TRAIN.BATCH_ACCUM == cfg.TRAIN.BATCH_ACCUM - 1 or i == len(seq_index_list) - 1:
                if cfg.TRAIN.CLIP_GRAD:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.CLIP_GRAD)
                optimizer.step()

            precision[seq_index] = evaluator(pos_score, neg_score)

            toc = time.time() - tic
            print('\rEpoch:{:3d}/{:3d}, Iteration:{:3d}/{:3d} Domain {:3d}, Loss {:.3f}, Precision {:.3f}, Time {:.3f}'
                  .format((epoch+1), cfg.TRAIN.EPOCHS, i+1, len(seq_index_list), seq_index, loss.item(), 
                          precision[seq_index], toc), end="", flush=True)
        print('')    
        toc_epoch = time.time()
        cur_prec = precision.mean()
        logger.info('Epoch:{:-3d} Mean Precision:{:.3f}, Epoch Time:{:.3f}'
                    .format((epoch+1), cur_prec, toc_epoch-tic_epoch))

        if (epoch+1) % 25 == 0:
            save_model(model, logger, epoch, isCheckpoint=True)
        if cur_prec > best_prec:
            best_prec = cur_prec
            save_model(model, logger, epoch, isBest=True)
        if cur_prec > 0.95:
            save_model(model, logger, epoch)


def main():
    model = MDNet(len(dataset))
    # model = ModelBuilder().cuda()
    init_model(model, cfgs)
    if cfg.DEVICE:
        model = model.cuda()
    criterion = BCELoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, cfg.TRAIN.LR.BASE, cfgs.TRAIN.STAGE.LAYER, cfgs.TRAIN.STAGE.LR_MULT)
    train_mdnet(dataset, model, criterion, evaluator, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Change it by yourslef to train different attribute branches
    parser.add_argument("-stage", default=1, type=int, help='current train stage')
    parser.add_argument("-challenge", default='ILL', type=str)
    parser.add_argument("-resume", default='', type=str, help='resume model weight path')
    args = parser.parse_args()

    ## option setting
    # cfg = get_config(args)
    # if opts['stage'] == 2:
    #     opts['resume'] = './resume/' + opts['challenge'] + '.pth'
    logger = get_logger()
    dataset = init_datasets()
    cfgs = get_config()
    main()