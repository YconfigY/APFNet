# encoding=utf-8
import os, sys
import yaml
import time
import argparse
import numpy as np
import pdb
import torch
import threading

from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

from data.dataset import init_datasets

# from APFNet.model_bulider import MDNet
from APFNet.model import MDNet
from APFNet.model_repvgg import MDNet_REPVGG
from APFNet.utils.model_params import init_model

from utils.metric import BCELoss, Precision
from utils.optimizer import set_optimizer
from utils.log import get_logger
from utils.model_saver import save_model

# Rememeber to change the cfg for stage
from utils.config import get_config

from new_dataloader import newDataset, newSampler

os.environ["CUDA_VISIBLE_DEVICES"] = '0'



def train_mdnet(model, criterion, evaluator, optimizer):
    # Main trainig loop
    best_prec = 0.8
    logger.info('start training {}-train_on_{}, stage:{} challenge: {}'
                .format(cfg.MODEL.NAME, cfg.DATA.TRAIN_DATASET, cfg.MODEL.STAGE_TYPE, cfg.TRAIN.CHALLENGE))
    new_dataset = newDataset(dataset)
    # scaler = GradScaler()
    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.EPOCHS):
        print('==== Start Epoch {:d}/{:d} ===='.format(epoch+1, cfg.TRAIN.EPOCHS))
        if epoch in cfg.TRAIN.LR.DECAY:
            print('decay learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= cfg.TRAIN.LR.GAMMA
        # Training
        model.train()
        precision = np.zeros(len(dataset))
        seq_index_list = newSampler(dataset)
        dataloader = DataLoader(new_dataset, 1, sampler=seq_index_list, num_workers=16)
        tic_epoch = time.time()
        for i, (pos_regions_v, neg_regions_v, pos_regions_i, neg_regions_i) in enumerate(dataloader):
            optimizer.zero_grad()
            seq_index = seq_index_list[i]
            tic = time.time()
            # with autocast():
            pos_regions_v = pos_regions_v[0].cuda()
            neg_regions_v = neg_regions_v[0].cuda()
            pos_regions_i = pos_regions_i[0].cuda()
            neg_regions_i = neg_regions_i[0].cuda()
            pos_score = model(pos_regions_v, pos_regions_i, seq_index)
            neg_score = model(neg_regions_v, neg_regions_i, seq_index)
            loss = criterion(pos_score, neg_score)
            if i % cfg.TRAIN.BATCH_ACCUM == 0:
                model.zero_grad()
            # scaler.scale(loss).backward()
            loss.backward()
            if i % cfg.TRAIN.BATCH_ACCUM == cfg.TRAIN.BATCH_ACCUM - 1 or i == len(seq_index_list) - 1:
                if cfg.TRAIN.CLIP_GRAD:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.CLIP_GRAD)
                optimizer.step()
                # scaler.step(optimizer)
                # scaler.update()
                
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
    global dataset
    dataset = init_datasets()
    torch.backends.cudnn.benchmark = True
    # model = MDNet(num_branches=len(dataset))
    model = MDNet_REPVGG(deploy=True, num_branches=len(dataset))

    init_model(model, cfg)
    if cfg.MODEL.DEVICE:
        model = model.cuda()
    criterion = BCELoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, cfg.TRAIN.LR.BASE, cfg.MODEL.STAGE.LR_LAYER, cfg.MODEL.STAGE.LR_MULT)
    train_mdnet(model, criterion, evaluator, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Change it by yourslef to train different attribute branches
    # parser.add_argument("-stage", default=1, type=int, help='current stage')
    # parser.add_argument("-epoch", default=200, type=int, help='current stage train epoch')
    # parser.add_argument("-challenge", default=, type=str)
    # args = parser.parse_args()
    cfg = get_config()
    logger = get_logger()
    main()
        