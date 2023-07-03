import pickle
import os
import numpy as np
import cv2
import torch
import math

from torch.utils.data import Dataset

from data.sample_generator import SampleGenerator
from utils.utils import crop_image2
from utils.config import cfg


class RegionDataset(Dataset):
    def __init__(self, img_list_v, img_list_i, gt):
        self.img_list_v = np.asarray(img_list_v)
        self.img_list_i = np.asarray(img_list_i)
        self.gt = gt

        self.batch_frames = cfg.DATA.BATCH_FRAMES  # Number of random sampling frames(default=8)
        self.batch_pos = cfg.DATA.BATCH_POS  # Number of positive samples from every random sampling frame
        self.batch_neg = cfg.DATA.BATCH_NEG  # Number of negative samples from every random sampling frame

        self.overlap_pos = cfg.DATA.OVERLAP_POS
        self.overlap_neg = cfg.DATA.OVERLAP_NEG

        self.crop_size = cfg.DATA.IMG_SIZE  # Size of input images
        self.padding = cfg.DATA.PADDING

        self.flip = cfg.AUG.FLIP
        self.rotate = cfg.AUG.ROTATE
        self.blur = cfg.AUG.BLUR

        self.index = np.random.permutation(len(self.img_list_v))  # img list index after shuffled
        self.pointer = 0  # strat point

        image_v = np.asarray(cv2.imread(img_list_v[0])[:, :, ::-1])
        self.pos_generator = SampleGenerator('uniform', image_v.size, cfg.DATA.TRANS_POS, cfg.DATA.SCALE_POS)
        self.neg_generator = SampleGenerator('uniform', image_v.size, cfg.DATA.TRANS_NEG, cfg.DATA.SCALE_NEG)

    def __iter__(self):
        return self

    def __next__(self):
        next_pointer = min(self.pointer + self.batch_frames, len(self.img_list_v))
        idx = self.index[self.pointer:next_pointer]
        if len(idx) < self.batch_frames:
            self.index = np.random.permutation(len(self.img_list_v))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer  # next_pointer become the next starting point

        pos_regions_v = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')
        neg_regions_v = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')
        pos_regions_i = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')
        neg_regions_i = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')
        
        pos_examples = np.empty((0, 4), dtype='float32')
        neg_examples = np.empty((0, 4), dtype='float32')
        
        for i, (img_path_v, img_path_i, bbox) in enumerate(
            zip(self.img_list_v[idx], self.img_list_i[idx], self.gt[idx])):
            
            # image_v = np.asarray(cv2.imread(img_path_v)[:, :, ::-1])
            # image_i = np.asarray(cv2.imread(img_path_i)[:, :, ::-1])

            # n_pos = (self.batch_pos - len(pos_regions_v)) // (self.batch_frames - i)
            # n_neg = (self.batch_neg - len(neg_regions_v)) // (self.batch_frames - i)

            # pos_examples = self.pos_generator(bbox, 32, overlap_range=self.overlap_pos)  # 4*4, 4 positive bbox
            # neg_examples = self.neg_generator(bbox, 96, overlap_range=self.overlap_neg)  # 12*4, 12 negative bbox

            pos_examples = np.concatenate((pos_examples, self.pos_generator(
                bbox, 4, overlap_range=self.overlap_pos)), axis=0)  # 4*4, 4 positive bbox
            neg_examples = np.concatenate((neg_examples, self.neg_generator(
                bbox, 12, overlap_range=self.overlap_neg)), axis=0)  # 12*4, 12 negative bbox

            # pos_regions_v = np.concatenate((pos_regions_v, self.extract_regions(image_v, pos_examples)), axis=0)
            # neg_regions_v = np.concatenate((neg_regions_v, self.extract_regions(image_v, neg_examples)), axis=0)

            # pos_regions_i = np.concatenate((pos_regions_i, self.extract_regions(image_i, pos_examples)), axis=0)
            # neg_regions_i = np.concatenate((neg_regions_i, self.extract_regions(image_i, neg_examples)), axis=0)


        pos_regions_v = torch.from_numpy(self.extract_regions(self.img_list_v[idx], pos_examples, isPos=True))  # [32,3,107,107]
        neg_regions_v = torch.from_numpy(self.extract_regions(self.img_list_v[idx], neg_examples, isNeg=True))  # [96,3,107,107]
        pos_regions_i = torch.from_numpy(self.extract_regions(self.img_list_i[idx], pos_examples, isPos=True))  # [32,3,107,107]
        neg_regions_i = torch.from_numpy(self.extract_regions(self.img_list_i[idx], neg_examples, isNeg=True))  # [96,3,107,107]    
            
        # pos_regions_v = torch.from_numpy(pos_regions_v)  # [32,3,107,107]
        # neg_regions_v = torch.from_numpy(neg_regions_v)  # [96,3,107,107]
        # pos_regions_i = torch.from_numpy(pos_regions_i)  # [32,3,107,107]
        # neg_regions_i = torch.from_numpy(neg_regions_i)  # [96,3,107,107]
        return pos_regions_v, neg_regions_v, pos_regions_i, neg_regions_i
    
    next = __next__

    def extract_regions(self, images, samples, isPos=False, isNeg=False):
        regions = np.zeros((len(samples), self.crop_size, self.crop_size, 3), dtype='uint8')  # [4,107,107,3]
        img_idx = 0
        for i, sample in enumerate(samples):
            if (i % 4 == 0 and isPos) or (i % 12 == 0 and isNeg):
                image = np.asarray(cv2.imread(images[img_idx])[:, :, ::-1])
                img_idx += 1
            regions[i] = crop_image2(image, sample, 
                                     self.crop_size, self.padding, self.flip, self.rotate, self.blur)
        regions = regions.transpose(0, 3, 1, 2)  # [4,107,107,3] -> [4,3,107,107]
        regions = regions.astype('float32') - 128.
        return regions


class RegionExtractor(Dataset):
    def __init__(self, image_v, image_i, samples):
        self.image_v = np.asarray(image_v)
        self.image_i = np.asarray(image_i)
        self.samples = samples

        self.crop_size = cfg.DATA.TEST_SIZE
        self.padding = cfg.DATA.PADDING
        self.batch_size = cfg.DATA.TEST_BATCH_SIZE

        self.index = np.arange(len(samples))
        self.pointer = 0

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer
            regions_v, regions_i = self.extract_regions(index)
            regions_v = torch.from_numpy(regions_v)
            regions_i = torch.from_numpy(regions_i)
            return regions_v, regions_i

    next = __next__

    def extract_regions(self, index):
        regions_v = np.zeros((len(index), self.crop_size, self.crop_size, 3), dtype='uint8')
        regions_i = np.zeros((len(index), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(self.samples[index]):
            regions_v[i] = crop_image2(self.image_v, sample, self.crop_size, self.padding)
            regions_i[i] = crop_image2(self.image_i, sample, self.crop_size, self.padding)
        regions_v = regions_v.transpose(0, 3, 1, 2).astype('float32') - 128.
        regions_i = regions_i.transpose(0, 3, 1, 2).astype('float32') - 128.
        return regions_v, regions_i


def init_datasets():
    # Init dataset
    img_home = os.path.join(cfg.DATA.DATA_PATH, cfg.DATA.TRAIN_DATASET)
    dataset_list_path = os.path.join('./data/dataset_list/', cfg.DATA.TRAIN_DATASET, cfg.TRAIN.CHALLENGE + '.pkl')
    with open(dataset_list_path, 'rb') as fp:
        data = pickle.load(fp)
    len_data = len(data)
    dataset = [None] * len_data
    for k, seq in enumerate(data.values()):
        for key, value in seq.items():
            if key != 'gt':
                for i in range(len(seq[key])):
                    seq[key][i] = img_home + '/' + value[i]
        dataset[k] = RegionDataset(seq['images_v'], seq['images_i'], seq['gt'])
    
    return dataset