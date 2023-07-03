import numpy as np
from torch.utils.data import Sampler, Dataset


class newSampler(Sampler):
    def __init__(self, data_source):
        self.lst = np.random.permutation(len(data_source))
        self.data_source = data_source
        
    def __iter__(self):
        return iter(self.lst)

    def __len__(self):
        return len(self.data_source)
    
    def __getitem__(self, index):
        return self.lst[index]


class newDataset(Dataset):
    def __init__(self, dataset, flag='train'):
        super().__init__()
        self.dataset = dataset
        self.flag = flag
        
    def __getitem__(self, index):
        if self.flag == 'train':
            return self.dataset[index].next()
        else:
            return self.dataset.next()
        
    def __len__(self):
        return len(self.dataset)
