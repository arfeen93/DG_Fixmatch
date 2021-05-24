from torch.utils.data import DataLoader, random_split, ConcatDataset
import numpy as np
from copy import deepcopy
from dataloader.Dataset import DG_Dataset
import torch
import torchvision
from torchvision.utils import save_image
import matplotlib.pyplot as plt 
from torchvision.utils import make_grid
from torch.utils.data.sampler import Sampler
import itertools
def random_split_dataloader (data, data_root, source_domain, target_domain, batch_size, labeled_batch_size,
                   get_domain_label=False, get_cluster=False, num_workers=4, color_jitter=True, min_scale=0.8):

    if data=='VLCS': 
        split_rate = 0.7
    else: 
        split_rate = 0.9
    source = DG_Dataset(root_dir=data_root, domain=source_domain, split='val',
                                     get_domain_label=False, get_cluster=False, color_jitter=color_jitter, min_scale=min_scale)
    lbl_indexes, unlbl_indexes = source.lbl_unlbl_indexes()

    seed = 10
    torch.manual_seed(seed)
    source_train, source_val = random_split(source, [int(len(source)*split_rate), len(source)-int(len(source)*split_rate)])
    batch_sampler = TwoStreamBatchSampler(unlbl_indexes, lbl_indexes, batch_size, labeled_batch_size)


    source_train = deepcopy(source_train)
    
    source_train.dataset.split='randaugment'
    source_train.dataset.set_transform('randaugment')
    source_train.dataset.get_domain_label = get_domain_label
    source_train.dataset.get_cluster = get_cluster


    target_test =  DG_Dataset(root_dir=data_root, domain=target_domain, split='test',
                                   get_domain_label=False, get_cluster=False)
    
    print('target_test length :', len(target_test))
    print('Train: {}, Val: {}, Test: {}'.format(len(source_train), len(source_val), len(target_test)))
    
    #source_train = DataLoader(source_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    source_train = DataLoader(source_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True)
    
    # Debugging for augmentation
    # for images, trgt_lbl, dom_lbl in source_lbl_train:
    #     img = images[2]
    #     fig, ax = plt.subplots(figsize=(12,12))
    #     ax.set_xticks([]); ax.set_yticks([])
    #     save_image(make_grid(img[:128], nrow=16), "lbld_data.png")
    #     break
    #print("source_unlbl_train_ldr is:",(list(source_unlbl_train_ldr)[0]))
    # for images, trgt_lbl, pseudo_dom_lbl in source_unlbl_train:
    #     img_w = images[0]
    #     fig,ax = plt.subplots(figsize=(12,12))
    #     ax.set_xticks([]); ax.set_yticks([])
    #     save_image(make_grid(img_w[:128], nrow=16), "unlbld_data_weak_aug.png")
    #     break
    # for images, trgt_lbl, pseudo_dom_lbl in source_unlbl_train:
    #     img_s = images[1]
    #     fig,ax = plt.subplots(figsize=(12,12))
    #     ax.set_xticks([]); ax.set_yticks([])
    #     save_image(make_grid(img_s[:128], nrow=16), "unlbld_data_strong_aug.png")
    #     break
    source_val  = DataLoader(source_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    target_test = DataLoader(target_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return source_train, source_val, target_test


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)