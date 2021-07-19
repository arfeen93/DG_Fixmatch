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
from torchvision.datasets.folder import make_dataset, default_loader
from torchvision import transforms
from dataloader.Dataset import TransformFix
def random_split_dataloader(data, data_root, source_domain, target_domain, batch_size, labeled_batch_size,
                   get_domain_label=False, get_cluster=False, num_workers=4, color_jitter=True, min_scale=0.8):

    if data=='VLCS': 
        split_rate = 0.7
    else: 
        split_rate = 0.9

    source = DG_Dataset(root_dir=data_root, domain=source_domain, split='val',
                                     get_domain_label=False, get_cluster=False, color_jitter=color_jitter, min_scale=min_scale)

    #print('get_domain_label:', get_domain_label)
    #print('get_cluster:', get_cluster)


    lbl_indexes, unlbl_indexes, val_indices = source.lbl_unlbl_indexes()
    train_indexes = lbl_indexes + unlbl_indexes
    train_val_indexes = train_indexes + val_indices
    #train_val = Train_val_data(train_val_indexes, root_dir=data_root, domain=source_domain, split='val')
    source_train = Train_val_data(train_indexes, root_dir=data_root, domain=source_domain, split='val')
    source_lbl_train = Train_val_data(lbl_indexes, root_dir=data_root, domain=source_domain, split='val')
    #print("source_train.clss len:", len(source_train.clss))
    train_lbl_indices = [x for x in range(len(lbl_indexes))]
    train_unlbl_indices = [x for x in range(len(lbl_indexes), len(train_indexes))]
    source_train.split='randaugment'
    source_train.set_transform('randaugment')
    source_train.get_domain_label = get_domain_label
    source_train.get_cluster = get_cluster
    source_lbl_train.get_domain_label=get_domain_label
    source_lbl_train.get_cluster=get_cluster

    source_val = Train_val_data(val_indices, root_dir=data_root, domain=source_domain, split='val')
    # source_val.split='randaugment'
    # source_val.set_transform('randaugment')


    # trgt = []
    # for i, data in enumerate(source_val):
    #     sample_cls = list(source_val)[i][1]
    #     trgt.append(sample_cls)
    # print("0 count", trgt.count(0))
    # print("1 count", trgt.count(1))
    # print("2 count", trgt.count(2))
    # print("3 count", trgt.count(3))
    # print("4 count", trgt.count(4))
    # print("5 count", trgt.count(5))
    # print("6 count", trgt.count(6))

    target_test = DG_Dataset(root_dir=data_root, domain=target_domain, split='test',
                             get_domain_label=False, get_cluster=False)
    print('Train: {}, Val: {}, Test: {}'.format(len(source_train), len(source_val), len(target_test)))
    # source_train, source_val = random_split(source, [int(len(source)*split_rate), len(source)-int(len(source)*split_rate)],
    #                                         generator=torch.Generator().manual_seed(10))
    #print(dir(source_train))
    #train_indices = source_train.indices
    #print("train_indices :", (train_indices))
    #val_indices = source_val.indices
    #print("val_indices :", (val_indices))
    # flag = 0
    # if (all(x in train_indices for x in lbl_indexes)):
    #     flag = 1
    # if (flag):
    #     print("subset")
    # else:
    #     print("not subset")

    batch_sampler = TwoStreamBatchSampler(train_unlbl_indices, train_lbl_indices, batch_size, labeled_batch_size)


    source_train = deepcopy(source_train)

    # source_train.dataset.split='randaugment'
    # source_train.dataset.set_transformation('randaugment')
    # source_train.dataset.get_domain_label = get_domain_label
    # source_train.dataset.get_cluster = get_cluster

    #source_train = DataLoader(source_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    source_train = DataLoader(source_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True)

    #print(("hvdnvlmv:", source_train.dataset.num_class))

    target_test =  DG_Dataset(root_dir=data_root, domain=target_domain, split='test',
                                   get_domain_label=False, get_cluster=False)
    
    #print('target_test length :', len(target_test))
    #print('Train: {}, Val: {}, Test: {}'.format(len(source_train), len(source_val), len(target_test)))
    

    
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
    #print("source_val dataloader len :", len(source_val))
    target_test = DataLoader(target_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return source_train, source_lbl_train, source_val, target_test #, train_val


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
            in zip(grouper(secondary_iter, self.secondary_batch_size),
                   grouper(primary_iter, self.primary_batch_size))
        )
        # return (
        #     primary_batch + secondary_batch
        #     for (primary_batch, secondary_batch)
        #     in  zip(grouper(primary_iter, self.primary_batch_size),
        #             grouper(secondary_iter, self.secondary_batch_size))
        # )

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



"-----------------------------------------------"
"Class for train and val set for respective indexes"
class Train_val_data(DG_Dataset):
    def __init__(self, indexes, root_dir, domain, split):
        super(Train_val_data, self).__init__(root_dir, domain, split
                                             , get_domain_label=False, get_cluster=False, color_jitter=True, min_scale=0.8 )
        self.indexes = indexes
        self.dataload()
        #clusters=self.clusters

    def __getitem__(self, index):
        image_path = self.images[index]
        image_data = self.loader(image_path)
        image_data = self.transform(image_data)
        target = self.clss[index]

        output_data = [image_data, target]
        if self.get_domain_label:
            #print("inside self.get_domain_label")
            domain_lbl = np.copy(self.domains_lbl[index])
            domain_lbl = np.int64(domain_lbl)
            output_data.append(domain_lbl)
        if self.get_cluster:
            #print("inside self.get_cluster")
            clusters_lbl = np.copy(self.cluster_lbl[index])
            clusters_lbl = np.int64(clusters_lbl)
            output_data.append(clusters_lbl)
        #print("outputdata:", tuple(output_data))
        return  tuple(output_data)

    def set_cluster_index(self, cluster_list):
        if len(cluster_list) != len(self.images):
            raise ValueError("The length of cluster_list must to be same as self.images")
        else:
            self.cluster_lbl = cluster_list

    def set_domain_index(self, domain_list):
        if len(domain_list) != len(self.images):
            raise ValueError("The length of domain_list must to be same as self.images")
        else:
            self.domains_lbl = domain_list

    def dataload(self):
        image = [s[0] for s in self.total_samples]
        self.images = [image[idx] for idx in self.indexes]
        cls = [s[1] for s in self.total_samples]
        self.clss = [cls[idx] for idx in self.indexes]
        self.cluster_lbl = [self.clusters[idx] for idx in self.indexes]
        self.cluster_lbl = np.array(self.cluster_lbl)
        self.domains_lbl = [self.domains[idx] for idx in self.indexes]
        self.domains_lbl = np.array(self.domains_lbl)
