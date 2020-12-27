from torch.utils.data import DataLoader, random_split, ConcatDataset
import numpy as np
from copy import deepcopy
from dataloader.Dataset import DG_Dataset
import torch
import torchvision
from torchvision.utils import save_image
import matplotlib.pyplot as plt 
from torchvision.utils import make_grid

def random_split_dataloader (data, data_root, source_domain, target_domain, batch_size,
                   get_domain_label=False, get_cluster=False, num_workers=4, color_jitter=True, min_scale=0.8):
    if data=='VLCS': 
        split_rate = 0.7
    else: 
        split_rate = 0.9
    source_lbl_train = DG_Dataset(root_dir=data_root, domain=source_domain, split='val',labelling='lbl',
                                     get_domain_label=False, get_cluster=False, color_jitter=color_jitter, min_scale=min_scale)
    # print(' source label train data :', list(source_lbl_train))
    source_unlbl = DG_Dataset(root_dir=data_root, domain=source_domain, split='val', labelling='unlbl',
                                     get_domain_label=False, get_cluster=False, color_jitter=color_jitter, min_scale=min_scale)
    # source_train, source_val = random_split(source, [int(len(source)*split_rate), len(source)-int(len(source)*split_rate)])
    # print('source_lbl_train length :', len(source_lbl_train))
    # print('source_unlbl_train length :', len(source_unlbl))
    
    # Labelled training and validation setup
    # random_seed_lbl = 0
    # torch.manual_seed(random_seed_lbl)
    
    # source_lbl_train, source_lbl_val = random_split(source_lbl, [int(len(source_lbl)*split_rate), len(source_lbl)-int(len(source_lbl)*split_rate)])

    
    # unabelled training and validation setup
    seed = 1
    torch.manual_seed(seed)
    source_unlbl_train, source_unlbl_val = random_split( source_unlbl, [int(len( source_unlbl)*split_rate), len( source_unlbl)-int(len( source_unlbl)*split_rate)])
    
    # torch.save(source_lbl_val, 'source_lbl_val_{}.pt'.format(t))
    # torch.save(source_unlbl_val, 'source_unlbl_val_{}.pt'.format(t))
   
    # source_train = deepcopy(source_train)
    source_lbl_train = deepcopy(source_lbl_train)
    source_unlbl_train = deepcopy(source_unlbl_train)
    # print('source_lbl_train without augmentation :', list(source_lbl_train)[0])
    source_val = source_unlbl_val
    # source_val = ConcatDataset([source_lbl_val, source_unlbl_val])
    
    source_unlbl_train.dataset.split='randaugment'
    # source_unlbl_train.dataset.set_transform('train')
    # print("Doing unlabelled RandAugment")
    source_unlbl_train.dataset.set_transform('randaugment')
    source_unlbl_train.dataset.get_domain_label = get_domain_label
    # source_lbl_train.dataset.get_cluster=get_cluster
    source_unlbl_train.dataset.get_cluster = get_cluster
   
    
    
    # source_lbl_train.dataset.split='randaugment'
    source_lbl_train.split='randaugment'
    # source_lbl_train.dataset.set_transform('train')
    # print("Doing labelled RandAugment")
    # source_lbl_train.dataset.set_transform('randaugment')
    source_lbl_train.set_transform('randaugment')
    # source_lbl_train.dataset.get_domain_label=get_domain_label
    source_lbl_train.get_domain_label = get_domain_label
    # source_lbl_train.dataset.get_cluster=get_cluster
    source_lbl_train.get_cluster = get_cluster

    
    # print('source_lbl_train with augmentation :', list(source_lbl_train)[0])
    # print('source_lbl_train data :',list(source_lbl_train)[0][0][0])
    # source_train, source_val = random_split(source, [int(len(source)*split_rate), len(source)-int(len(source)*split_rate)])

    target_test =  DG_Dataset(root_dir=data_root, domain=target_domain, split='test',labelling='lbl',
                                   get_domain_label=False, get_cluster=False)
    
    print('target_test length :', len(target_test))
    source_train_len = len(source_lbl_train) + len(source_unlbl_train)
    print('Train: {}, Val: {}, Test: {}'.format(source_train_len, len(source_val), len(target_test)))
    
    # source_train = DataLoader(source_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    
    source_lbl_train_ldr = DataLoader(source_lbl_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # Debugging for augmentation
    for images, trgt_lbl, dom_lbl in source_lbl_train:
        img = images[2]
        fig, ax = plt.subplots(figsize=(12,12))
        ax.set_xticks([]); ax.set_yticks([])
        save_image(make_grid(img[:128], nrow=16), "lbld_data.png")
        break
        
    source_unlbl_train_ldr = DataLoader(source_unlbl_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    for images, trgt_lbl, pseudo_dom_lbl in source_unlbl_train:
        img_w = images[0]
        fig,ax = plt.subplots(figsize=(12,12))
        ax.set_xticks([]); ax.set_yticks([])
        save_image(make_grid(img_w[:128], nrow=16), "unlbld_data_weak_aug.png")
        break
    for images, trgt_lbl, pseudo_dom_lbl in source_unlbl_train:
        img_s = images[1]
        fig,ax = plt.subplots(figsize=(12,12))
        ax.set_xticks([]); ax.set_yticks([])
        save_image(make_grid(img_s[:128], nrow=16), "unlbld_data_strong_aug.png")
        break
    source_val  = DataLoader(source_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    target_test = DataLoader(target_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return source_lbl_train_ldr, source_unlbl_train_ldr, source_val, target_test, source_lbl_train
    # return source_train, source_val, target_test, source_lbl_train
