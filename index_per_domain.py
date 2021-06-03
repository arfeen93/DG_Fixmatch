import sys
sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
from sklearn_extra.cluster import KMedoids

import os
import random

from dataloader.dataloader import random_split_dataloader_init
from util.util import *
from copy import deepcopy
device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")


def split_domain(domains, split_idx, print_domain=True):
    source_domain = deepcopy(domains)
    target_domain = [source_domain.pop(split_idx)]
    if print_domain:
        print('Source domain: ', end='')
        for domain in source_domain:
            print(domain, end=', ')
        print('Target domain: ', end='')
        for domain in target_domain:
            print(domain)
    return source_domain, target_domain


domain_map = {
    'PACS': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'PACS_random_split': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'OfficeHome': ['Art', 'Clipart', 'Product', 'RealWorld'],
    'VLCS': ['Caltech', 'Labelme', 'Pascal', 'Sun']
}


def get_domain(name):
    if name not in domain_map:
        raise ValueError('Name of dataset unknown %s' % name)
    return domain_map[name]

ROOT_DIR = "../"
data_root = f"{ROOT_DIR}/PACS/kfold/"
domain_samples = 170
RANDOM_STATE = 42

INDICES_DIR = f"{ROOT_DIR}/saved-indices"
INDICES_PATH = f"{INDICES_DIR}/indices_final_train.pt"

exp_nums = 0

