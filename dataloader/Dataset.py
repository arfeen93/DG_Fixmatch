from torch.utils.data import Dataset
import sys
import os
import random
from torchvision import transforms
from torchvision.datasets.folder import make_dataset, default_loader
import numpy as np
import torch
from copy import deepcopy
from .randaugment import RandAugmentMC
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class TransformFix(object):
    def __init__(self, aug_policy, net, mean, std):
        self.net = net
        self.aug_policy = aug_policy
        if self.net == 'caffenet':
            self.crop_size = 227
        else:
            self.crop_size = 224
            #self.crop_size = 227
        #print("crop size is:", self.crop_size)
        # Might want to add resize image as done in other transforms
        self.weak = transforms.Compose([
           # self.resize(256,256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=self.crop_size,
                                  padding=int(self.crop_size*0.125),
                                  padding_mode='reflect')])
        if self.aug_policy == "randaugment":
            self.strong = transforms.Compose([
               # ResizeImage(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=self.crop_size,
                                    padding=int(self.crop_size*0.125),
                                    padding_mode='reflect'),
                RandAugmentMC(n=3, m=10)])

        self.standard = transforms.Compose([
            #transforms.resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
   # @staticmethod
    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.standard(x)


class DG_Dataset(Dataset):
    def __init__(self, root_dir, domain, split, get_domain_label=False, get_cluster=False, color_jitter=True, min_scale=0.8):
        self.root_dir = root_dir
        self.domain = domain
        self.split = split
        self.get_domain_label = get_domain_label
        self.get_cluster = get_cluster
        self.color_jitter = color_jitter
        self.min_scale = min_scale
        self.set_transform(self.split)
        self.loader = default_loader
        #self.TransformFix = TransformFix 
        self.load_dataset()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        image = self.loader(path)
        image = self.transform(image)
        output = [image, target]        
        
        if self.get_domain_label:
            domain = np.copy(self.domains[index])
            domain = np.int64(domain)
            output.append(domain)
            
        if self.get_cluster:
            cluster = np.copy(self.clusters[index])
            #print("clusters:", cluster)
            cluster = np.int64(cluster)
            output.append(cluster)
        #print("tuple output is :", tuple(output))
        return tuple(output)
    
    def find_classes(self, dir_name):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir_name) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def load_dataset(self):
        total_samples = []
        self.domains = np.zeros(0)

        classes, class_to_idx = self.find_classes(self.root_dir + self.domain[0] + '/')
        self.num_class = len(classes)
        for i, item in enumerate(self.domain):
            path = self.root_dir + item + '/'
            samples = make_dataset(path, class_to_idx, IMG_EXTENSIONS)
            total_samples.extend(samples)
            self.domains = np.append(self.domains, np.ones(len(samples)) * i)
        #print("self.domains:",self.domains)
        self.clusters = np.zeros(len(self.domains), dtype=np.int64)
        #print("self.clusters:", self.clusters)
        self.images = [s[0] for s in total_samples]
        self.labels = [s[1] for s in total_samples]
        self.total_samples = total_samples

    def set_cluster(self, cluster_list):
        if len(cluster_list) != len(self.images):
            raise ValueError("The length of cluster_list must to be same as self.images")
        else:
            self.clusters = cluster_list

    def set_domain(self, domain_list):
        if len(domain_list) != len(self.images):
            raise ValueError("The length of domain_list must to be same as self.images")
        else:
            self.domains = domain_list
            
    def set_transform(self, split):
        if split == 'train':
            if self.color_jitter:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(self.min_scale, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(.4, .4, .4, .4),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(self.min_scale, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        elif split == 'val' or split == 'test':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif split == 'randaugment':
            self.transform = TransformFix("randaugment", "caffenet", mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        else:
            raise Exception('Split must be train or val or test!!')





    def lbl_unlbl_indexes(self):
        all_total_samples = []
        all_unlbl_indices = []
        all_lbl_indices = []
        all_val_indices = []
        total_dom_indice = 0

        classes, class_to_idx = self.find_classes(self.root_dir + self.domain[0] + '/')
        self.num_class = len(classes)
        """indexes using Clustering method---start """
        # indice = torch.load('/home/arfeen/papers_code/dom_gen_aaai_2020/dg_mmld-master/PACS_indices_final_510.pt')['indices']  #<--- for 170 samples per domain for caffenet
        # #indice = torch.load('/home/arfeen/papers_code/dom_gen_aaai_2020/dg_mmld-master/PACS_indices_final_210.pt')['indices']  #<--- for 70 samples per domain for caffenet
        # # print(indice)
        # #print("domains are:", self.domain)
        # # indices_mixed = torch.load ('/home/arfeen/papers_code/dom_gen_aaai_2020/saved-indices/mixed_domain_clustering_indices_all_indices.pt')['indices']
        # for i, item in enumerate(self.domain):
        #     # print(item)
        #     path = self.root_dir + item + '/'
        #     samples_data = make_dataset(path, class_to_idx, IMG_EXTENSIONS)
        #     all_total_samples.extend(samples_data)
        #     sampled_indices = indice[item]['clustering_indices']
        #     # print('len of sampled indices', len(sampled_indices))
        #
        #     label_indice = [x + total_dom_indice for x in sampled_indices]
        #     unlbl_indice = [k + total_dom_indice for k in range(len(samples_data)) if k not in sampled_indices]
        #
        #     all_unlbl_indices.extend(unlbl_indice)
        #     all_lbl_indices.extend(label_indice)
        #     # print('length of unlabelled indices :', len(all_unlbl_indices))
        #     total_dom_indice = total_dom_indice + len(samples_data)
        """indexes using Clustering method---end """

        """picking 10 random indexing per class for labelled data---start """
        for i, item in enumerate(self.domain):
            # print(item)
            path = self.root_dir + item + '/'
            samples_data = make_dataset(path, class_to_idx, IMG_EXTENSIONS)
            sample_labels = [data[1] for data in samples_data]
            for j in range(self.num_class):
                indexes = [idx for idx, k in enumerate(sample_labels) if k==j]
                label_indice = random.sample(indexes, 10)
                label_indices = [idx + total_dom_indice for idx in label_indice]
                unlbl_indice = [k + total_dom_indice for k in indexes if k not in label_indice]
                all_lbl_indices.extend(label_indices)
                all_unlbl_indices.extend(unlbl_indice)
                #print('len of label indexes:', len(all_lbl_indices))
            total_dom_indice = total_dom_indice + len(samples_data)

            all_total_samples.extend(samples_data)
        """Using random indexing per class---end """
        # print("all_unlbl_indices:", all_unlbl_indice_train)
        # print("all_lbl_indices:", all_lbl_indices)
        # print("all_val_indices:", val_indice)
        unlbl_len = len(all_unlbl_indices)
        lbl_len = len(all_lbl_indices)
        #print("unlbl_len:", unlbl_len)
        #print("lbl_len:", lbl_len)
        total_source_data = lbl_len + unlbl_len
        #val_indices = random.sample(all_unlbl_indices, np.int64(np.ceil(total_source_data * 0.1)))  #<--- 0.1 for PACS
        val_indices = random.sample(all_unlbl_indices, np.int64(np.ceil(total_source_data * 0.3)))   #<--- 0.3 for VLCS
        #val_indices = random.sample(all_unlbl_indices, 556)
        unlbl_train_indices = [x for x in all_unlbl_indices if x not in val_indices]
        print("no of labelled samples:", len(all_lbl_indices))

        return all_lbl_indices,  unlbl_train_indices, val_indices







