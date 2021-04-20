from torch.utils.data import Dataset
import sys
import os
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
    def __init__(self, root_dir, domain, split,labelling, get_domain_label=False, get_cluster=False, color_jitter=True, min_scale=0.8):
        self.root_dir = root_dir
        self.domain = domain
        self.split = split
        self.labelling = labelling
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
        #if not self.split == "randaugment":
       # print(self.transform)
        #if self.split == "randaugment":
        #    self.transform = TransformFix("randaugment", "alexnet", mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        image = self.transform(image)
        output = [image, target]        
        
        if self.get_domain_label:
            domain = np.copy(self.domains[index])
            domain = np.int64(domain)
            output.append(domain)
            
        if self.get_cluster:
            cluster = np.copy(self.clusters[index])
            cluster = np.int64(cluster)
            output.append(cluster)

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
        total_domain_samples = []
        two_domain_all_samples = []
        self.domains = np.zeros(0)
        dom_lab_all = np.zeros(0)
        two_domains= np.zeros(0)
        one_domain_indice=[]
        all_total_samples =[]
        all_unlbl_indices=[]
        total_dom_indice =0
        
        classes, class_to_idx = self.find_classes(self.root_dir + self.domain[0] + '/')
        self.num_class = len(classes)
      
        indice = torch.load('/home/arfeen/papers_code/dom_gen_aaai_2020/dg_mmld-master/indices_final.pt')['indices']
        #print(indice)
        
        #indices_mixed = torch.load ('/home/arfeen/papers_code/dom_gen_aaai_2020/saved-indices/mixed_domain_clustering_indices_all_indices.pt')['indices']
        for i, item in enumerate(self.domain):
            path = self.root_dir + item + '/' 
            samples = make_dataset(path, class_to_idx, IMG_EXTENSIONS)
            all_total_samples.extend(samples)
            
            #if self.split == 'val_lbl':
            if self.split == 'val':
                sampled_indices = indice[item]['clustering_indices']
                #print('len of sampled indices', len(sampled_indices))
               
                
                unlbl_indice = [k+total_dom_indice for k in range(len(samples)) if k not in sampled_indices]
                
                all_unlbl_indices.extend(unlbl_indice)
                #print('length of unlabelled indices :', len(all_unlbl_indices))
                total_dom_indice+=len(samples)
                
                if self.labelling == 'lbl':
                    domain_samples = [samples[i] for i in sampled_indices]
                    total_domain_samples.extend(domain_samples)
                    #total_domain_samples['labelled'].extend(domain_samples)
                    self.domains = np.append(self.domains, np.ones(len(domain_samples)) * i)
                    total_samples.extend(domain_samples)
                    print('length of domain wise combined samples', len(total_domain_samples))

            if self.split == 'test':
                self.domains = np.append(self.domains, np.ones(len(samples)) * i)
                total_samples.extend(samples)
            #total_samples.extend(total_domain_samples)
        
        
        #if self.split == 'val_unlbl':
        if self.split == 'val' and self.labelling == 'unlbl':
            total_samples.extend(samples)
            #mixed_sampled_indices = indices_mixed['mixed_{}_source_domain_indices'.format(str(self.domain)[1:-1])]['random_indices']
            mixed_sampled_indices = all_unlbl_indices
            print('length of unlabelled indices outside :', len(all_unlbl_indices))
            mixed_samples = [all_total_samples[i] for i in mixed_sampled_indices]
            #if self.labelling == 'unlbl':

            total_samples.clear()
            total_samples.extend(total_domain_samples) 
            #total_samples.update(total_domain_samples) 
            print('length of mixed samples', len(mixed_samples))
            self.domains = np.append(self.domains, np.ones(len(mixed_samples)))
            #total_mixed_samples['unlabelled'].extend(mixed_samples)
            total_samples.extend(mixed_samples)
            #total_samples.update(total_mixed_samples)
            print('length of all combined samples', (len(total_samples)))
        
        
       
        self.clusters = np.zeros(len(self.domains), dtype=np.int64)
        self.images = [s[0] for s in total_samples]
        self.labels = [s[1] for s in total_samples]
        self.samples = samples

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
