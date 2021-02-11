from torch import nn
from util.util import split_domain
import torch
from numpy.random import *
import numpy as np
from loss.EntropyLoss import HLoss
from loss.MaximumSquareLoss import MaximumSquareLoss
import torch.nn.functional as F
import random
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid



def train(model, source_lbl_train_ldr, source_unlbl_train_ldr, source_lbl_train,
          optimizers, device, epoch, num_epoch, filename, entropy, alpha_mixup, disc_weight=None, entropy_weight=1.0, grl_weight=1.0):
    class_criterion = nn.CrossEntropyLoss()
    # print(disc_weight)
    # print('source_unlbl_train_ldr are :', list(source_unlbl_train_ldr))
    domain_criterion = nn.CrossEntropyLoss(weight=disc_weight)
    if entropy == 'default':
        entropy_criterion = HLoss()
    else:
        entropy_criterion = MaximumSquareLoss()
    p = epoch / num_epoch
    alpha = (2. / (1. + np.exp(-10 * p)) -1) * grl_weight
    beta = (2. / (1. + np.exp(-10 * p)) -1) * entropy_weight
    model.discriminator.set_lambd(alpha)
    model.train()  # Set model to training mode
    running_loss_class = 0.0
    running_correct_class = 0
    running_loss_domain = 0.0
    running_correct_domain = 0
    running_loss_entropy = 0
    running_total_loss = 0.0
    # Iterate over data.
    labeled_iter = iter(source_lbl_train_ldr)
    # print(' train_lbl_data loader :', list(train_lbl_data))
    unlabeled_iter = iter(source_unlbl_train_ldr)
    # for idx, (batch_lbl,batch_unlbl) in enumerate(train_data):
    train_step = 100
    total_unlbl = 0
    total_lbl = 0
    for batch_idx in range(train_step): 
        if batch_idx % len(source_lbl_train_ldr) == 0:
             labeled_iter = iter(source_lbl_train_ldr)

        if batch_idx % len(source_unlbl_train_ldr) == 0:
             unlabeled_iter = iter(source_unlbl_train_ldr)

        input_x, target_x, lbl_dataldr_domain_lbl = next(labeled_iter)
        #     print("label data finished")
        lbl_input = input_x[2]
        labels = target_x
        lbld_domains = lbl_dataldr_domain_lbl
        lbl_input = lbl_input.to(device)
        labels = labels.to(device)
        lbld_domains = lbld_domains.to(device)

        input_w, input_lbl, unlbl_dataldr_dom_lbl = next(unlabeled_iter)
        input_weak = input_w[0]
        input_strong = input_w[1]
        input_strong = input_strong.to(device)
        input_weak = input_weak.to(device)
        unlbl_domains = unlbl_dataldr_dom_lbl
        unlbl_domains = unlbl_domains.to(device)
        input_lbl = input_lbl.to(device)
        # input_weak = batch_unlbl[0][0].to(device)
        # input_strong = batch_unlbl[0][1].to(device)

        # zero the parameter gradients
        for optimizer in optimizers:
            optimizer.zero_grad()
        # forward - do pseudo labelling part here
        pred_weak_aug, output_domain = model(input_weak)
        pred_strong_aug, output_domain = model(input_strong)

        prob_weak_aug = F.softmax(pred_weak_aug, dim=1)
        prob_strong_aug = F.softmax(pred_strong_aug, dim=1)
        
        # print('value of prob_weak_aug :', (prob_weak_aug))
        # Considering only the examples which have confidence above a certain threshold
        mask_loss = prob_weak_aug.max(1)[0] > 0.9
        # print('mask_loss :', mask_loss))
        
        # count = 0
        # for i in mask_loss:
        #     if i == True:
        #         count += 1
        # print('count of true mask_loss :', count)
        pseudo_labels_orig = pred_weak_aug.max(axis=1)[1]
        # print('pseudo_labels_orig are :', pseudo_labels_orig)
        pseudo_labels = F.one_hot(pseudo_labels_orig, num_classes=7)


        #lam = lam.to(device)
        #alpha_mixup = alpha_mixup.to(device)
        #source_lbl_train = source_lbl_train.to(device)
        '''doing mixup of strong_aug unlbled data and lbled input of same class'''
        # mixedup, lbl_dom, unlbl_dom, lam = mixup(device, input_strong, source_lbl_train,  pseudo_labels_orig, target_x,
        #                                            lbld_domains, unlbl_domains, alpha_mixup, batch_size=128)
        '''doing mixup of strong_aug unlbled data as per mixup paper'''
        mixedup, input_clss_lbl, input_clss_lbl_idx, unlbl_dom, unlbl_dom_idx, lam = mixup(device, input_strong, input_lbl,
                                                      unlbl_domains, alpha_mixup, batch_size=128)

        #lbl_dom = lbl_dom.to(device)
        #unlbl_dom = unlbl_dom.to(device)
        #mixedup = mixedup.to(device)

        mixedup_output, mixup_output_domain = model(mixedup)
        prob_mixup_output = F.softmax(mixedup_output, dim=1)
        # loss_pseudo_unl = -torch.mean((mask_loss.int())*torch.sum(pseudo_labels * (torch.log(prob_strong_aug + 1e-5)), 1)) # pseudo label loss
        # mixup domain loss
        # mixup_domain_loss = (1 - lam) * domain_criterion(mixup_output_domain, lbl_dom) + lam * domain_criterion(
        #                                                                               mixup_output_domain, unlbl_dom)
        loss_pseudo_unl = -torch.mean((mask_loss.int()) * torch.sum(pseudo_labels * (torch.log(prob_mixup_output + 1e-5)), 1))
        mixup_domain_loss = (1 - lam) * domain_criterion(mixup_output_domain, unlbl_dom_idx) + lam * domain_criterion(
            mixup_output_domain, unlbl_dom)

        mixup_cls_loss = (1 - lam) * class_criterion(mixedup_output, input_clss_lbl_idx) + lam * class_criterion(
            mixedup_output, input_clss_lbl)
        #print(loss_pseudo_unl.cpu().data)
        # print('mixup_domain_loss :', mixup_domain_loss)

        # pseudo label loss
        # loss_pseudo_unl.backward(retain_graph=True)
        # for optimizer in optimizers:
        #    optimizer.step()
        inputs = lbl_input
        inputs = inputs.to(device)

        output_class, output_domain = model(inputs)

        loss_class_lbl = class_criterion(output_class, labels)
        loss_class = loss_class_lbl + loss_pseudo_unl + mixup_cls_loss
        loss_domain_lbl = domain_criterion(output_domain, lbld_domains)
        loss_domain = loss_domain_lbl + mixup_domain_loss

        loss_entropy_lbl = entropy_criterion(output_class)
        loss_entropy_unlbl = entropy_criterion(mixedup_output)
        loss_entropy = loss_entropy_lbl + loss_entropy_unlbl
        _, lbl_pred_class = torch.max(output_class, 1)
        _, lbl_pred_domain = torch.max(output_domain, 1)

        _, unlbl_pred_class = torch.max(mixedup_output, 1)
        _, unlbl_pred_domain = torch.max(mixup_output_domain, 1)
        # print('beta in total loss :', beta)

        total_loss = loss_class + loss_domain + loss_entropy * beta

        # print('total_loss :', total_loss)

        total_loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        
        # running_loss_class += loss_class.item() * inputs.size(0)
        running_loss_class += loss_class.item()
        total_correct_class = torch.sum(lbl_pred_class == labels.data) + torch.sum(unlbl_pred_class == input_lbl.data)

        running_correct_class += total_correct_class
        # running_loss_domain += loss_domain.item() * inputs.size(0)
        running_loss_domain += loss_domain.item()
        total_correct_domain = torch.sum(lbl_pred_domain == lbld_domains.data) + torch.sum(unlbl_pred_domain == unlbl_domains.data)
        running_correct_domain += total_correct_domain

        # running_loss_entropy += loss_entropy.item() * inputs.size(0)
        # running_total_loss += total_loss.item() * inputs.size(0)
        running_loss_entropy += loss_entropy.item()
        running_total_loss += total_loss.item()
        total_unlbl +=  mixedup.size(0)
        total_lbl += inputs.size(0)

    # print('total labelled :', total_lbl)
    # print('total unlabelled :', total_unlbl)
    # train_data_len = len(source_lbl_train_ldr.dataset)+len(source_unlbl_train_ldr.dataset)
    total = total_unlbl + total_lbl
    epoch_loss_class = running_loss_class / train_step
    epoch_acc_class = running_correct_class.double() / total
    epoch_loss_domain = running_loss_domain / train_step

    epoch_acc_domain = running_correct_domain.double() / total
    epoch_loss_entropy = running_loss_entropy / train_step

    epoch_total_loss = running_total_loss / train_step
    log = 'Train: Epoch: {} Alpha: {:.4f} Loss Class: {:.4f} Acc Class: {:.4f}, Loss Domain: {:.4f} Acc Domain: {:.4f} ' \
          'Loss Entropy: {:.4f} '.format(epoch, alpha, epoch_loss_class, epoch_acc_class, epoch_loss_domain,
                                     epoch_acc_domain, epoch_loss_entropy)
    print(log)
    with open(filename, 'a') as f: 
        f.write(log + '\n') 
    return model, optimizers

# Returns mixed inputs of same class
def mixup(device, input_strong_ldr, input_clss_lbl, unlbl_domains, alpha_mixup, batch_size) :
    # all_lbl_images_order = []
    # all_lbl_images = []
    # all_targets = []
    # all_dom_lab = []
    # lbl_domain_order = []
    # print('input_strong_ldr is :', input_strong_ldr.shape)
    # print('alpha_mixup :', alpha_mixup)
    if alpha_mixup > 0:
        lam = np.random.beta(alpha_mixup, alpha_mixup)
    else:
        lam = 1
    """mixup of same class--- start"""
    # i = 0
    # for lbl_inputs, lbl_targets, dom_lbl in train_lbl_data:
    #     # print('')
    #
    #     lbl_images = lbl_inputs[2]
    #     all_lbl_images.append(lbl_images)
    #     all_targets.append(lbl_targets)
    # # print('all_lbl_images :', all_lbl_images)
    #     all_dom_lab.append(dom_lbl)
    # for lbl in pseudo_labels_orig:
    #     lbl = lbl.item()
    #     # print('lbl is :', lbl)
    #     indexes = [i for i, x in enumerate(all_targets) if x == lbl]
    #     index = random.choice(indexes)
    #     # index = all_targets.index(lbl)
    #     # print('index is :', index)
    #     all_lbl_images_order.append(all_lbl_images[index])
    #     lbl_domain_order.append(all_dom_lab[index])

    #lbl_domain_order = torch.tensor(lbl_domain_order)
    # print('lbl_domain_order :', lbl_domain_order)
    # print('unlbl_domains :', unlbl_domains)
    #all_lbl_images_order = torch.stack(all_lbl_images_order)
    #all_lbl_images_order = all_lbl_images_order.to(device)
    # print('all_lbl_images_order is :', all_lbl_images_order.shape)
    # mixed_input_ldr = lam * input_strong_ldr + (1 - lam) * all_lbl_images_order
    """mixup of same class--- end"""
    """mixup irrespective of classes(as per paper code)--- start"""
    index = torch.randperm(len(input_strong_ldr))
    index = index.to(device)
    input_clss_lbl_idx = input_clss_lbl[index]
    unlbl_domains_idx = unlbl_domains[index]
    mixed_input_ldr = lam * input_strong_ldr + (1-lam) * input_strong_ldr[index, :]
    # lbl_dom, unlbl_dom = lbl_domain_order, unlbl_domains
    # lbl_dom = lbl_dom.to(device)
    return mixed_input_ldr, input_clss_lbl, input_clss_lbl_idx, unlbl_domains, unlbl_domains_idx, lam








