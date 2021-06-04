from torch import nn
from util.util import split_domain
import torch
from numpy.random import *
import numpy as np
from loss.EntropyLoss import HLoss
from loss.MaximumSquareLoss import MaximumSquareLoss
import torch.nn.functional as F
from torch.nn import MSELoss
import random
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def train(model, reg_model, reg_optimizers, source_train, optimizers,
          device, epoch, num_epoch, filename, entropy, alpha_mixup, disc_weight=None, entropy_weight=1.0, grl_weight=1.0, label_batch_size=50):
    class_criterion = nn.CrossEntropyLoss(reduction='mean')
    reg_criterion = nn.BCEWithLogitsLoss()
    # print(disc_weight)
    #print('source_unlbl_train_ldr len are :', len(list(source_unlbl_train_ldr)[0]))
    domain_criterion = nn.CrossEntropyLoss(weight=disc_weight)
    if entropy == 'default':
        entropy_criterion = HLoss()
    else:
        entropy_criterion = MaximumSquareLoss()
    p = epoch / num_epoch
    alpha = (2. / (1. + np.exp(-10 * p)) -1) * grl_weight
    beta = (2. / (1. + np.exp(-10 * p)) -1) * entropy_weight
    # print(f"Beta {beta}, Entropy {entropy_weight}, p {p}")
    model.discriminator.set_lambd(alpha)
    model.train()  # Set model to training mode
    #reg_model.train()
    running_loss_class = 0.0
    running_correct_class = 0
    running_loss_domain = 0.0
    running_correct_domain = 0
    running_loss_entropy = 0
    running_loss_total = 0
    batch_samples = 0
    #print(list(source_train)[0])
    for input_x, target_x, domain_lbl  in source_train:
        #print("lndbsvbdkjvd")
        input_weak = input_x[0]
        input_strong = input_x[1]
        input_std = input_x[2]
        input_std = input_std.to(device)
        input_strong = input_strong.to(device)
        input_weak = input_weak.to(device)
        domains_lbl = domain_lbl.to(device)
        target_x = target_x.to(device)

        for optimizer in optimizers:
            optimizer.zero_grad()
        #reg_optimizers.zero_grad()

        # forward - do pseudo labelling part here
        pred_weak_aug, output_domain = model(input_weak)
        pred_strong_aug, output_domain = model(input_strong)

        prob_weak_aug = F.softmax(pred_weak_aug, dim=1)
        prob_strong_aug = F.softmax(pred_strong_aug, dim=1)

        # print('value of prob_weak_aug :', (prob_weak_aug))
        # Considering only the examples which have confidence above a certain threshold
        mask_loss = prob_weak_aug.max(1)[0] > 0.9
        #print('mask_loss :', mask_loss)

        # count = 0
        # for i in mask_loss:
        #     if i == True:
        #         count += 1
        # print('count of true mask_loss :', count)
        pseudo_labels_orig = pred_weak_aug.max(axis=1)[1]
        #print("pseudo_labels_orig:", pseudo_labels_orig)
        lbl_target = target_x[:label_batch_size]
        unlbl_pseudo_labels_orig = pseudo_labels_orig[label_batch_size:]
        pseudo_labels_mixed = torch.cat((lbl_target, unlbl_pseudo_labels_orig), 0)
        pseudo_labels = F.one_hot(pseudo_labels_mixed, num_classes=7)
        mask_loss_lbl= [True for x in range(len(mask_loss)) if x<label_batch_size]
        mask_loss_lbl = torch.Tensor(mask_loss_lbl)
        mask_loss_lbl = mask_loss_lbl.to(device)
        mask_loss_unlbl = mask_loss[label_batch_size:]
        mask_loss_unlbl = mask_loss_unlbl.to(device)
        mask_loss_all = torch.cat((mask_loss_lbl, mask_loss_unlbl), 0)
        #print('pseudo_labels_orig are :', pseudo_labels_orig)
        #s = torch.sum(pseudo_labels_orig==input_lbl)
        # if batch_idx==0:
        #     print("no of elements same in pseudo and orig is:", s)


        #lam = lam.to(device)
        #alpha_mixup = alpha_mixup.to(device)
        #source_lbl_train = source_lbl_train.to(device)
        '''doing mixup of strong_aug unlbled data and lbled input of same class'''
        # mixedup, lbl_dom, unlbl_dom, lam = mixup(device, input_strong, source_lbl_train,  pseudo_labels_orig, target_x,
        #                                            lbld_domains, unlbl_domains, alpha_mixup, batch_size=128)
        '''doing mixup of strong_aug unlbled data as per mixup paper'''
        mixedup, input_clss_lbl, input_clss_lbl_idx, unlbl_dom, unlbl_dom_idx, mask_loss_all_idx, lam = mixup(device, input_strong, pseudo_labels_mixed,
                                                       domains_lbl, alpha_mixup, mask_loss_all, batch_size=128)

        "mixup for regressor model"
        # mixed_input_std, input_clss_lbl_reg, input_clss_lbl_idx_reg, unlbl_dom_reg, unlbl_dom_idx_reg, lam_reg = mixup(device, input_std, pseudo_labels_orig,
        #                                                unlbl_domains, alpha_mixup, batch_size=128)
        # mixed_input_std = mixed_input_std.to(device)
        # print("lambda for mixup:", lam)
        # lbl_dom = lbl_dom.to(device)
        # unlbl_dom = unlbl_dom.to(device)
        mixedup = mixedup.to(device)

        "Unlabelled data model output---start"
        mixedup_output, mixup_output_domain = model(mixedup)
        "Unlabelled data model output---end"
        "Novelty Regressor model data passing--start"
        # x, x1 = model.features(mixedup)
        # x2, x3 = model.features(input_strong)
        # x, x1 = model.features(mixed_input_std)
        # x2, x3 = model.features(input_std)
        #mixup_unlbl_str_dlr_ftre_cat = torch.cat((x1, x3), 1)
        #mixup_unlbl_str_dlr_cat = torch.cat((mixedup, input_strong), 1)
        #lambda_pred = reg_model(mixup_unlbl_str_dlr_ftre_cat)
        #lambda_pred_cls = torch.round(torch.sigmoid(lambda_pred))

        #lambda_pred_clss = torch.round(lambda_pred)
        #prob_lambda_pred = F.softmax(lambda_pred, dim = 1)
        #lambda_pred = reg_model(mixedup, input_strong)
        "Novelty Regressor model data passing--end"
        prob_mixup_output = F.softmax(mixedup_output, dim=1)
        prob_mixup_domain_output = F.softmax(mixup_output_domain, dim=1)
        input_dom_lbl_one_hot = F.one_hot(unlbl_dom, num_classes=3)
        input_dom_lbl_idx_one_hot = F.one_hot(unlbl_dom_idx, num_classes=3)

        # loss_pseudo_unl = -torch.mean((mask_loss.int())*torch.sum(pseudo_labels * (torch.log(prob_strong_aug + 1e-5)), 1)) # pseudo label loss
        # mixup domain loss
        # mixup_domain_loss = (1 - lam) * domain_criterion(mixup_output_domain, lbl_dom) + lam * domain_criterion(
        #                                                                               mixup_output_domain, unlbl_dom)
        # loss_pseudo_unl = -torch.mean((mask_loss.int()) * torch.sum(pseudo_labels * (torch.log(prob_mixup_output + 1e-5)), 1))
        # pseudo label loss
        #loss_pseudo_unl.backward(retain_graph=True)
        # for optimizer in optimizers:
        #     optimizer.step()
        #for k in range(len(input_clss_lbl)):
        #print("input class lbl:", input_clss_lbl)
        #print("input class lbl idx:", input_clss_lbl_idx)
        # "Novelty regressor part---start"
        # lamdba_gt = [1 if input_clss_lbl[k]==input_clss_lbl_idx[k] else 0 for k in range(len(input_clss_lbl))]
        # #lamdba_gt = [1 if input_clss_lbl_reg[k]==input_clss_lbl_idx_reg[k] else lam for k in range(len(input_clss_lbl_reg))]
        # lamdba_gt = torch.Tensor(lamdba_gt)
        # #print("lambda_gt is:", lamdba_gt)
        # lamdba_gt = lamdba_gt.reshape(lamdba_gt.shape[0], 1)
        # lamdba_gt = lamdba_gt.to(device)

        # abs_reg_loss = torch.abs(lamdba_gt - lambda_pred)
        # rel_reg_loss_ratio = torch.div(abs_reg_loss, lamdba_gt)
        # "--------End"
        #print("lambda ground truth is :", lamdba_gt)
        # if (epoch+1)%10==0:
        #      if batch_idx==2 or batch_idx==5:
        #         print("lambda ground truth is :", lamdba_gt)
        #         print("training lambda predicted class is :", lambda_pred_cls)
        # "Novelty Regressor loss--start"
        # reg_loss = reg_criterion(lambda_pred, lamdba_gt)
        #lambda_correct = torch.max(lambda_pred, 1)
        # mse_loss = MSELoss()  # noveltrain.out is without reduction="sum"
        # reg_loss = mse_loss(lambda_pred, lamdba_gt)
        # l1_loss = nn.L1Loss()
        # reg_loss = l1_loss(lambda_pred, lamdba_gt)

        "Novelty Regressor loss---end"

        # mixup_domain_loss = (1 - lam) * domain_criterion(mixup_output_domain, unlbl_dom_idx) + lam * domain_criterion(
        #     mixup_output_domain, unlbl_dom)
        mixup_dom_loss_orig = -torch.mean((mask_loss_all.int() * lam) * torch.sum(input_dom_lbl_one_hot * (torch.log(prob_mixup_domain_output + 1e-5)), 1))
        mixup_dom_loss_idx = -torch.mean((mask_loss_all_idx.int() * (1 - lam)) * torch.sum(input_dom_lbl_idx_one_hot * (torch.log(prob_mixup_domain_output + 1e-5)), 1))
        mixup_domain_loss = mixup_dom_loss_orig + mixup_dom_loss_idx
        # mixup_cls_loss = (1 - lam) * class_criterion(mixedup_output, input_clss_lbl_idx) + lam * class_criterion(
        #    mixedup_output, input_clss_lbl)
        input_clss_lbl_one_hot = F.one_hot(input_clss_lbl, num_classes=7)
        input_clss_lbl_idx_one_hot = F.one_hot(input_clss_lbl_idx, num_classes=7)

        mixup_cls_loss_orig = -torch.mean((mask_loss_all.int() * lam) * torch.sum(input_clss_lbl_one_hot * (torch.log(prob_mixup_output + 1e-5)), 1))
        mixup_cls_loss_idx = -torch.mean((mask_loss_all_idx.int() * (1 - lam)) * torch.sum(input_clss_lbl_idx_one_hot * (torch.log(prob_mixup_output + 1e-5)), 1))
        #mixup_cls_loss = lam * mixup_cls_loss_orig + (1 - lam) * mixup_cls_loss_idx
        mixup_cls_loss = mixup_cls_loss_orig + mixup_cls_loss_idx
        #print(loss_pseudo_unl.cpu().data)
        # print('mixup_domain_loss :', mixup_domain_loss)

        #loss_class_lbl = class_criterion(lbl_output_class, labels)/inputs.size(0)
        loss_class = mixup_cls_loss     # + loss_pseudo_unl
        loss_domain = mixup_domain_loss

        loss_entropy = entropy_criterion(mixedup_output)
        # _, lbl_pred_class = torch.max(output_class, 1)
        # _, lbl_pred_domain = torch.max(output_domain, 1)
        #print("inputs:", list(inputs))
        "labelled data giving correct predictions--start"
        # if batch_idx<4:
        #     inputs_reshaped = F.interpolate(inputs, size = 224)
        #     #print("inputs_reshaped size is :", inputs_reshaped.size())
        #     for i in range(inputs.size(0)):
        #         if lbl_pred_class[i] == labels[i]:
        #             lbl_data_correct_pred.append(inputs_reshaped[i])
        #             lbl_data_correct_pred_cls.append(lbl_pred_class[i])
        # if batch_idx==4:
        #     lbl_data_correct_pred = torch.stack(lbl_data_correct_pred)
        #
        #     lbl_data_correct_pred_cls = torch.stack(lbl_data_correct_pred_cls)


        "labelled data giving correct predictions -- end"
        _, pred_class = torch.max(mixedup_output, 1)
        _, pred_domain = torch.max(mixup_output_domain, 1)
        # print('beta in total loss :', beta)

        total_loss =   loss_class + loss_domain + loss_entropy * beta # + reg_loss
        # zero the parameter gradients

        total_loss.backward()
        #reg_loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        #reg_optimizers.step()

        # running_loss_class += loss_class.item() * inputs.size(0)
        running_loss_class += loss_class.item()

        #total_correct_class = torch.sum(pred_class == pseudo_labels_mixed.data)
        total_correct_class = torch.sum(lam * (pred_class==input_clss_lbl.data)) + \
                              torch.sum((1 - lam) * (pred_class==input_clss_lbl_idx.data))
        #print(total_correct_class)
        #total_reg_correct_class = torch.sum(lambda_pred_cls == lamdba_gt).item()

        running_correct_class += total_correct_class
        #running_reg_correct_class += total_reg_correct_class
        running_loss_domain += loss_domain.item()
        # running_loss_domain += loss_domain.item()
        total_correct_domain = torch.sum(pred_domain == domains_lbl.data)
        running_correct_domain += total_correct_domain
        running_loss_entropy += loss_entropy.item()
        running_loss_total += total_loss.item()
        #running_reg_loss +=  reg_loss.item()
        batch_samples +=  mixedup.size(0)


    # lbl_data_correct_pred = torch.stack(lbl_data_correct_pred)
    # lbl_data_correct_pred_cls = torch.stack(lbl_data_correct_pred_cls)

    # print('total labelled :', total_lbl)
    # print('total unlabelled :', total_unlbl)
    # train_data_len = len(source_lbl_train_ldr.dataset)+len(source_unlbl_train_ldr.dataset)
    total = batch_samples
    epoch_total_loss = running_loss_total/num_epoch
    epoch_loss_class = running_loss_class /num_epoch
    epoch_acc_class = running_correct_class.double() / total
    #print(epoch_acc_class)
    #epoch_reg_acc_class = running_reg_correct_class/total_unlbl
    #epoch_reg_loss = running_reg_loss /train_step
    epoch_loss_domain = running_loss_domain /num_epoch
    epoch_acc_domain = running_correct_domain.double() / total
    epoch_loss_entropy = running_loss_entropy/num_epoch
    log = 'Train: Epoch: {} Alpha: {:.4f} Loss Class: {:.4f} Acc Class: {:.4f} Loss Domain: {:.4f} Acc Domain: {:.4f} ' \
          'Loss Entropy: {:.4f} Total loss:{:.4f} '.format(epoch, alpha, epoch_loss_class, epoch_acc_class, epoch_loss_domain,
                                     epoch_acc_domain, epoch_loss_entropy, epoch_total_loss)

    # log_anthr = 'Train: Epoch: {} Alpha: {:.4f} loss_class_lbl: {:.4f}, mixup_cls_loss: {:.4f} loss_entropy_lbl: {:.4f}' \
    #             ' loss_entropy_unlbl: {:.4f} '.format(epoch, alpha, epoch_loss_class_lbl, epoch_mixup_cls_loss,
    #                                  epoch_loss_entropy_lbl, epoch_loss_entropy_unlbl)
    print(log)
    print('----------------------------')
    #print(log_anthr)
    with open(filename, 'a') as f: 
        f.write(log + '\n') 
    return model, optimizers

# Returns mixed inputs of same class
def mixup(device, input_strong_ldr, input_clss_lbl, unlbl_domains, alpha_mixup, mask_loss_all, batch_size) :
    # all_lbl_images_order = []
    # all_lbl_images = []
    # all_targets = []
    # all_dom_lab = []
    # lbl_domain_order = []
    # print('input_strong_ldr is :', input_strong_ldr.shape)
    # print('alpha_mixup :', alpha_mixup)
    RG = np.random.default_rng()
    #
    # if alpha_mixup > 0:
    #   lam = np.random.beta(alpha_mixup, alpha_mixup)
    # else:
    #   lam = 1
    #print("unlbl_domains.size(0) :", unlbl_domains.size(0))
    lam_batch = torch.from_numpy(RG.beta(alpha_mixup, alpha_mixup, size=unlbl_domains.size(0))).float()
    lam_batch_unsqeeze = lam_batch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    lam_batch_unsqeeze = lam_batch_unsqeeze.to(device)
    lam_batch = lam_batch.to(device)
    #print(lam_batch.size())

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
    #print("index is:", index)
    input_clss_lbl_idx = input_clss_lbl[index]
    unlbl_domains_idx = unlbl_domains[index]
    mask_loss_all_idx = mask_loss_all[index]
    #print('input_strong_ldr size is :', input_strong_ldr.size())
    #mixed_input_ldr = lam * input_strong_ldr + (1-lam) * input_strong_ldr[index, :]
    mixed_input_ldr = lam_batch_unsqeeze * input_strong_ldr + (1. - lam_batch_unsqeeze) * input_strong_ldr[index, :]

    #concat_input = torch.cat((mixed_input_ldr, input_strong_ldr), 0)



    # lbl_dom, unlbl_dom = lbl_domain_order, unlbl_domains
    # lbl_dom = lbl_dom.to(device)
    return mixed_input_ldr, input_clss_lbl, input_clss_lbl_idx, unlbl_domains, unlbl_domains_idx, mask_loss_all_idx, lam_batch








