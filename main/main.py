import sys
sys.path.append('../')
import numpy as np
from torch.utils.data import DataLoader
import torch
import argparse
import os
from util.util import *
from train.eval import *
from util.scheduler import *
from clustering.domain_split import domain_split
from dataloader.dataloader import random_split_dataloader
from model.purity_predictor import *

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data-root', default='/home/arfeen/papers_code/dom_gen_aaai_2020/PACS/kfold/')
    parser.add_argument('--save-root', default='/home/arfeen/papers_code/dom_gen_aaai_2020/')
    parser.add_argument('--result-dir', default='default')
    parser.add_argument('--train', default='general')
    parser.add_argument('--data', default='PACS')
    parser.add_argument('--model', default='caffenet')
    parser.add_argument('--clustering', action='store_true')
    parser.add_argument('--clustering-method', default='Kmeans')
    parser.add_argument('--num-clustering', type=int, default=3)
    parser.add_argument('--clustering-step', type=int, default=1)
    parser.add_argument('--entropy', choices=['default', 'maximum_square'])
    parser.add_argument('--exp-num', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-epoch', type=int, default=30)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--save-step', type=int, default=100)
    
    parser.add_argument('--batch-size', type=int, default=128)  # batch size is 128 for Alexnet and 100 for resnet
    parser.add_argument('--labeled-batch-size', type=int, default=50, help='no of labelled sample in a batch') #<---labeled-batch-size=50 for Alexnet and 40 for Resnet
    parser.add_argument('--scheduler', default='step')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-step', type=int, default=24)
    parser.add_argument('--lr-decay-gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--fc-weight', type=float, default=1.0)
    parser.add_argument('--disc-weight', type=float, default=1.0)
    parser.add_argument('--entropy-weight', type=float, default=1.0)
    parser.add_argument('--grl-weight', type=float, default=1.0)
    parser.add_argument('--loss-disc-weight', action='store_true')
    
    parser.add_argument('--color-jitter', action='store_true')
    parser.add_argument('--min-scale', type=float, default=0.8)

    parser.add_argument('--instance-stat', action='store_true')
    parser.add_argument('--feature-fixed', action='store_true')
    parser.add_argument('--alpha-mixup', type=float, default=1.0,
                        help='mixup interpolation coefficient (default: 1)')
    
    args = parser.parse_args()
    
    path = args.save_root + args.result_dir
    if not os.path.isdir(path):
        os.makedirs(path)
        os.makedirs(path + '/models')
    
    with open(path+'/args.txt', 'w') as f:
        f.write(str(args))
        
    domain = get_domain(args.data)
    source_domain, target_domain = split_domain(domain, args.exp_num)
    # "For labelled and unlabelled indexes---start"
    # dom_wise_samples = [1503, 1843, 2109, 3536]
    # dom_wise_samples.pop(args.exp_num)
    # train_sam_nos = dom_wise_samples
    # #print(train_sam_nos)
    # "---end"

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    print('device is :', device)
    print("no of epoch:", args.num_epoch)
    get_domain_label, get_cluster = train_to_get_label(args.train, args.clustering)
    print('alpha_mixup is :', args.alpha_mixup)
    print("lr is:", args.lr)
    #print("lr step is:", args.lr_step)
    print("labelled batch size:", args.labeled_batch_size)

    source_train, source_lbl_train, source_val, target_test = random_split_dataloader(
        data=args.data, data_root=args.data_root, source_domain=source_domain, target_domain=target_domain,
        batch_size=args.batch_size, labeled_batch_size=args.labeled_batch_size,  get_domain_label=get_domain_label, get_cluster=get_cluster, num_workers=4,
        color_jitter=args.color_jitter, min_scale=args.min_scale)
    #print('source_train is:', source_train.dataset.dataset)
    # print('source_unlbl_train_ldr is :', list(source_unlbl_train_ldr)[0])
    # print('cls_lbl is :', cls_lbl)
#     num_epoch = int(args.num_iteration / len(source_train))
#     lr_step = int(args.lr_step / min([len(domain) for domain in source_train]))
#     print(num_epoch)
    num_epoch = args.num_epoch
    lr_step = args.lr_step
    label_batch_size = args.labeled_batch_size
    
    disc_dim = get_disc_dim(args.train, args.clustering, len(source_domain), args.num_clustering)
    num_classes = source_train.dataset.num_class
    
    # print('no of class :',source_lbl_train_ldr.dataset.num_class)
    # model = get_model(args.model, args.train)(
        # num_classes=source_lbl_train.dataset.dataset.num_class, num_domains=disc_dim, pretrained=True)
    # model = get_model(args.model, args.train)(
    #     num_classes=source_train.dataset.dataset.num_class, num_domains=disc_dim, pretrained=True)
    model = get_model(args.model, args.train)(
        num_classes=source_train.dataset.num_class, num_domains=disc_dim, pretrained=True)
    model = model.to(device)
    # reg_model = Purity()
    # reg_model = reg_model.to(device)
    #print('model is :', model)
    model_lr = get_model_lr(args.model, args.train, model,  fc_weight=args.fc_weight, disc_weight=args.disc_weight)
    print("model lr:", model_lr)
    optimizers = [get_optimizer( model_part, args.lr * alpha, args.momentum, args.weight_decay,
                                args.feature_fixed, args.nesterov, per_layer=False) for model_part, alpha in model_lr]
    # print("optimizers:", optimizers)
    #reg_optimizers = reg_optimizer(reg_model, reg_lr=0.01, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    # "Regression model optimizer"


    if args.scheduler == 'inv':
        schedulers = [get_scheduler(args.scheduler)(optimizer=opt, alpha=10, beta=0.75, total_epoch=num_epoch)
                     for opt in optimizers]
    elif args.scheduler == 'step':
        schedulers = [get_scheduler(args.scheduler)(optimizer=opt, step_size=lr_step, gamma=args.lr_decay_gamma)
                     for opt in optimizers]
        # reg_scheduler = get_scheduler('multistep')(optimizer=reg_optimizers, milestones=[60, 90, 150, 200],
        #                                            gamma=0.1)
    elif args.scheduler == 'multistep':
        schedulers = [get_scheduler(args.scheduler)(optimizer=opt, milestones =[ 50, 280], gamma=args.lr_decay_gamma) #<---milestone = 25  for caffenet and no lrstep for resnet
                     for opt in optimizers]
        # reg_scheduler = get_scheduler('multistep')(optimizer=reg_optimizers, milestones=[ 100, 180], gamma=0.1)
        # print('multistep scheduler is used')
    else:
        raise ValueError('Name of scheduler unknown %s' %args.scheduler)

    best_acc = 0.0
    test_acc = 0.0
    best_epoch = 0
    best_reg_acc = 0

    "Training the model"
    for epoch in range(num_epoch):

        print('Epoch: {}/{}, Lr: {:.6f}'.format(epoch, num_epoch-1, optimizers[0].param_groups[0]['lr']))
        print('Temporary Best Accuracy is {:.4f} ({:.4f} at Epoch {})'.format(test_acc, best_acc, best_epoch))

        dataset = source_train.dataset
        #dataset = deepcopy(source)
        #print("vnf:", (dataset.cluster_lbl))
        if args.clustering:
            if epoch % args.clustering_step == 0:
                pseudo_domain_label = domain_split(dataset, model, device=device,
                                                       cluster_before=dataset.cluster_lbl,
                                                       filename=path + '/nmi.txt', epoch=epoch,
                                                       nmb_cluster=args.num_clustering, method=args.clustering_method,
                                                       pca_dim=256, whitening=False, L2norm=False,
                                                       instance_stat=args.instance_stat)
                #dataset.set_cluster(np.array(pseudo_domain_label))
                dataset.set_cluster_index(np.array(pseudo_domain_label))

        if args.loss_disc_weight:
            if args.clustering:
                hist = dataset.cluster_lbl

            else:
                hist = dataset.domains_lbl

            #print("np.histogram(hist, bins=model.num_domains)[0]:", np.histogram(hist, bins=model.num_domains)[0])
            weight = 1. / np.histogram(hist, bins=model.num_domains)[0]
            weight = weight / weight.sum() * model.num_domains
            weight = torch.from_numpy(weight).float().to(device)


        else:
            weight = None


        # print('weight :', weight)
        # print('unlbl_weight :', unlbl_weight)
        model, optimizers = get_train(args.train)(model=model,source_train=source_train, optimizers=optimizers, device=device,
                            epoch=epoch, num_epoch=num_epoch,filename=path+'/source_train.txt', entropy=args.entropy,
                            alpha_mixup=args.alpha_mixup, disc_weight=weight,entropy_weight=args.entropy_weight,
                                                  grl_weight=args.grl_weight, label_batch_size = args.labeled_batch_size)

        if epoch % args.eval_step == 0:
            acc, reg_acc = eval_model(model, source_val, source_lbl_train, device, epoch, path+'/source_eval.txt')
            acc_, reg_acc_ = eval_model(model, target_test, source_lbl_train, device, epoch, path+'/target_test.txt')

        if epoch % args.save_step == 0:
            torch.save(model.state_dict(), os.path.join(
                path, 'models',
                "model_{}.pt".format(epoch)))

        if acc >= best_acc:
            best_acc = acc
            test_acc = acc_
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(
                path, 'models',
                "model_best.pt"))

        # if reg_acc>=best_reg_acc:
        #     best_reg_acc=reg_acc


        # curr_reg_lr = get_reg_lr(reg_optimizers)
        # print("Current reg model lr :", curr_reg_lr)
        for scheduler in schedulers:
            scheduler.step()



    best_model = get_model(args.model, args.train)(num_classes=source_train.dataset.num_class, num_domains=disc_dim, pretrained=False)
    best_model.load_state_dict(torch.load(os.path.join(
                path, 'models',
                "model_best.pt"), map_location=device))
    best_model = best_model.to(device)
    test_acc, test_reg_acc = eval_model(best_model, target_test, source_lbl_train, device, best_epoch, path+'/target_best.txt')
    print('Test Accuracy by the best model on the source domain is {} Reg cls Acc:{} (at Epoch {})'.format(test_acc, test_reg_acc, best_epoch))
