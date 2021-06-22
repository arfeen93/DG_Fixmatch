from torch import nn
import torch
from torchvision import transforms
import torch.nn.functional as F
from numpy.random import *
import random
from torch.nn import MSELoss
import numpy as np


def eval_model(model, eval_data, train_lbl_data, device, epoch, filename):
    criterion = nn.CrossEntropyLoss()
    #reg_criterion = nn.BCEWithLogitsLoss()      # Loss function for Regressor
    sigmoid = nn.Sigmoid()
    #softmax = nn.Softmax(dim = 1)
    model.eval()  # Set model to eval mode
    #reg_model.eval() # Set regressor model to eval mode
    running_loss = 0.0
    running_corrects = 0
    #running_reg_correct = 0
    running_reg_class_correct = 0
    # Iterate over data.
    #data_num = 0

    for inputs, labels in eval_data:

        with torch.no_grad():

            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward
            "classification loss and acc"
            outputs = model(inputs, inputs)
            if isinstance(outputs, tuple):
                output_score = outputs[0]
            loss = criterion(output_score, labels)
            _, preds = torch.max(output_score, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            running_loss += loss * inputs.size(0)
            # print("output is :", outputs)
            "regressor loss and acuracy"
            lambda_predict, top3_class = purity_predict(model, inputs, labels, outputs, criterion, train_lbl_data, device)

            #reg_values, reg_idx = torch.max(lambda_predict, 1)
            # lambda_predict_sftmax = softmax(lambda_predict)
            lambda_predict_sigmoid = sigmoid(lambda_predict)
            # if (epoch+1)%10==0:
            #     if idx ==2 or idx==5:
            #         print('lambda_pred sigmoid for eval:', lambda_predict_sigmoid)
            values, indexes = torch.max(lambda_predict_sigmoid, 1)
            indexes = indexes.long()
            #print("labels shape :", labels.shape)
            lambda_predict_cls = top3_class.gather(1, indexes.view(-1,1))
            #print("lambda_predict_cls shape :", lambda_predict_cls.shape)
            running_reg_class_correct += torch.sum(lambda_predict_cls == labels.data).item()   # Regression class accuracy

    epoch_loss = running_loss / len(eval_data.dataset)
    epoch_acc = running_corrects / len(eval_data.dataset)
    epoch_reg_class_Acc = running_reg_class_correct / len(eval_data.dataset)

    # if (epoch+1) % 20 == 0:
    #     print("lambda predicted is :", lambda_predict)
    #     print("max lambda predicted is :", values)
    # print("input for eval:", labels)

    log = 'Eval: Epoch: {} Loss: {:.4f} Class Acc. : {:.4f}  ' \
        .format(epoch, epoch_loss,
                epoch_acc)
    regression_log = 'Eval: Epoch: {} Reg Class Acc. : {:.4f}  ' \
        .format(epoch, epoch_reg_class_Acc)
    print(log)
    print(regression_log)
    with open(filename, 'a') as f:
        f.write(log + '\n')
    return epoch_acc, epoch_reg_class_Acc


"Novelty purity predictor---start"
def purity_predict(model, inputs, labels, outputs, criterion, train_lbl_data, device):

    all_train_data = []
    all_target = []
    all_train_dom = []

    for input_x, target_x,dom in train_lbl_data:
        train_in = input_x
        all_train_data.append(train_in)
        all_target.append(target_x)
        all_train_dom.append(dom)


    outputs_cls=outputs[0]
    #outputs_dom = outputs[1]
    #_, preds_dom = torch.max(outputs_dom, 1)
    top3_logit = torch.topk(outputs_cls, 3)
    #print('top3_logit:', top3_logit)
    top3_class = top3_logit[1]
    #print(" top3_class shape is:", top3_class.shape[1])
    #print("shape of all_train_data:", len(all_train_data))
    lambda_predict = torch.zeros(0)
    lambda_predict = lambda_predict.to(device)
    "Eval for regrression for top3 classes only"
    for k in range(top3_class.shape[1]):
        train_data_btch1 = []
        train_data_btch2 = []
        train_data_btch3 = []
        classes = top3_class[:,k]
        #print('shape of classes in eval:', classes.shape)
        for j in classes:
            #train_data = torch.zeros(all_train_data[0].shape)
            index = [i for i, x in enumerate(all_target) if x == j]
            #print("index:", index)
            #indx = random.choice(index)
            sample_indexes = random.sample(index, 3)
            train_data_btch1.append(all_train_data[sample_indexes[0]])
            train_data_btch2.append(all_train_data[sample_indexes[1]])
            train_data_btch3.append(all_train_data[sample_indexes[2]])
            #train_data_btch.append(all_train_data[indx])
        #train_data_btch1 = train_data_btch
        train_data_ldr0 = torch.stack(train_data_btch1)
        train_data_ldr0 = train_data_ldr0.to(device)

        train_data_ldr1 = torch.stack(train_data_btch2)
        train_data_ldr1 = train_data_ldr1.to(device)

        train_data_ldr2 = torch.stack(train_data_btch3)
        train_data_ldr2 = train_data_ldr2.to(device)
        # print("input size:{} train_Data_ldr0 size:{}".format(inputs.size(), train_data_ldr0.size()))
        # RG = np.random.default_rng()
        # lam_batch = torch.from_numpy(RG.beta(0.3, 0.3, size=labels.size(0))).float()
        # lam_batch_unsqeeze = lam_batch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # lam_batch_unsqeeze = lam_batch_unsqeeze.to(device)
        #mixup_ldr0 = 0.5 * inputs + 0.5 * train_data_ldr0
        mixup_ldr0 = 0.5 * inputs + 0.5 * train_data_ldr0
        mixup_ldr1 = 0.5 * inputs + 0.5 * train_data_ldr1
        mixup_ldr2 = 0.5 * inputs + 0.5 * train_data_ldr2

        #outputs = model(mixup_ldr0, inputs)
        outputs0 = model(mixup_ldr0, inputs)
        outputs1 = model(mixup_ldr1, inputs)
        outputs2 = model(mixup_ldr2, inputs)

        # lambda_predict0 = outputs[2]
        # lambda_predict0 = lambda_predict0.reshape(lambda_predict0.size(0), 1)
        # lambda_predict = torch.cat((lambda_predict, lambda_predict0), 1)

        lambda_predict0 = outputs0[2]
        lambda_predict0 = lambda_predict0.reshape(lambda_predict0.size(0), 1)

        lambda_predict1 = outputs1[2]
        lambda_predict1 = lambda_predict0.reshape(lambda_predict1.size(0), 1)

        lambda_predict2 = outputs2[2]
        lambda_predict2 = lambda_predict2.reshape(lambda_predict2.size(0), 1)

        lambda_predict_avg = (lambda_predict0 + lambda_predict1 + lambda_predict2)/3
        lambda_predict = torch.cat((lambda_predict, lambda_predict_avg), 1)

    return lambda_predict, top3_class

"Eval for regrression for all classes"
    # lambda_predict = torch.zeros(0)
    # lambda_predict=lambda_predict.to(device)
    # losses = 0
    # run_corrects = 0
    # for cls in range(7):
    #     train_data_btch = []
    #     #indexes = [i for i, x in enumerate(all_target) if x==cls]
    #     indexes = [i for i, x in enumerate(all_target) if x == cls]
    #     for id in range(inputs.size(0)):
    #         indx = random.choice(indexes)
    #         #indx = all_target.index(cls)
    #         #train_data_btch.append(all_train_data[indx])
    #         train_data_btch.append(all_train_data[indx])
    #         #if k==0:
    #     train_data_btch0 = train_data_btch
    #     train_data_ldr0 = torch.stack(train_data_btch0)
    #     train_data_ldr0 = train_data_ldr0.to(device)
    #     #print("input size:{} train_Data_ldr0 size:{}".format(inputs.size(), train_data_ldr0.size()))
    #     mixup_ldr0 = 0.5 * inputs + 0.5 * train_data_ldr0
        # x1, x2 = model.features(inputs)
        # x3, x4 = model.features(mixup_ldr0)
        # mixup_eval_ftr_cat_ldr = torch.cat((x4, x2), 1)
        #mixup_eval_dlr_cat = torch.cat((mixup_ldr0, inputs), 1)
        #mixup_eval_ftr_cat_ldr = mixup_eval_ftr_cat_ldr.to(device)
        #print("mixup_eval_dlr_cat size is:", mixup_eval_dlr_cat.size())
    #     outputs = model(inputs, mixup_ldr0)
    #
    #     lambda_predict0 = outputs[2]
    #     if isinstance(outputs, tuple):
    #         outputs = outputs[0]
    #     loss = criterion(outputs, labels)
    #     _, preds = torch.max(outputs, 1)
    #     run_corrects += torch.sum(preds == labels.data).item()
    #     losses += loss.item()
    #
    #     #_, preds = torch.max(outputs, 1)
    #     #lambda_predict0_cls = torch.round(torch.sigmoid(lambda_predict0))
    #     lambda_predict0=lambda_predict0.reshape(lambda_predict0.size(0), 1)
    #     lambda_predict = torch.cat((lambda_predict, lambda_predict0), 1)
    # loss_avg = losses/7
    # corrects_avg = run_corrects/7
    # return lambda_predict, loss_avg, corrects_avg

#
#
#
#
#
#
