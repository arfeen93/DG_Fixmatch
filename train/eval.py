from torch import nn
import torch
from torchvision import transforms
import torch.nn.functional as F
from numpy.random import *
import random
from torch.nn import MSELoss


def eval_model(model,reg_model, eval_data, train_lbl_data, device, epoch, filename):
    criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.BCEWithLogitsLoss()      # Loss function for Regressor
    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim = 1)
    model.eval()  # Set model to eval mode
    reg_model.eval() # Set regressor model to eval mode
    running_loss = 0.0
    running_reg_loss = 0.0
    running_corrects = 0
    running_reg_correct = 0
    running_reg_class_correct = 0
    # Iterate over data.
    data_num = 0
    idx = 0
    for inputs, labels in eval_data:
        idx+=1
        with torch.no_grad():

            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(inputs)
            # print("output is :", outputs)
            lambda_predict = purity_predict(model, reg_model, inputs, train_lbl_data, outputs, device)

            #reg_values, reg_idx = torch.max(lambda_predict, 1)
            # lambda_predict_sftmax = softmax(lambda_predict)
            lambda_predict_sigmoid = sigmoid(lambda_predict)
            if (epoch+1)%10==0:
                if idx ==2 or idx==5:
                    print('lambda_pred sigmoid for eval:', lambda_predict_sigmoid)
            values, indexes = torch.max(lambda_predict_sigmoid, 1)
            lambda_predict_cls = torch.round(torch.sigmoid(values))
            #reg_eval_loss = long(reg_eval_loss)
            #reg_eval_loss = reg_criterion(reg_values, labels)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            # print("classification pred:", preds)
            # print("lamda_cls_pred:", lamda_cls_pred)
            # statistics
            running_loss += loss.item() * inputs.size(0)
            #running_reg_loss += reg_eval_loss.item() * inputs.size(0)
            # for i in range(preds.shape[0]):
            # if lamda_cls_pred[i] == preds[i] and preds[i]==labels.data[i]:
            running_corrects += torch.sum(preds == labels.data).item()
            running_reg_correct += torch.sum(values >= 0.75).item()       # Regression accuracy
            running_reg_class_correct += torch.sum(indexes == labels.data).item()   # Regression class accuracy
            # running_corrects+=1
            # if lamda_cls_pred[i] == preds[i] or preds[i] == labels.data[i]:
            # running_corrects_new += 1
            # if lamda_cls_pred[i] == labels[i]:
            # running_corrects_lam_orig +=1
            # running_corrects_lam_orig += torch.sum(labels.data == lamda_cls_pred.data).item()
            # running_corrects_pred_orig += torch.sum(preds == labels.data).item()

            data_num += inputs.size(0)
    epoch_loss = running_loss / len(eval_data.dataset)
    epoch_acc = running_corrects / len(eval_data.dataset)
    epoch_reg_Acc = running_reg_correct/len(eval_data.dataset)
    #epoch_reg_eval_loss = running_reg_loss/ len(eval_data.dataset)
    epoch_reg_class_Acc = running_reg_class_correct / len(eval_data.dataset)
    # epoch_lam_orig_acc = running_corrects_lam_orig / len(eval_data.dataset)
    # epoch_lam_or_pred_orig_acc = running_corrects_new/len(eval_data.dataset)
    # epoch_pred_orig_acc = running_corrects_pred_orig/len(eval_data.dataset)

    # if (epoch+1) % 20 == 0:
    #     print("lambda predicted is :", lambda_predict)
    #     print("max lambda predicted is :", values)
    # print("input for eval:", labels)

    log = 'Eval: Epoch: {} Loss: {:.4f} Class Acc. : {:.4f}  ' \
        .format(epoch, epoch_loss,
                epoch_acc)
    regression_log = 'Eval: Epoch: {} Reg Acc : {:.4f} Reg Class Acc. : {:.4f}  ' \
        .format(epoch, epoch_reg_Acc, epoch_reg_class_Acc)
    print(log)
    print(regression_log)
    with open(filename, 'a') as f:
        f.write(log + '\n')
    return epoch_acc


"Novelty purity predictor---start"
def purity_predict(model, reg_model, inputs, train_lbl_data, outputs, device):

    all_train_data = []
    all_target = []
#     outputs_cls=outputs[0]
    #top3_logit = torch.topk(outputs_cls, 3)
    #print('top3_logit:', top3_logit)
    #top3_class = top3_logit[1]
    #print("top3_cls is:", top3_class)
    #print("regression model is :", reg_model)
    for input_x, target_x in train_lbl_data:
        train_in = input_x
        all_train_data.append(train_in)
        all_target.append(target_x)
    #print("shape of all_train_data:", len(all_train_data))
    #for k in range(top3_class.shape[1]):
#
#         #for j in range(top3_class.shape[0]):
#         #classes = top3_class[:,k]
#         #print("cls is :", cls)
#         #print("all target:", all_target)
#         #indx = (all_target == cls).nonzero(as_tuple=True)[0]
#         #for cls in classes:
    lambda_predict = torch.zeros(0)
    lambda_predict=lambda_predict.to(device)
    for cls in range(7):
        train_data_btch = []
        #indexes = [i for i, x in enumerate(all_target) if x==cls]
        indexes = [i for i, x in enumerate(all_target) if x == cls]
        for id in range(inputs.size(0)):
            indx = random.choice(indexes)
            #indx = all_target.index(cls)
            #train_data_btch.append(all_train_data[indx])
            train_data_btch.append(all_train_data[indx])
            #if k==0:
        train_data_btch0 = train_data_btch
        train_data_ldr0 = torch.stack(train_data_btch0)
        train_data_ldr0 = train_data_ldr0.to(device)
        #print("input size:{} train_Data_ldr0 size:{}".format(inputs.size(), train_data_ldr0.size()))
        mixup_ldr0 = 0.5 * inputs + 0.5 * train_data_ldr0
        x1, x2 = model.features(inputs)
        x3, x4 = model.features(mixup_ldr0)
        mixup_eval_ftr_cat_ldr = torch.cat((x4, x2), 1)
        #mixup_eval_dlr_cat = torch.cat((mixup_ldr0, inputs), 1)
        mixup_eval_ftr_cat_ldr = mixup_eval_ftr_cat_ldr.to(device)
        #print("mixup_eval_dlr_cat size is:", mixup_eval_dlr_cat.size())
        lambda_predict0 = reg_model(mixup_eval_ftr_cat_ldr)
        #lambda_predict0_cls = torch.round(torch.sigmoid(lambda_predict0))
        lambda_predict0=lambda_predict0.reshape(lambda_predict0.size(0), 1)
        lambda_predict = torch.cat((lambda_predict, lambda_predict0), 1)
    return lambda_predict
    #print("all 7 lambda predicted are:", lambda_predict)
#
#
#         # elif k==1:
#         #     train_data_btch1 = train_data_btch
#         #     train_data_ldr1 = torch.stack(train_data_btch1)
#         #     train_data_ldr1 = train_data_ldr1.to(device)
#         #
#         # else:
#         #     train_data_btch2 = train_data_btch
#         #     train_data_ldr2 = torch.stack(train_data_btch2)
#         #     train_data_ldr2 = train_data_ldr2.to(device)
# #print("inputs size:", inputs.shape)
# #print("train_Data_ldr0 size:", train_data_ldr0.shape)
#
#     # mixup_ldr0 = 0.5 * inputs + 0.5 * train_data_ldr0
#     # mixup_ldr1 = 0.5 * inputs + 0.5 * train_data_ldr1
#     # mixup_ldr2 = 0.5 * inputs + 0.5 * train_data_ldr2
#
#     # x0 = model.features(mixup_ldr0)
#     # lambda_predict0 = reg_model(x0)
#     # x1 = model.features(mixup_ldr1)
#     # lambda_predict1 = reg_model(x1)
#     # x2 = model.features(mixup_ldr2)
#     # lambda_predict2 = reg_model(x2)
#     # lambda_predict0 = reg_model(mixup_ldr0)
#     # lambda_predict1 = reg_model(mixup_ldr1)
#     # lambda_predict2 = reg_model(mixup_ldr2)
#     #lambda_predict = torch.cat((lambda_predict0, lambda_predict1, lambda_predict2), 1)
#     #values, indexes = torch.max(lambda_predict, 1)
#     #print("all 3lambda predicted are:", lambda_predict)
#     # lamda_cls_pred = [top3_class[i][indexes[i]] for i in range(indexes.shape[0])]
#     # lamda_cls_pred = torch.stack(lamda_cls_pred)
#     #lamda_cls_pred.append(top3_class[i][indexes[i]])
#
#     return  lambda_predict  #, lamda_cls_pred,
#
# "------end"
#
#
#
#
#
#
