from torch import nn
import torch
from torchvision import transforms
import torch.nn.functional as F
from numpy.random import *
import random
from torch.nn import MSELoss

def eval_model(model,reg_model, eval_data,train_lbl_data, device, epoch, filename):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Set model to training mode
    reg_model.eval()
    running_loss = 0.0
    running_corrects = 0
    running_corrects_lam_orig = 0
    running_corrects_pred_orig = 0
    running_corrects_new = 0

    # Iterate over data.
    data_num = 0

    for inputs, labels in eval_data:
        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(inputs)
            #print("output is :", outputs)



            lamda_cls_pred, lambda_predict = purity_predict(model, reg_model, inputs, train_lbl_data, outputs, device)

            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            #print("classification pred:", preds)
            #print("lamda_cls_pred:", lamda_cls_pred)
            # statistics
            running_loss += loss.item() * inputs.size(0)
            for i in range(preds.shape[0]):
                if lamda_cls_pred[i] == preds[i] and preds[i]==labels.data[i]:
                    #running_corrects += (preds == labels.data).item()
                    running_corrects+=1
                if lamda_cls_pred[i] == preds[i] or preds[i] == labels.data[i]:
                    running_corrects_new += 1
                #if lamda_cls_pred[i] == labels[i]:
                    #running_corrects_lam_orig +=1
            running_corrects_lam_orig += torch.sum(labels.data == lamda_cls_pred.data).item()
            running_corrects_pred_orig += torch.sum(preds == labels.data).item()

            data_num += inputs.size(0)
    epoch_loss = running_loss / len(eval_data.dataset)
    epoch_acc = running_corrects / len(eval_data.dataset)
    epoch_lam_orig_acc = running_corrects_lam_orig / len(eval_data.dataset)
    epoch_lam_or_pred_orig_acc = running_corrects_new/len(eval_data.dataset)
    epoch_pred_orig_acc = running_corrects_pred_orig/len(eval_data.dataset)

    #if (epoch) % 10 == 0:
    #print("lambda predicted is :", lambda_predict)

    log = 'Eval: Epoch: {} Loss: {:.4f} main Acc. : {:.4f} classifier pred acc : {:.4f}' \
          'lambda pred & orig acc:{:.4f} lam or pred orig acc:{:.4f} '\
        .format(epoch, epoch_loss, epoch_acc, epoch_pred_orig_acc,epoch_lam_orig_acc, epoch_lam_or_pred_orig_acc)
    print(log)
    with open(filename, 'a') as f: 
        f.write(log + '\n')
    # tb1.close()
    return epoch_acc


def purity_predict(model, reg_model, inputs, train_lbl_data, outputs, device):
    all_train_data = []

    all_target = []
    outputs_cls=outputs[0]
    top3_logit = torch.topk(outputs_cls, 3)
    #print('top3_logit:', top3_logit)
    top3_class = top3_logit[1]
    #print("top3_cls is:", top3_class)
    #print("regression model is :", reg_model)

    for input_x, target_x in train_lbl_data:
        train_in = input_x
        all_train_data.append(train_in)
        all_target.append(target_x)
    #print("shape of all_train_data:", len(all_train_data))
    for k in range(top3_class.shape[1]):
        train_data_btch = []
        #for j in range(top3_class.shape[0]):
        classes = top3_class[:,k]
        #print("cls is :", cls)
        #print("all target:", all_target)
        #indx = (all_target == cls).nonzero(as_tuple=True)[0]
        for cls in classes:
            indexes = [i for i, x in enumerate(all_target) if x==cls]
            indx = random.choice(indexes)
            #indx = all_target.index(cls)
            train_data_btch.append(all_train_data[indx])
            if k==0:
                train_data_btch0 = train_data_btch
                train_data_ldr0 = torch.stack(train_data_btch0)
                train_data_ldr0 = train_data_ldr0.to(device)

            elif k==1:
                train_data_btch1 = train_data_btch
                train_data_ldr1 = torch.stack(train_data_btch1)
                train_data_ldr1 = train_data_ldr1.to(device)

            else:
                train_data_btch2 = train_data_btch
                train_data_ldr2 = torch.stack(train_data_btch2)
                train_data_ldr2 = train_data_ldr2.to(device)
    #print("inputs size:", inputs.shape)
    #print("train_Data_ldr0 size:", train_data_ldr0.shape)

    mixup_ldr0 = 0.5 * inputs + 0.5 * train_data_ldr0
    mixup_ldr1 = 0.5 * inputs + 0.5 * train_data_ldr1
    mixup_ldr2 = 0.5 * inputs + 0.5 * train_data_ldr2

    # x0 = model.features(mixup_ldr0)
    # lambda_predict0 = reg_model(x0)
    # x1 = model.features(mixup_ldr1)
    # lambda_predict1 = reg_model(x1)
    # x2 = model.features(mixup_ldr2)
    # lambda_predict2 = reg_model(x2)
    lambda_predict0 = reg_model(mixup_ldr0)
    lambda_predict1 = reg_model(mixup_ldr1)
    lambda_predict2 = reg_model(mixup_ldr2)
    lambda_predict = torch.cat((lambda_predict0, lambda_predict1, lambda_predict2), 1)
    values, indexes = torch.max(lambda_predict, 1)
    #print("all 3lambda predicted are:", lambda_predict)
    lamda_cls_pred = [top3_class[i][indexes[i]] for i in range(indexes.shape[0])]
    lamda_cls_pred = torch.stack(lamda_cls_pred)
    #lamda_cls_pred.append(top3_class[i][indexes[i]])

    return lamda_cls_pred, lambda_predict











