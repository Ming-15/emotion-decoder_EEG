# _*_ coding:utf-8 _*_
'''
@author: Ming
@time: 18.01.2022
@target: ________
      __      __
     &&&     &&&              &&
    && $$   $$ &&            &&
   &&   $$ $$   &&          &&
  &&     $$$     &&        &&
 &&      ￥       &&      &&
&&                 &&    &&&&&&&&&&&&
'''
# _*_ coding:utf-8 _*_

import os
import time
import random
import glob
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from GNN_models.models import GAT
from loaddata import load_eeg
from GNN_models.GN_EEG import EEG_GAT


cuda = torch.cuda.is_available()
seed = 15
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

# import pickle
# with open('tr_data.pkl','rb') as f:
#     td = pickle.load(f)
# tr_data,tr_label = td[0],td[1]

adj, data_for_use, label_for_use = load_eeg(data_path=r'/data3/home/mlzhang/projects/SEED/ExtractedFeatures/1_20131027.mat')
tr_data = data_for_use[0]
tr_label = label_for_use[0]
val_data = data_for_use[1]
val_label = label_for_use[1]
te_data = data_for_use[2]
te_label = label_for_use[2]
model = EEG_GAT(node_num=62,nfeat=5,nhid=11,nclass=3,dropout=0.8,alpha=0.2,nheads=5)
optimizer = optim.Adam(model.parameters(),  lr=0.005,  weight_decay=0.0005)

model.criterion = torch.nn.CrossEntropyLoss()
model.optimizer = torch.optim.Adam(model.parameters(),  lr=0.005,  weight_decay=0.0005)
# model_cp.optimizer = torch.optim.Adam(model_cp.parameters(), lr=args.l_rate)
model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=100,
                                                     gamma=0.1)


if cuda:
    model.cuda()
    # adj = adj.cuda()
    tr_data = torch.FloatTensor(tr_data).cuda()
    tr_label = torch.LongTensor(tr_label).cuda()
    val_data = torch.FloatTensor(val_data).cuda()
    val_label = torch.LongTensor(val_label).cuda()
    te_data = torch.FloatTensor(te_data).cuda()
    te_label = torch.LongTensor(te_label).cuda()
# tr_data = Variable(tr_data)
tr_data,te_data,val_data = Variable(tr_data),Variable(te_data),Variable(val_data)
tr_label,te_label,val_label = Variable(tr_label),Variable(te_label),Variable(val_label)


def accuracy(prediction, labels):
    preds = prediction.type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
def train(epoch):
    t = time.time()
    tr_id = [i for i in range(len(tr_data))]
    tr_step = int(len(tr_data)/10)
    model.train()
    predictions = []
    ac_label = []
    pred_loss = []
    for bn in range(10):
        use = tr_id[bn*tr_step:tr_step*(bn+1)]
        model.optimizer.zero_grad()
        output = model(tr_data[use])
        predicted = torch.max(output.data, 1)[1]
        loss_train = model.criterion(output, tr_label[use])
        loss_train.backward()
        model.optimizer.step()
        ac_label.extend(tr_label[use].cpu().detach().numpy().tolist())
        predictions.extend(predicted.cpu().detach().numpy().tolist())
        pred_loss.append(loss_train.data.item())

    acc_train = accuracy(torch.Tensor(predictions), torch.Tensor(ac_label))
    train_loss = sum(pred_loss)
    print('train_acc',acc_train,'---train_loss',train_loss)

    model.eval()
    predictions = []
    ac_label = []
    pred_loss = []
    for bn in range(10):
        val_id = [i for i in range(len(val_data))]
        val_step = int(len(val_data)/10)
        val_use = val_id[bn*val_step:val_step*(bn+1)]
        output = model(val_data[val_use])
        predicted = torch.max(output.data, 1)[1]
        loss_val = model.criterion(output, val_label[val_use])

        predictions.extend(predicted.cpu().detach().numpy().tolist())
        pred_loss.append(loss_val.data.item())
        ac_label.extend(val_label[val_use].cpu().detach().numpy().tolist())
    acc_val = accuracy(torch.Tensor(predictions), torch.Tensor(ac_label))
    val_loss = sum(pred_loss)
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(train_loss),
          'acc_train: {:.4f}'.format(acc_train),
          'loss_val: {:.4f}'.format(val_loss),
          'acc_val: {:.4f}'.format(acc_val),
          'time: {:.4f}s'.format(time.time() - t))

    model.scheduler.step()
    return val_loss



t_total = time.time()
loss_values = []
bad_counter = 0
epochs = 1000
best = epochs + 1
best_epoch = 0
patience = 10
best_model = None
for epoch in range(epochs):
    loss_value = train(epoch)
    loss_values.append(loss_value)

    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
        # best_model_path = 'save_train/{}.pkl'.format(epoch)
        torch.save(model.state_dict(), 'save_train/{}.pkl'.format(epoch))

    else:
        bad_counter += 1

    if bad_counter == patience:
        break

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('save_train/6.pkl'.format(best_epoch)))

# Testing
model.eval()
predictions = []
ac_label = []
pred_loss = []
for bn in range(3):
    te_id = [i for i in range(len(te_data))]
    te_step = int(len(te_data) / 3)
    te_use = te_id[bn * te_step:te_step * (bn + 1)]
    output = model(te_data[te_use])
    predicted = torch.max(output.data, 1)[1]
    te_loss = model.criterion(output, te_label[te_use])

    predictions.extend(predicted.cpu().detach().numpy().tolist())
    pred_loss.append(te_loss.data.item())
    ac_label.extend(te_label[te_use].cpu().detach().numpy().tolist())
te_acc = accuracy(torch.Tensor(predictions), torch.Tensor(ac_label))
te_loss = sum(pred_loss)
print('bestmodel:',
      'loss_test: {:.4f}'.format(te_loss),
      'acc_test: {:.4f}'.format(te_acc),)


