import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from loaddata import load_eeg_for_linear
import time
torch.random.manual_seed(15)

import torchvision.models as models
models.AlexNet()


class linear_classifier(nn.Module):
    def __init__(self,input_dim=310,nclass=3):
        super(linear_classifier,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=16, bias=True),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(in_features=16, out_features=16, bias=True),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(16, nclass),# 0, 1, 2
            nn.Softmax(dim=-1) # 0.5,0.35,0.15  TOP-1 predicted_label = 0
        )
    def forward(self, x):
        x = self.classifier(x)
        return x

data_for_use, label_for_use = load_eeg_for_linear(data_path=r'SEED\ExtractedFeatures\1_20131027.mat')
tr_data = data_for_use[0]
tr_label = label_for_use[0]
val_data = data_for_use[1]
val_label = label_for_use[1]
te_data = data_for_use[2]
te_label = label_for_use[2]

model = linear_classifier(input_dim=310,nclass=3)
model.criterion = torch.nn.CrossEntropyLoss()
model.optimizer = torch.optim.SGD(model.parameters(),  lr=0.01,  weight_decay=0.1)
model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=120, gamma=0.1)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
# optimizer = torch.optim.AdamW(model.parameters(),lr=args.l_rate,weight_decay=args.lr_decay)
# model.optimizer = torch.optim.AdamW(model.parameters(),lr=args.l_rate)
# # model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, )
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)

if torch.cuda.is_available():
    model.cuda()
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
        # model.optimizer.zero_grad()
        output = model(tr_data[use])
        predicted = torch.max(output.data, 1)[1]
        loss_train = model.criterion(output, tr_label[use])
        model.optimizer.zero_grad()
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
    return acc_val,val_loss



t_total = time.time()
loss_values = []
bad_counter = 0
epochs = 250
best = epochs + 1
best_epoch = 0
patience = 10
best_model = None
accuracy_list =[]
for epoch in range(epochs):
    val_acc,loss_value = train(epoch)
    accuracy_list.append(val_acc)
    # loss_values.append(loss_value)
    # if loss_values[-1] < best:
    #     best = loss_values[-1]
    #     best_epoch = epoch
    #     bad_counter = 0
    #     # best_model_path = 'save_train/{}.pkl'.format(epoch)
    #     torch.save(model.state_dict(), 'save_train/best.pkl')
    # # else:
    #     bad_counter += 1
    #
    # if bad_counter == patience:
    #     break

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
# model.load_state_dict(torch.load('save_train/6.pkl'.format(best_epoch)))
plt.plot(accuracy_list)
plt.show()