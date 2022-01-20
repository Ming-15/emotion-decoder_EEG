import os

import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
# eeg_trails = ['de_movingAve1', 'de_movingAve2', 'de_movingAve3', 'de_movingAve4', 'de_movingAve5', 'de_movingAve6', 'de_movingAve7', 'de_movingAve8', 'de_movingAve9', 'de_movingAve10', 'de_movingAve11', 'de_movingAve12', 'de_movingAve13', 'de_movingAve14', 'de_movingAve15']
# label = [ 1,  0, -1, -1,  0,  1, -1, 0,  1,  1,  0, -1,  0,  1, -1]
#
#
# data = sio.loadmat(r'D:\projects\seed\ExtractedFeatures\1_20131027.mat')
# data2 = sio.loadmat(r'D:\projects\seed\ExtractedFeatures\1_20131030.mat')
# data3 = sio.loadmat(r'D:\projects\seed\ExtractedFeatures\1_20131107.mat')
# for i in range(15):
#     if data[eeg_trails[i]].shape[1] == data2[eeg_trails[i]].shape[1]:
#         if data[eeg_trails[i]].shape[1] == data3[eeg_trails[i]].shape[1]:
#             print('True')
def load_eeg(data_path,sample_ratio=(0.5,0.25,0.25)):
    # fns = os.listdir(data_path)
    # label = sio.loadmat(os.path.join(data_path,'label.mat'))
    eeg_trails = ['de_movingAve1', 'de_movingAve2', 'de_movingAve3', 'de_movingAve4', 'de_movingAve5', 'de_movingAve6',
                  'de_movingAve7', 'de_movingAve8', 'de_movingAve9', 'de_movingAve10', 'de_movingAve11',
                  'de_movingAve12', 'de_movingAve13', 'de_movingAve14', 'de_movingAve15']
    label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
    data = sio.loadmat(data_path)
    label_unique = list(np.unique(label))
    storelist = []
    for i in range(len(label_unique)):
        storelist.append([])

    for i in range(len(label)):
        temp_data = data[eeg_trails[i]]
        all_epoch_label = label[i]
        temp_data = temp_data.swapaxes(1,0)
        storelist[label_unique.index(all_epoch_label)].append(temp_data)
    tr_labels = []
    val_labels = []
    te_labels = []
    tr = []
    val = []
    te = []
    nn_2_aclabel = []
    for i in range(len(label_unique)):
        label_one = label_unique[i]
        nn_2_aclabel.append((i,label_one))
        labeled_data = np.concatenate(storelist[i],axis=0)
        all_lenth = labeled_data.shape[0]
        tr_num = int(all_lenth*sample_ratio[0])
        val_num = int(all_lenth*sample_ratio[1])
        te_num = int(all_lenth-tr_num-val_num)
        ids = [i for i in range(all_lenth)]
        np.random.shuffle(ids)
        tr_sample = labeled_data[ids[:tr_num],:,:]
        val_sample = labeled_data[ids[tr_num:tr_num+val_num],:,:]
        te_sample = labeled_data[ids[tr_num+val_num:],:,:]
        tr.append(tr_sample)
        tr_labels.extend([i]*tr_num)
        val.append(val_sample)
        val_labels.extend([i]*val_num)
        te.append(te_sample)
        te_labels.extend([i]*te_num)
    tr = np.concatenate(tr,axis=0)
    val = np.concatenate(val,axis=0)
    te = np.concatenate(te,axis=0)
    tr_labels = np.array(tr_labels)
    val_labels = np.array(val_labels)
    te_labels = np.array(te_labels)
    rd = [i for i in range(tr.shape[0])]
    np.random.shuffle(rd)
    tr = tr[rd,:,:]
    tr_labels=tr_labels[rd]
    data_for_load = [tr,val,te]
    label_data_for_load = [tr_labels,val_labels,te_labels]
    for i in range(len(data_for_load)):
        data_for_load[i] = torch.FloatTensor(data_for_load[i])
    for i in range(len(label_data_for_load)):
        label_data_for_load[i] = torch.LongTensor(label_data_for_load[i])

    nodes_num = tr.shape[1]
    adj = np.ones(shape=(nodes_num,nodes_num))
    adj = torch.FloatTensor(adj)



    return adj, data_for_load,label_data_for_load



# def load_data(path="./data/cora/", dataset="cora"):
#     """Load citation network dataset (cora only for now)"""
#     print('Loading {} dataset...'.format(dataset))
#
#     idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
#     features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     labels = encode_onehot(idx_features_labels[:, -1])
#
#     # build graph
#     idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#     idx_map = {j: i for i, j in enumerate(idx)}
#     edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
#
#     # build symmetric adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#
#     features = normalize_features(features)
#     adj = normalize_adj(adj + sp.eye(adj.shape[0]))
#
#     idx_train = range(140)
#     idx_val = range(200, 500)
#     idx_test = range(500, 1500)
#
#     adj = torch.FloatTensor(np.array(adj.todense()))
#     features = torch.FloatTensor(np.array(features.todense()))
#     labels = torch.LongTensor(np.where(labels)[1])
#
#     idx_train = torch.LongTensor(idx_train)
#     idx_val = torch.LongTensor(idx_val)
#     idx_test = torch.LongTensor(idx_test)
#
#     return adj, features, labels, idx_train, idx_val, idx_test