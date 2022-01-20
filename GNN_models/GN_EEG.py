import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):


    def __init__(self, node_num,in_features, out_features, dropout, alpha, adj=None,concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.node_num = node_num
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        if adj is None:
            adj = np.ones(shape=(node_num,node_num))
        self.adj = nn.Parameter(torch.Tensor(adj),requires_grad=True)
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)


    def forward(self, h):
        # Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        attention = torch.matmul(attention,self.adj)
        h_prime = torch.matmul(attention, Wh)

        # if self.concat:
        #     return F.elu(h_prime)
        # else:
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.view(Wh2.shape[0],1,62)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class EEG_GAT(nn.Module):
    def __init__(self, node_num,nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(EEG_GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(node_num,nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(node_num,nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=node_num*nhid*nheads, out_features=64, bias=True),
            nn.Dropout(dropout),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=64, bias=True),
            nn.Dropout(dropout),
            nn.ELU(),
            nn.Linear(64, nclass),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x) for att in self.attentions], dim=x.dim()-1)
        x = x.view(x.shape[0],-1)
        x = self.classifier(x)
        return x
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.out_att(x)
        # x = F.elu(x)
        # return F.log_softmax(x, dim=1)