import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as nf


class GCN(nn.Module):
    '''
    A variant of
    GCN: Graph Convolutional Network, ICLR 2017
    https://arxiv.org/pdf/1609.02907.pdf
    '''
    def __init__(self, feat_len, num_class, hidden=[64,32], dropout=[0]):
        super().__init__()
        self.feat_len, self.hidden = feat_len, num_class
        self.gcn1 = GraphConv(feat_len, hidden[0])
        self.gcn2 = GraphConv(hidden[0], hidden[1])
        self.acvt = nn.Sequential(nn.ReLU(), nn.Dropout(dropout[0]))
        self.classifier = nn.Linear(hidden[1], num_class, bias = False)
        
    def forward(self, x, neighbor):
        x, neighbor = nf.normalize(x), [nf.normalize(n) for n in neighbor]
        
        x, neighbor = self.gcn1(x, neighbor)
        x, neighbor = self.acvt(x), [self.acvt(adj) for adj in neighbor]
        x, neighbor = self.gcn2(x, neighbor)
        x = self.acvt(x)
        
        return self.classifier(x).squeeze(1)


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x, neighbor):
        x = self.linear(x)
        neighbor = [self.linear(n) for n in neighbor]
        x, neighbor = sum_aggregation(x, neighbor)
        return x, neighbor

def sum_aggregation(x, neighbor):
    batch_id = x.shape[0]
    aggred_x = torch.stack([neighbor[i].sum(0) for i in range(batch_id)])
    neighbor = [torch.add(x[i].unsqueeze(0),neighbor[i]) for i in range(batch_id)]
    return aggred_x, neighbor