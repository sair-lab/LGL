import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as nf


class MLP(nn.Module):
    '''
    A variant of
    GCN: Graph Convolutional Network, ICLR 2017
    https://arxiv.org/pdf/1609.02907.pdf
    '''
    def __init__(self, feat_len, num_class, hidden=[64,32], dropout=[0]):
        super().__init__()
        self.feat_len, self.hidden = feat_len, num_class
        self.tran1 = nn.Linear(feat_len, hidden[0], bias = False)
        self.tran2 = nn.Linear(hidden[0], hidden[1], bias = False)
        self.acvt = nn.Sequential(nn.ReLU(), nn.Dropout(dropout[0]))
        self.classifier = nn.Linear(hidden[1], num_class, bias = False)
        
    def forward(self, x, neighbor):
        x = nf.normalize(x)
        
        x = self.tran1(x)
        x = self.acvt(x)
        x = self.tran2(x)
        x = self.acvt(x)
        return self.classifier(x).squeeze(1)
