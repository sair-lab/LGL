import torch
import numpy as np
import torch.nn as nn
from models.layer import FeatTransKCat, AttnFeatTransKCat 


class KTransCAT(nn.Module):
    '''
    Using a logit like ResNet and DenseNet to encode the neighbor in different level
    '''
    def __init__(self, feat_len, num_class, hidden = [64, 32], dropout = [0,0], k=1):
        super(KTransCAT, self).__init__()
        self.k = k
        c = [1, 4, hidden[1]]
        f = [feat_len, int(hidden[0]/c[1]), 1]
        self.feat1 = FeatTransKCat(in_channels=c[0], in_features=f[0], out_channels=c[1], out_features=f[1], khop = k)
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(c[1]), nn.Softsign())
        self.feat2 = FeatTransKCat(in_channels=c[1], in_features=f[1], out_channels=c[2], out_features=f[2], khop = k)
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(c[2]), nn.Softsign())
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p=0.1), nn.Linear(c[2]*f[2], num_class))


    def forward(self, x, neighbor):
        neighbor1 = [i[0] for i in neighbor]
        ## Temp setup the level we chose is also depends on the layer we have 
        if self.k == 1:
            neighbor2 = neighbor1
        else:
            neighbor2 = [i[1] for i in neighbor]

        x, adj = self.feat1(x, neighbor1)
        neighbor2 = [self.feat1.transform(neighbor2[i], adj[i:i+1]) for i in range(x.size(0))]
        x, neighbor2 = self.acvt1(x), [self.acvt1(n) for n in neighbor2]

        x, adj = self.feat2(x, neighbor2)
        x = self.acvt2(x)
        return self.classifier(x)
    

class AttnKTransCAT(nn.Module):
    '''
    Using a logit like ResNet and DenseNet to encode the neighbor in different level
    '''
    def __init__(self, feat_len, num_class, hidden = [64, 32], dropout = [0,0], k=1):
        super(AttnKTransCAT, self).__init__()
        self.k = k
        c = [1, 4, hidden[1]]
        f = [feat_len, int(hidden[0]/c[1]), 1]
        self.feat1 = AttnFeatTransKCat(in_channels=c[0], in_features=f[0], out_channels=c[1], out_features=f[1], khop = k)
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(c[1]), nn.Softsign())
        self.feat2 = AttnFeatTransKCat(in_channels=c[1], in_features=f[1], out_channels=c[2], out_features=f[2], khop = k)
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(c[2]), nn.Softsign())
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p=0.1), nn.Linear(c[2]*f[2], num_class))


    def forward(self, x, neighbor):
        neighbor1 = [i[0] for i in neighbor]
        ## Temp setup the level we chose is also depends on the layer we have 
        if self.k == 1:
            neighbor2 = neighbor1
        else:
            neighbor2 = [i[1] for i in neighbor]

        x, adj = self.feat1(x, neighbor1)
        neighbor2 = [self.feat1.transform(neighbor2[i], adj[i:i+1]) for i in range(x.size(0))]
        x, neighbor2 = self.acvt1(x), [self.acvt1(n) for n in neighbor2]

        x, adj = self.feat2(x, neighbor2)
        x = self.acvt2(x)
        return self.classifier(x)