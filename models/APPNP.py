import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as nf

from .GCN import sum_aggregation

class APPNP(nn.Module):
    '''
    APPNP: ICLR 2019
    Predict then Propagate: Graph Neural Networks Meet Personalized Pagerank
    https://arxiv.org/pdf/1810.05997.pdf
    '''
    def __init__(self, feat_len, num_class, hidden=[64,32], dropout=[0], alpha=0.1):
        super().__init__()
        self.feat_len, self.hidden = feat_len, num_class
        self.tran1 = nn.Linear(feat_len, hidden[0])
        self.tran2 = nn.Linear(hidden[0], hidden[1])
        self.app1 = GraphAppnp(alpha)
        self.app2 = GraphAppnp(alpha)
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(1), nn.ReLU(), nn.Dropout(dropout[0]))
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(1), nn.ReLU(), nn.Dropout(dropout[0]))
        self.classifier = nn.Linear(hidden[1], num_class)


    def forward(self, x, neighbor):
#         x, neighbor = nf.normalize(x), [nf.normalize(n) for n in neighbor]
        
        h, neighbor = self.tran1(x), [self.tran1(n) for n in neighbor]
        x, neighbor_agg = self.app1(h, neighbor, h, neighbor)
        x, neighbor_agg = self.acvt1(x), [self.acvt1(n) for n in neighbor_agg]
        x, neighbor = self.app2(x, neighbor_agg, h, neighbor)
        
        h, neighbor = self.tran2(x), [self.tran2(n) for n in neighbor]
        x, neighbor_agg = self.app1(h, neighbor, h, neighbor)
        x, neighbor_agg = self.acvt1(x), [self.acvt1(n) for n in neighbor_agg]
        x, _ = self.app2(x, neighbor_agg, h, neighbor)
        
        x[torch.isnan(x)] = 0
        return self.classifier(x).squeeze(1)

    
class APP(nn.Module):
    '''
    APPNP: ICLR 2019
    Predict then Propagate: Graph Neural Networks Meet Personalized Pagerank
    https://arxiv.org/pdf/1810.05997.pdf
    
    A modified version for the graph lifelong learning
    '''
    def __init__(self, feat_len, num_class, hidden=[64,32], dropout=[0], alpha=0.1):
        super().__init__()
        self.feat_len, self.hidden = feat_len, num_class
        self.tran1 = nn.Linear(feat_len, hidden[0])
        self.tran2 = nn.Linear(hidden[0], hidden[1])
        self.app1 = GraphApp(alpha)
        self.app2 = GraphApp(alpha)
        self.acvt = nn.Sequential(nn.ReLU(), nn.Dropout(dropout[0]))
        self.classifier = nn.Linear(hidden[1], num_class)

    def forward(self, x, neighbor):
        x, neighbor = nf.normalize(x), [nf.normalize(n) for n in neighbor]
        
        h, neighbor = self.tran1(x), [self.tran1(n) for n in neighbor]
        x, neighbor_agg = self.app1(h, neighbor, h, neighbor)
        x, neighbor_agg = self.acvt(x), [self.acvt(n) for n in neighbor_agg]
        x, neighbor = self.app2(x, neighbor_agg, h, neighbor)
        
        h, neighbor = self.tran2(x), [self.tran2(n) for n in neighbor]
        x, neighbor_agg = self.app1(h, neighbor, h, neighbor)
        x, neighbor_agg = self.acvt(x), [self.acvt(n) for n in neighbor_agg]
        x, _ = self.app2(x, neighbor_agg, h, neighbor)
        return self.classifier(x).squeeze(1)


class GraphApp(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, neighbor_agg, h, neighbor):
        # operate adj @ x for the subgraph
        x, neighbor_agg = sum_aggregation(x, neighbor_agg)
        
        # momentum operation
        x = (1-self.alpha) * x + self.alpha * h
        return x, neighbor


class GraphAppnp(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, neighbor_agg, h, neighbor):
        # operate adj @ x for the subgraph
        x, neighbor_agg = sum_aggregation(x, neighbor_agg)
        
        # momentum operation
        x = (1-self.alpha) * x + self.alpha * h
        neighbor_agg = [((1 - self.alpha) * n_agg +  self.alpha * n) for (n_agg,n) in zip(neighbor_agg, neighbor)]
        return x, neighbor_agg