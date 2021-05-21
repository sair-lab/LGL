
import math
import torch
import torch.nn as nn
import torch.nn.functional as nf


class GAT(nn.Module):
    def __init__(self, feat_len, num_class, hidden=[64,32], dropout=[0]):
        '''
        GAT: Graph Attention Network, ICLR, 2018
        https://arxiv.org/pdf/1710.10903.pdf
        '''
        super().__init__()
        self.feat1 = GraphAttn(in_channels = feat_len, out_channels = hidden[0])
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(1), nn.ReLU())
        self.feat2 = GraphAttn(in_channels = hidden[0], out_channels = hidden[1])
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(1), nn.ReLU())
        self.linear = nn.Sequential(nn.Flatten(), nn.Dropout(dropout[-1]), nn.Linear(hidden[1], num_class))

    def forward(self, x, neighbor):
        x, neighbor = nf.normalize(x), [nf.normalize(n) for n in neighbor]
        
        x, neighbor = self.feat1(x, neighbor)
        x, neighbor = self.acvt1(x), [self.acvt1(adj) for adj in neighbor]
        x, neighbor = self.feat2(x, neighbor)
        x = self.acvt2(x)
        return self.linear(x)


class GraphAttn(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.2):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.tran = nn.Linear(in_channels, out_channels)
        self.att1 = nn.Linear(out_channels, 1, bias=False)
        self.att2 = nn.Linear(out_channels, 1, bias=False)
        self.norm = nn.Sequential(nn.LeakyReLU(alpha),nn.Softmax(dim=0))

    def forward(self, x, neighbor):
        B, C, F = x.shape
        x, neighbor = self.tran(x), [self.tran(n) for n in neighbor]

        batched_A = [self.att1(x[i]).unsqueeze(0) + self.att2(neighbor[i]) for i in range(B)] 
        A = [self.norm(a.squeeze(1)) for a in batched_A]
        
        attn_x = torch.stack([torch.einsum("nc,ncf -> cf", A[i], neighbor[i]) for i in range(B)])
        
        return attn_x , neighbor