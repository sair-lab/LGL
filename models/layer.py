# Copyright <2019> <Chen Wang <https://chenwang.site>, Carnegie Mellon University>

# Redistribution and use in source and binary forms, with or without modification, are 
# permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of 
# conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list 
# of conditions and the following disclaimer in the documentation and/or other materials 
# provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be 
# used to endorse or promote products derived from this software without specific prior 
# written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH 
# DAMAGE.

import os
import torch
import os.path
import torch.nn as nn


class FeatBrd1d(nn.Module):
    '''
    Feature Broadcasting Layer for multi-channel 1D features.
    Input size should be (n_batch, in_channels, n_features)
    Output size is (n_batch, out_channels, n_features)
    Args:
        in_channels (int): number of feature input channels
        out_channels (int): number of feature output channels
        adjacency (Tensor): feature adjacency matrix
    '''
    def __init__(self, in_channels, out_channels, adjacency=None):
        super(FeatBrd1d, self).__init__()
        self.register_buffer('adjacency', adjacency)
        # Taking advantage of parallel efficiency of nn.Conv1d
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, adj=None):
        if adj is not None:
            return (adj.unsqueeze(1) @ (self.conv(x).unsqueeze(-1))).view(x.size(0),-1,x.size(-1))
        else:
            return (self.adjacency @ (self.conv(x).unsqueeze(-1))).view(x.size(0),-1,x.size(-1))


class FeatTrans1d(nn.Module):
    '''
    Feature Transforming Layer for multi-channel 1D features.
    Input size should be (n_batch, in_channels, in_features)
    Output size is (n_batch, out_channels, out_features)
    Args:
        in_channels (int): number of feature input channels
        out_channels (int): number of feature output channels
        in_features (int): dimension of input features
        out_features (int): dimension of output features
    '''
    def __init__(self, in_channels, in_features, out_channels, out_features):
        super(FeatTrans1d, self).__init__()
        self.out_channels, self.out_features = out_channels, out_features
        # Taking advantage of parallel efficiency of nn.Conv1d
        self.conv = nn.Conv1d(in_channels, out_channels*out_features, kernel_size=in_features, bias=False)

    def forward(self, x, neighbor):
        adj = self.feature_adjacency(x, neighbor)
        x = self.transform(x, adj)
        neighbor = [self.transform(neighbor[i], adj[i:i+1]) for i in range(x.size(0))]
        return x, neighbor

    def transform(self, x, adj):
        return self.conv((adj.unsqueeze(1) @ x.unsqueeze(-1)).squeeze(-1)).view(x.size(0), self.out_channels, self.out_features)

    def feature_adjacency(self, x, y):
        fadj = torch.stack([torch.einsum('ca,ncb->ab', x[i], y[i]) for i in range(x.size(0))])
        fadj += fadj.transpose(-2, -1)
        return self.row_normalize(self.sgnroot(fadj))

    def sgnroot(self, x):
        return x.sign()*(x.abs().sqrt())

    def row_normalize(self, x):
        x = x / (x.abs().sum(1, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0
        return x


class FeatTransKhop(nn.Module):
    '''
    Feature Transforming Layer for K-hop multi-channel 1D features.
    Input size should be (n_batch, in_channels, in_features)
    Output size is (n_batch, out_channels, out_features)
    This will encode the k-hop into the channel of the nodex
    Args:
        in_channels (int): number of feature input channels
        k-hop (int): number of the k level.
        out_channels (int): number of feature output channels
        in_features (int): dimension of input features
        out_features (int): dimension of output features

    '''
    def __init__(self, in_channels, khop, in_features, out_channels, out_features):
        super(FeatTransKhop, self).__init__()
        self.khop = khop
        self.out_channels, self.out_features = out_channels, out_features
        self.in_channels, self.in_features = khop*in_channels, in_features
        # Taking advantage of parallel efficiency of nn.Conv1d
        self.conv = nn.Conv1d(self.in_channels, out_channels*out_features, kernel_size=in_features, bias=False)

    def forward(self, x, neighbor):
        '''
        x: (N,c_in,f) c_in of level 1 is 1
        W: (k,c_out*k,f,f)
        neighbor: (N,k,n,f) N and k is list
        adj: (N,k,f,f)
        '''
        adj = self.feature_adjacency(x, neighbor)
        x = self.transform(x, adj)
        neighbor = [[self.transform(neighbor[i][k], adj[i:i+1]) for k in range(self.khop)] for i in range(x.size(0))]
        return x, neighbor

    def transform(self, x, adj):
        '''
        Current set up address for neighbor c = 1
        adj: (N,k,f,f)
        x: (N,1,f)
        neighbor: (n,k,f)
        adj: (N,k,1,f,f)
        mm(N,k,1,f,f @ N,1,1,f,1)->(N,k,1,f,1); view->(N,k*c_in,f)
        W: (N, c_out, f)
        '''
        return self.conv(((adj.unsqueeze(2) @ x.unsqueeze(-1).unsqueeze(1)).view(x.size(0), self.in_channels, self.in_features))).view(x.size(0), self.out_channels, self.out_features)

    def feature_adjacency(self, x, y):
        '''
        x:Ncf, y:Nkncf -> Nkff where k and N is for list
        k is k-hop, N is batch number, n is neibor number
        fadj (N,k,f,f)
        '''
        fadj = torch.stack([torch.stack([torch.einsum('ca,ncb->ab', x[i], y[i][k]) for k in range(self.khop)]) for i in range(x.size(0))])
        fadj += fadj.transpose(-2, -1) # swap the last two channels -> Nkba
        return self.row_normalize(self.sgnroot(fadj))

    def sgnroot(self, x):
        return x.sign()*(x.abs().clamp(min=1e-8).sqrt())

    def row_normalize(self, x):
        ## TODO row dimension to -1 or -2
        x = x / (x.abs().sum(-2, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0
        return x