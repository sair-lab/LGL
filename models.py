# Copyright <2020> <Chen Wang <https://chenwang.site>, Carnegie Mellon University>

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

import torch
import torch.nn as nn
from layer import FeatBrd1d


class Net(nn.Module):
    def __init__(self, args, hidden=2, num_class=7):
        super(Net, self).__init__()
        self.args = args
        feat_len = 1433
        self.feat1 = FeatBrd1d(in_channels=1, out_channels=hidden)
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(hidden), nn.Softsign())
        self.feat2 = FeatBrd1d(in_channels=hidden, out_channels=hidden)
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(hidden), nn.Softsign())
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(feat_len*hidden, num_class))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=args.lr, momentum=args.momentum)
        self.register_buffer('adj', torch.zeros(1, feat_len, feat_len))

    def forward(self, x, neighbor):
        fadj = self.feature_adjacency(x, neighbor)
        self.adj += fadj.sum(0)
        x = self.feat1(x, fadj)
        x = self.acvt1(x)
        x = self.feat2(x, self.adj)
        x = self.acvt2(x)
        return self.classifier(x)

    def observe(self, inputs, targets, neighbor):
        for i in range(self.args.iteration):
            self.optimizer.zero_grad()
            outputs = self.forward(inputs, neighbor)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

    def feature_adjacency(self, x, y):
        adj = (x.unsqueeze(-1) * y.unsqueeze(-2)).sum(dim=1).squeeze(dim=1)
        adj += adj.transpose(-1, -2)
        return self.row_normalize(adj.sqrt()) + torch.eye(x.size(-1), device=x.device)

    def row_normalize(self, x):
        x = x / x.sum(1, keepdim=True)
        x[torch.isinf(x)] = 0
        x[torch.isnan(x)] = 0
        return x


if __name__ == "__main__":
    '''
    Debug script for FGN model 
    '''
    n_feature, n_channel, n_batch = 1433, 1, 3
    feature = torch.FloatTensor(n_batch, n_channel, n_feature).random_()
    adjacency = torch.FloatTensor(n_feature, n_feature).random_()

    model = Net(adjacency)
    label = model(feature)
    print('Input: {}; Output: {}'.format(feature.shape, label.shape))
