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

import torch
import torch.nn as nn
from models.layer import FeatBrd1d, FeatTrans1d


class FGN(nn.Module):
    def __init__(self, adjacency, hidden=2, num_class=7):
        super(FGN, self).__init__()

        self.feat1 = FeatBrd1d(in_channels=1, out_channels=hidden)
        self.batch1 = nn.BatchNorm1d(hidden)
        self.acvt = nn.Softsign()
        self.feat2 = FeatBrd1d(in_channels=hidden, out_channels=hidden, adjacency=adjacency)
        self.batch2 = nn.BatchNorm1d(hidden)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(adjacency.size(0)*hidden, num_class))

    def forward(self, x, adj=None):
        x = self.feat1(x, adj)
        x = self.batch1(x)
        x = self.acvt(x)
        x = self.feat2(x)
        x = self.batch2(x)
        x = self.acvt(x)
        return self.classifier(x)


class LGL(nn.Module):
    def __init__(self, feat_len, num_class, hidden=2):
        super(LGL, self).__init__()
        c = [1, 4, 32]
        f = [feat_len, 16, 1]
        self.feat1 = FeatTrans1d(in_channels=c[0], in_features=f[0], out_channels=c[1], out_features=f[1])
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(c[1]), nn.Softsign())
        self.feat2 = FeatTrans1d(in_channels=c[1], in_features=f[1], out_channels=c[2], out_features=f[2])
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(c[2]), nn.Softsign())
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c[2]*f[2], num_class))

    def forward(self, x, neighbor):
        x, neighbor = self.feat1(x, neighbor)
        x, neighbor = self.acvt1(x), [self.acvt1(n) for n in neighbor]
        x, neighbor = self.feat2(x, neighbor)
        x = self.acvt2(x)
        return self.classifier(x)

class LSLGL(nn.Module):
    def __init__(self, feat_len, num_class, hidden=2):
        super(LGL, self).__init__()
        c = [1, 4, 8, 32]
        f = [feat_len, 64, 16, 1]
        self.feat1 = FeatTrans1d(in_channels=c[0], in_features=f[0], out_channels=c[1], out_features=f[1])
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(c[1]), nn.Softsign())
        self.feat2 = FeatTrans1d(in_channels=c[1], in_features=f[1], out_channels=c[2], out_features=f[2])
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(c[2]), nn.Softsign())
        self.feat3 = FeatTrans1d(in_channels=c[2], in_features=f[2], out_channels=c[3], out_features=f[3])
        self.acvt3 = nn.Sequential(nn.BatchNorm1d(c[3]), nn.Softsign())
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c[3]*f[3], num_class))

    def forward(self, x, neighbor):
        x, neighbor = self.feat1(x, neighbor)
        x, neighbor = self.acvt1(x), [self.acvt1(n) for n in neighbor]
        x, neighbor = self.feat2(x, neighbor)
        x, neighbor = self.acvt2(x), [self.acvt1(n) for n in neighbor]
        x, neighbor = self.feat3(x, neighbor)
        x = self.acvt3(x)
        return self.classifier(x)



if __name__ == "__main__":
    '''
    Debug script for FGN model 
    '''
    n_feature, n_channel, n_batch = 1433, 1, 3
    feature = torch.FloatTensor(n_batch, n_channel, n_feature).random_()
    adjacency = torch.FloatTensor(n_feature, n_feature).random_()

    model = FGN(adjacency)
    label = model(feature)
    print('Input: {}; Output: {}'.format(feature.shape, label.shape))
