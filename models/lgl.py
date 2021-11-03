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
import numpy as np
import torch.nn as nn
import torch.nn.functional as nf
from models.KTransCat import KTransCAT

from models.layer import FeatBrd1d, FeatTrans1d, FeatTransKhop, FeatTransKCat, FeatTransKhop, Mlp, AttnFeatTrans1d, AttnFeatTrans1dSoft


class LGL(nn.Module):
    def __init__(self, feat_len, num_class, hidden = [64, 32], dropout = [0,0]):
        ## the Flag ismlp will encode without neighbor
        super(LGL, self).__init__()
        c = [1, 4, hidden[1]]
        f = [feat_len, int(hidden[0]/c[1]), 1]

        self.feat1 = FeatTrans1d(in_channels=c[0], in_features=f[0], out_channels=c[1], out_features=f[1])
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(c[1]), nn.Softsign(),  nn.Dropout(p=dropout[0]))
        self.feat2 = FeatTrans1d(in_channels=c[1], in_features=f[1], out_channels=c[2], out_features=f[2])
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(c[2]), nn.Softsign(), nn.Dropout(p=dropout[1]))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c[2]*f[2], num_class))


    def forward(self, x, neighbor):

        x, neighbor = self.feat1(x, neighbor)
        x, neighbor = self.acvt1(x), [self.acvt1(n) for n in neighbor]
        x, neighbor = self.feat2(x, neighbor)
        x = self.acvt2(x)
        return self.classifier(x)


class AFGN(nn.Module):
    def __init__(self, feat_len, num_class, hidden = [64, 32], dropout = [0,0]):
        ## the Flag ismlp will encode without neighbor
        super(AFGN, self).__init__()
        c = [1, 4, hidden[1]]
        f = [feat_len, int(hidden[0]/c[1]), 1]
        self.feat1 = AttnFeatTrans1dSoft(in_channels=c[0], in_features=f[0], out_channels=c[1], out_features=f[1])
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(c[1]), nn.Softsign(),  nn.Dropout(p=dropout[0]))
        self.feat2 = AttnFeatTrans1dSoft(in_channels=c[1], in_features=f[1], out_channels=c[2], out_features=f[2])
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(c[2]), nn.Softsign(),  nn.Dropout(p=dropout[1]))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c[2]*f[2], num_class))

    def forward(self, x, neighbor):

        x, neighbor = self.feat1(x, neighbor)
        x, neighbor = self.acvt1(x), [self.acvt1(n) for n in neighbor]
        x, neighbor = self.feat2(x, neighbor)
        x = self.acvt2(x)
        return self.classifier(x)


class KCAT(nn.Module):
    '''
    TODO the locals or the __dict__ cause some issue for net.to(device), the weight of feat didn't load to cuda
    Concate the k level in the last classifier layer
    '''
    def __init__(self, feat_len, num_class, k=1, device='cuda:0'):
        super(KCAT, self).__init__()
        self.k = k
        self.device = device
        c = [1, 4, 32]
        f = [feat_len, 16, 1]
        for k in range(k):
            self.__dict__["feat1%i"%k] = FeatTrans1d(in_channels=c[0], in_features=f[0], out_channels=c[1], out_features=f[1]).to(device)
            self.__dict__["acvt1%i"%k]= nn.Sequential(nn.BatchNorm1d(c[1]), nn.Softsign()).to(device)
            self.__dict__["feat2%i"%k] = FeatTrans1d(in_channels=c[1], in_features=f[1], out_channels=c[2], out_features=f[2]).to(device)
            self.__dict__["acvt2%i"%k] = nn.Sequential(nn.BatchNorm1d(c[2]), nn.Softsign()).to(device)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p=0.1), nn.Linear(c[2]*f[2]*self.k, c[2]*f[2]),nn.ReLU(),nn.Dropout(p=0.1),nn.Linear(c[2]*f[2], num_class))

    def forward(self, x, neighbor):
        x_khops = torch.FloatTensor().to(self.device)
        for k in range(self.k):
            kneighbor = [i[k] for i in neighbor]
            khop, kneighbor = self.__dict__["feat1%i"%k](x, [i[k] for i in neighbor])
            khop, kneighbor = self.__dict__["acvt1%i"%k](khop), [self.__dict__["acvt1%i"%k](item) for item in kneighbor]
            khop, kneighbor = self.__dict__["feat2%i"%k](khop, kneighbor)
            khop = self.__dict__["acvt2%i"%k](khop)
            x_khops = torch.cat((x_khops, khop), dim = 1)
        return self.classifier(x_khops)


class KLGL(nn.Module):
    def __init__(self, feat_len, num_class, k=1):
        super(KLGL, self).__init__()
        # x: (N,f); adj:(N, k, f, f)
        c = [1, 4, 32]
        f = [feat_len, 16, 1]
        self.feat1 = FeatTransKhop(in_channels=c[0], in_features=f[0], out_channels=c[1], out_features=f[1], khop = k)
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(c[1]), nn.Softsign())
        self.feat2 = FeatTransKhop(in_channels=c[1], in_features=f[1], out_channels=c[2], out_features=f[2], khop = k)
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(c[2]), nn.Softsign())
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p=0.1), nn.Linear(c[2]*f[2], num_class))

    def forward(self, x, neighbor):
        x, neighbor = self.feat1(x, neighbor)
        x, neighbor = self.acvt1(x), [[self.acvt1(k) for k in item] for item in neighbor]
        x, neighbor = self.feat2(x, neighbor)
        x = self.acvt2(x)
        return self.classifier(x)


class LifelongRehearsal(nn.Module):
    def __init__(self, args, BackBone, feat_len, num_class, k = None, hidden = [64,32], drop = [0,0]):
        super(LifelongRehearsal, self).__init__()
        self.args = args
        if not k:
            self.backbone = BackBone(feat_len, num_class, hidden = hidden, dropout = drop)
        else:
            self.backbone = BackBone(feat_len, num_class, k=k, hidden = hidden, dropout = drop)
        self.backbone = self.backbone.to(args.device)
        self.register_buffer('adj', torch.zeros(1, feat_len, feat_len))
        self.register_buffer('inputs', torch.Tensor(0, 1, feat_len))
        self.register_buffer('targets', torch.LongTensor(0))
        self.neighbor = []
        self.sample_viewed = 0
        self.memory_order = torch.LongTensor()
        self.memory_size = self.args.memory_size
        self.criterion = nn.CrossEntropyLoss()
        self.running_loss = 0
        exec('self.optimizer = torch.optim.%s(self.parameters(), lr=%f)'%(args.optm, args.lr))

    def forward(self, inputs, neighbor):
        return self.backbone(inputs, neighbor)

    def observe(self, inputs, targets, neighbor, replay=True):
        self.train()
        self.sample(inputs, targets, neighbor)
        
        for i in range(self.args.iteration):
            self.optimizer.zero_grad()
            inputs, targets, neighbor = self.todevice(inputs, targets, neighbor, device = self.args.device)
            outputs = self.forward(inputs, neighbor)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
        self.running_loss+=loss

        if replay:
            L = torch.randperm(self.inputs.size(0))
            minibatches = [L[n:n+self.args.batch_size] for n in range(0, len(L), self.args.batch_size)]
            for index in minibatches:
                self.optimizer.zero_grad()
                inputs, targets, neighbor = self.inputs[index], self.targets[index], [self.neighbor[i] for i in index.tolist()]
                inputs, targets, neighbor = self.todevice(inputs, targets, neighbor, device = self.args.device)
                outputs = self.forward(inputs, neighbor)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

    def todevice(self, inputs, targets, neighbor, device="cpu"):
        inputs, targets = inputs.to(device), targets.to(device)
        ## take the neighbor with k
        if not self.args.k:
            neighbor = [element.to(device) for element in neighbor]
        else:
            neighbor = [[element.to(device) for element in item]for item in neighbor]
        return inputs, targets, neighbor

    @torch.no_grad()
    def uniform_sample(self, inputs, targets, neighbor):
        inputs, targets, neighbor = self.todevice(inputs, targets, neighbor)
        self.inputs = torch.cat((self.inputs, inputs), dim=0)
        self.targets = torch.cat((self.targets, targets), dim=0)
        self.neighbor += neighbor

        if self.inputs.size(0) > self.args.memory_size:
            idx = torch.randperm(self.inputs.size(0))[:self.args.memory_size]
            self.inputs, self.targets = self.inputs[idx], self.targets[idx]
            self.neighbor = [self.neighbor[i] for i in idx.tolist()]

    @torch.no_grad()
    def sample(self, inputs, targets, neighbor):
        self.sample_viewed += inputs.size(0)
        self.memory_order += inputs.size(0)# increase the order 

        self.targets = torch.cat((self.targets, targets), dim=0)
        self.inputs = torch.cat((self.inputs,inputs), dim = 0)
        self.memory_order = torch.cat((self.memory_order, torch.LongTensor(list(range(inputs.size()[0]-1,-1,-1)))), dim = 0)# for debug
        self.neighbor += neighbor

        node_len = int(self.inputs.size(0))
        ext_memory = node_len - self.memory_size
        if ext_memory > 0:
            mask = torch.zeros(node_len,dtype = bool) # mask inputs order targets and neighbor
            reserve = self.memory_size # reserved memrory to be stored
            seg = np.append(np.arange(0,self.sample_viewed,self.sample_viewed/ext_memory),self.sample_viewed)
            for i in range(len(seg)-2,-1,-1):
                left = self.memory_order.ge(np.ceil(seg[i]))*self.memory_order.lt(np.floor(seg[i+1]))
                leftindex = left.nonzero()
                if leftindex.size()[0] > reserve/(i+1): # the quote is not enough, need to be reduced
                    leftindex = leftindex[torch.randperm(leftindex.size()[0])[:int(reserve/(i+1))]] # reserve the quote
                    mask[leftindex] = True
                else:
                    mask[leftindex] = True # the quote is enough
                reserve -= leftindex.size()[0] # deducte the quote
            self.inputs = self.inputs[mask]
            self.targets = self.targets[mask]
            self.memory_order = self.memory_order[mask]
            self.neighbor = [self.neighbor[i] for i in mask.nonzero()]
