
#!/usr/bin/env python3

import torch
import numpy as np
import torch.nn as nn


class SAGE(nn.Module):
    '''
    GraphSAGE: Inductive Representation Learning on Large Graphs, NIPS 2017
    https://arxiv.org/pdf/1706.02216.pdf
    '''
    def __init__(self, feat_len, num_class, hidden=[128,128], dropout=[0,0], aggr='gcn', k=1):
        super().__init__()
        aggrs = {'pool':PoolAggregator, 'mean':MeanAggregator, 'gcn':GCNAggregator}
        Aggregator = aggrs[aggr]
        self.tran1 = Aggregator(feat_len, hidden[0])
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(1), nn.ReLU())
        self.tran2 = Aggregator(hidden[0], hidden[1])
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(1), nn.ReLU())
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(hidden[1], num_class))

    def forward(self, x, neighbor):
        ## the neighbor should be (N,n,c,f)
        x, neighbor = self.tran1(x, neighbor)
        x, neighbor = self.acvt1(x), [self.acvt1(n) for n in neighbor]
        x, neighbor = self.tran2(x, neighbor)
        return self.classifier(self.acvt2(x))


class GCNAggregator(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.tran = nn.Linear(in_features, out_features, False)
    
    def forward(self, x, neighbor):
        f = torch.cat([n.mean(dim=0, keepdim=True) for n in neighbor])
        x = self.tran(x+f)
        neighbor = [self.tran(n) for n in neighbor]
        return x, neighbor


class MeanAggregator(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.tranx = nn.Linear(in_features, out_features, False)
        self.trann = nn.Linear(in_features, out_features, False)

    def forward(self, x, neighbor):
        f = torch.cat([n.mean(dim=0, keepdim=True) for n in neighbor])
        x = self.tranx(x) + self.trann(f)
        neighbor = [self.tranx(n) for n in neighbor]
        return x, neighbor


class PoolAggregator(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.tran = nn.Linear(in_features, out_features, True)

    def forward(self, x, neighbor):
        f = [self.tran(torch.cat([x[i:i+1], n])) for i, n in enumerate(neighbor)]
        x = torch.cat([x.max(dim=0, keepdim=True)[0] for x in f])
        neighbor = [self.tran(n).max(dim=0, keepdim=True)[0] for n in neighbor]
        return x, neighbor


class LifelongSAGE(SAGE):
    def __init__(self, args, feat_len, num_class, k=1):
        super().__init__(feat_len, num_class)
        self.args = args
        self.register_buffer('adj', torch.zeros(1, feat_len, feat_len))
        self.register_buffer('inputs', torch.Tensor(0, 1, feat_len))
        self.register_buffer('targets', torch.LongTensor(0))
        self.neighbor = []
        self.sample_viewed = 0
        self.memory_order = torch.LongTensor()
        self.memory_size = self.args.memory_size
        self.criterion = nn.CrossEntropyLoss()
        exec('self.optimizer = torch.optim.%s(self.parameters(), lr=%f)'%(args.optm, args.lr))

    def observe(self, inputs, targets, neighbor, reply=True):
        self.train()
        for i in range(self.args.iteration):
            self.optimizer.zero_grad()
            outputs = self.forward(inputs, neighbor)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

        self.sample(inputs, targets, neighbor)
        if reply:
            L = torch.randperm(self.inputs.size(0))
            minibatches = [L[n:n+self.args.batch_size] for n in range(0, len(L), self.args.batch_size)]
            for index in minibatches:
                self.optimizer.zero_grad()
                inputs, targets, neighbor = self.inputs[index], self.targets[index], [self.neighbor[i] for i in index.tolist()]
                outputs = self.forward(inputs, neighbor)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def uniform_sample(self, inputs, targets, neighbor):
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