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
import copy
import math
import torch
import os.path
import argparse
import numpy as np
import torch.nn as nn
from model import FGN
from dataset import Citation
import torch.utils.data as Data
from torch.autograd import Variable

from sampler import PlainSampler
from scheduler import EarlyStopScheduler

def train(loader, net, criterion, optimizer):
    train_loss, correct, total = 0, 0, 0

    for batch_idx, (inputs, targets, adj) in enumerate(loader):
        if torch.cuda.is_available():
            inputs, targets, adj = inputs.cuda(), targets.cuda(), adj.cuda().to_dense()
        optimizer.zero_grad()
        outputs = net(inputs, adj)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

    return (train_loss/(batch_idx+1), 100.*correct/total)


def performance(loader, net, criterion):
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, adj) in enumerate(loader):
            if torch.cuda.is_available():
                inputs, targets, adj = inputs.cuda(), targets.cuda(), adj.cuda().to_dense()
            outputs = net(inputs, adj)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()
        acc = 100.*correct/total

    return (test_loss/(batch_idx+1),acc)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Arguements
    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="learning rate")
    parser.add_argument("--dataset", type=str, default='cora', help="dataset name")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate, use 0.01 for citeseer")
    parser.add_argument("--min-lr", type=float, default=0.01, help="learning rate, use 0.01 for citeseer")
    parser.add_argument("--batch-size", type=int, default=10, help="number of minibatch size")
    parser.add_argument("--milestones", type=int, default=200, help="milestones for applying multiplier")
    parser.add_argument("--epochs", type=int, default=250, help="number of training epochs")
    parser.add_argument("--patience", type=int, default=15, help="number of epochs for early stop training")
    parser.add_argument("--momentum", type=float, default=0, help="momentum of the optimizer")
    parser.add_argument("--factor", type=float, default=0.1, help="learning rate multiplier")
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)

    # Datasets
    train_data = Citation(root=args.data_root,  name=args.dataset, data_type='train', download=True)
    val_data = Citation(root=args.data_root, name=args.dataset, data_type='val', download=True)
    batch_sampler = PlainSampler(train_data, batch_size=args.batch_size, shuffle=True)
    train_loader = Data.DataLoader(dataset=train_data, batch_sampler=batch_sampler, num_workers=0)
    val_loader = Data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)

    # Models
    net = FGN(adjacency=train_data.adjacency)
    if torch.cuda.is_available():
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = EarlyStopScheduler(optimizer, patience=args.patience, factor=args.factor, min_lr=args.min_lr)

    # Training
    print('number of parameters:', count_parameters(net))
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train(train_loader, net, criterion, optimizer)
        val_loss, val_acc = performance(val_loader, net, criterion) # validate
        print("epoch: %d, train_loss: %.4f, train_acc: %.2f, val_loss: %.4f, val_acc: %.2f" 
                % (epoch, train_loss, train_acc, val_loss, val_acc))
        
        if val_acc > best_acc:
            print("New best Model, saving...")
            best_acc, best_net = val_acc, copy.deepcopy(net)

        if scheduler.step(val_acc):
            print("Early Stopping!")
            break

    test_data = Citation(root=args.data_root, name=args.dataset, data_type='test', download=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=20, shuffle=False)
    test_loss, test_acc = performance(test_loader, best_net, criterion)
    print('test_loss, %.4f test_acc: %.2f'%(test_loss, test_acc))
