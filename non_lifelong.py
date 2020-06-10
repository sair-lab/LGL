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
import tqdm
import copy
import torch
import os.path
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

from models import LGL, PlainNet
from lifelong import performance
from datasets import continuum, citation_collate
from torch_util import count_parameters, EarlyStopScheduler


def train(loader, net, criterion, optimizer):
    net.train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets, neighbor) in enumerate(tqdm.tqdm(loader)):
        inputs, targets, neighbor = inputs.to(args.device), targets.to(args.device), [item.to(args.device) for item in neighbor]
        optimizer.zero_grad()
        outputs = net(inputs, neighbor)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

    return (train_loss/(batch_idx+1), correct/total)


if __name__ == '__main__':
    # Arguements
    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="learning rate")
    parser.add_argument("--dataset", type=str, default='flickr', help="cora, citeseer, pubmed")
    parser.add_argument("--save", type=str, default=None, help="model file to save")
    parser.add_argument("--optm", type=str, default='SGD', help="SGD or Adam")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--factor", type=float, default=0.1, help="ReduceLROnPlateau factor")
    parser.add_argument("--min-lr", type=float, default=0.01, help="minimum lr for ReduceLROnPlateau")
    parser.add_argument("--patience", type=int, default=5, help="patience for Early Stop")
    parser.add_argument("--batch-size", type=int, default=10, help="number of minibatch size")
    parser.add_argument("--milestones", type=int, default=15, help="milestones for applying multiplier")
    parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--early-stop", type=int, default=5, help="number of epochs for early stop training")
    parser.add_argument("--momentum", type=float, default=0, help="momentum of the optimizer")
    parser.add_argument("--gamma", type=float, default=0.1, help="learning rate multiplier")
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)

    # Datasets
    train_data = continuum(root=args.data_root, name=args.dataset, data_type='train', download=True)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, collate_fn=citation_collate)
    test_data = continuum(root=args.data_root, name=args.dataset, data_type='test', download=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=citation_collate)

    # Models
    Model = PlainNet if args.dataset.lower() in ['cora', 'citeseer', 'pubmed'] else LGL
    net = Model(feat_len=train_data.feat_len, num_class=train_data.num_class).to(args.device)
    criterion = nn.CrossEntropyLoss()
    exec('optimizer = torch.optim.%s(net.parameters(), lr=%f)'%(args.optm, args.lr))
    scheduler = EarlyStopScheduler(optimizer, factor=args.factor, verbose=True, min_lr=args.min_lr, patience=args.patience)

    # Training
    print('number of parameters:', count_parameters(net))
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train(train_loader, net, criterion, optimizer)
        test_acc = performance(test_loader, net, args.device) # validate
        print("epoch: %d, train_loss: %.4f, train_acc: %.3f, test_acc: %.3f"
                % (epoch, train_loss, train_acc, test_acc))

        if test_acc > best_acc:
            print("New best Model, copying...")
            best_acc, best_net = test_acc, copy.deepcopy(net)

        if scheduler.step(error=1-test_acc):
            print('Early Stopping!')
            break

    train_acc, test_acc = performance(train_loader, best_net, args.device), performance(test_loader, best_net, args.device)
    print('train_acc: %.3f, test_acc: %.3f'%(train_acc, test_acc))

    if args.save is not None:
        torch.save(best_net, args.save)
