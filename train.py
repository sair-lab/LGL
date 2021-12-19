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
import configargparse
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

from models import SAGE, GCN, APPNP, MLP, GAT, APP
from models import LGL, AFGN, PlainNet, AttnPlainNet
from models import KTransCAT, AttnKTransCAT
from datasets import continuum, graph_collate
from torch_util import count_parameters, EarlyStopScheduler, performance
import time 

## AFGN is LGL with attention; AttnPlainNet is the PlainNet with attention
nets = {'sage':SAGE, 'lgl': LGL, 'ktranscat':KTransCAT, 'attnktranscat':AttnKTransCAT, 'gcn':GCN, 'appnp':APPNP, 'app':APP, 'mlp':MLP, 'gat':GAT, 'afgn':AFGN, 'plain':PlainNet, 'attnplain':AttnPlainNet}

def train(loader, net, criterion, optimizer, device):
    net.train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets, neighbor) in enumerate(tqdm.tqdm(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        if not args.k:
            neighbor = [element.to(device) for element in neighbor]
        else:
            neighbor = [[element.to(device) for element in item]for item in neighbor]

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
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="learning rate")
    parser.add_argument("--model", type=str, default='LGL', help="LGL or SAGE")
    parser.add_argument("--dataset", type=str, default='cora', help="cora, citeseer, pubmed")
    parser.add_argument("--load", type=str, default=None, help="load pretrained model file")
    parser.add_argument("--save", type=str, default=None, help="model file to save")
    parser.add_argument("--optm", type=str, default='SGD', help="SGD or Adam")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--factor", type=float, default=0.1, help="ReduceLROnPlateau factor")
    parser.add_argument("--min-lr", type=float, default=0.001, help="minimum lr for ReduceLROnPlateau")
    parser.add_argument("--patience", type=int, default=3, help="patience for Early Stop")
    parser.add_argument("--batch-size", type=int, default=10, help="number of minibatch size")
    parser.add_argument("--milestones", type=int, default=15, help="milestones for applying multiplier")
    parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--early-stop", type=int, default=5, help="number of epochs for early stop training")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum of the optimizer")
    parser.add_argument("--gamma", type=float, default=0.1, help="learning rate multiplier")
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    parser.add_argument("--eval", type=str, default=None, help="the path to eval the acc")
    parser.add_argument("--k", type=int, default=None, help='khop.')
    parser.add_argument("--hidden", type=int, nargs="+", default=[10,10])
    parser.add_argument("--drop", type=float, nargs="+", default=[0,0])
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    # Datasets
    train_data = continuum(root=args.data_root, name=args.dataset, data_type='train', download=True, k_hop=args.k)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=graph_collate, drop_last=True)
    test_data = continuum(root=args.data_root, name=args.dataset, data_type='test', download=True, k_hop=args.k)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate, drop_last=True)
    valid_data = continuum(root=args.data_root, name=args.dataset, data_type='valid', download=True, k_hop=args.k)
    valid_loader = Data.DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate, drop_last=True)

    Net = nets[args.model.lower()]

    if args.model.lower() in ['ktranscat', 'attnktranscat']:
        net = Net(feat_len=train_data.feat_len, k=args.k, num_class=train_data.num_class).to(args.device)
    else:
        assert(args.k == None)
        net = Net(feat_len=train_data.feat_len, num_class=train_data.num_class, hidden = args.hidden, dropout = args.drop).to(args.device)
    print(net)

    if args.load is not None:
        net.load_state_dict(torch.load(args.load, map_location=args.device))
        train_acc, test_acc, valid_acc = performance(train_loader, net, args.device, args.k),  performance(test_loader, net, args.device, args.k), performance(valid_loader, net, args.device, args.k)
        print("Train Acc: %.5f, Test Acc: %.5f, Valid Acc: %.5f"%(train_acc, test_acc, valid_acc))
        exit()

    criterion = nn.CrossEntropyLoss()
    exec('optimizer = torch.optim.%s(net.parameters(), lr=%f)'%(args.optm, args.lr))
    scheduler = EarlyStopScheduler(optimizer, factor=args.factor, verbose=True, min_lr=args.min_lr, patience=args.patience)

    # Training
    paramsnumber = count_parameters(net)
    print('number of parameters:', paramsnumber)
    if args.eval:
        with open(args.eval+'-acc.txt','w') as file:
            file.write(str(args) + " number of prarams " + str(paramsnumber) + "\n")
            file.write("epoch | train_acc | test_acc | valid_acc |\n")

    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train(train_loader, net, criterion, optimizer, args.device)
        test_acc = performance(test_loader, net, args.device, args.k) # validate
        print("epoch: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f"
                % (epoch, train_loss, train_acc, test_acc))
        if args.eval:
            with open(args.eval+'-acc.txt','a') as file:
                file.write((str([epoch, train_acc, test_acc])+'\n').replace('[','').replace(']',''))  

        if test_acc > best_acc:
            print("New best Model, copying...")
            best_acc, best_net = test_acc, copy.deepcopy(net)

            if args.save is not None:
                torch.save(best_net, args.save)

        if scheduler.step(error=1-test_acc):
            print('Early Stopping!')
            break

    train_acc, test_acc, valid_acc = performance(train_loader, best_net, args.device, args.k), performance(test_loader, best_net, args.device, args.k), performance(valid_loader, best_net, args.device, args.k)
    print('train_acc: %.4f, test_acc: %.4f, valid_acc: %.4f'%(train_acc, test_acc, valid_acc))

    if args.eval:
        with open(args.eval+'-acc.txt','a') as file:
            file.write((str([epoch, train_acc, test_acc, valid_acc])+'\n').replace('[','').replace(']',''))