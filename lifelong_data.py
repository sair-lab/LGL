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

import os
import sys
import tqdm
import copy
import torch
import warnings
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data as Data

from datasets import continuum
from lifelong import performance
from datasets import graph_collate
from models import Net, LifelongLGL
from models import LifelongSAGE
from torch_util import count_parameters

sys.path.append('models')
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="dataset location")
    parser.add_argument("--dataset", type=str, default='cora', help="cora, citeseer, or pubmed")
    parser.add_argument("--load", type=str, default=None, help="load pretrained model file")
    parser.add_argument("--save", type=str, default=None, help="model file to save")
    parser.add_argument("--optm", type=str, default='SGD', help="SGD or Adam")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--model", type=str, default='LGL', help="LGL or SAGE")
    parser.add_argument("--batch-size", type=int, default=10, help="minibatch size")
    parser.add_argument("--iteration", type=int, default=5, help="number of training iteration")
    parser.add_argument("--memory-size", type=int, default=100, help="number of samples")
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    parser.add_argument("--eval", type=str, default=None, help="the path to eval the acc")
    parser.add_argument("--sample-rate", type=int, default=50, help="sampling rate for test acc, if ogb datasets please set it to 200")
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)

    train_data = continuum(root=args.data_root, name=args.dataset, data_type='train', download=True)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate, drop_last=True)
    test_data = continuum(root=args.data_root, name=args.dataset, data_type='test', download=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate, drop_last=True)
    if not args.dataset in ["cora", "citeseer", "pubmed"]:
        valid_data = continuum(root=args.data_root, name=args.dataset, data_type='valid', download=True)
        valid_loader = Data.DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate, drop_last=True)

    if args.eval:
        with open(args.eval+'.txt','w') as file:
            file.write(str(args))

    if args.load is not None:
        net = torch.load(args.load, map_location=args.device)
    else:
        Model = Net if args.dataset.lower() in ['cora', 'citeseer', 'pubmed'] else LifelongLGL
        nets = {'sage':LifelongSAGE, 'lgl': Model, 'plain': Net}
        Net = nets[args.model.lower()]
        net = Net(args, feat_len=test_data.feat_len, num_class=test_data.num_class).to(args.device)
        for batch_idx, (inputs, targets, neighbor) in enumerate(tqdm.tqdm(train_loader)):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            neighbor = [item.to(args.device) for item in neighbor]
            net.observe(inputs, targets, neighbor)

            if args.eval and batch_idx%args.sample_rate*10 == 0:
                test_acc = performance(test_loader, net, args.device)
#                 performance(train_loader, net, args.device)
                with open(args.eval+'-acc.txt','a') as file:
                    file.write((str([batch_idx*args.batch_size, test_acc])+'\n').replace('[','').replace(']',''))

        if args.save is not None:
            torch.save(net, args.save)

    test_acc, train_acc, valid_acc = performance(test_loader, net, args.device), performance(train_loader, net, args.device), performance(valid_loader, net, args.device)
    print("Train Acc: %.3f, Test Acc: %.3f, Valid Acc: %.3f"%(train_acc, test_acc, valid_acc))

    if args.eval:
        valid_acc = performance(valid_loader, net, args.device)
        with open(args.eval+'-acc.txt','a') as file:
            file.writ("| sample | train_acc | test_acc | valid_acc |\n")
            file.write((str([batch_idx*args.batch_size, train_acc, test_acc, valid_acc])+'\n').replace('[','').replace(']',''))