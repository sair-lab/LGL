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
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=10, help="minibatch size")
    parser.add_argument("--iteration", type=int, default=5, help="number of training iteration")
    parser.add_argument("--memory-size", type=int, default=100, help="number of samples")
    parser.add_argument("--momentum", type=float, default=0, help="momentum of SGD optimizer")
    parser.add_argument("--adj-momentum", type=float, default=0.9, help="momentum of the feature adjacency")
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)

    train_data = continuum(root=args.data_root, name=args.dataset, data_type='train', download=True)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate)
    test_data = continuum(root=args.data_root, name=args.dataset, data_type='test', download=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate)

    if args.load is not None:
        net = torch.load(args.load, map_location=args.device)
    else:
        Model = Net if args.dataset.lower() in ['cora', 'citeseer', 'pubmed'] else LifelongLGL
        net = Model(args, feat_len=train_data.feat_len, num_class=train_data.num_class).to(args.device)
        for batch_idx, (inputs, targets, neighbor) in enumerate(tqdm.tqdm(train_loader)):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            neighbor = [item.to(args.device) for item in neighbor]
            net.observe(inputs, targets, neighbor)

        if args.save is not None:
            torch.save(net, args.save)

    train_acc, test_acc = performance(train_loader, net, args.device),  performance(test_loader, net, args.device)
    print("Train Acc: %.3f, Test Acc: %.3f"%(train_acc, test_acc))
