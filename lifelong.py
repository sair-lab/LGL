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
import os.path
import argparse
import warnings
import numpy as np
import torch.nn as nn
import torch.utils.data as Data

from models import Net
from models import LifelongLGL
from models import LifelongSAGE
from datasets import continuum
from datasets import graph_collate
from torch_util import count_parameters

sys.path.append('models')
warnings.filterwarnings("ignore")


def performance(loader, net, device):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, neighbor) in enumerate(tqdm.tqdm(loader)):
            if torch.cuda.is_available():
                inputs, targets, neighbor = inputs.to(device), targets.to(device), [item.to(device) for item in neighbor]
            outputs = net(inputs, neighbor)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()
        acc = correct/total
    return acc


def accuracy(net, loader, device, num_class):
    net.eval()
    correct, total = 0, 0
    classes = torch.arange(num_class).view(-1,1).to(device)
    with torch.no_grad():
        for idx, (inputs, targets, neighbor) in enumerate(loader):
            if torch.cuda.is_available():
                inputs, targets, neighbor = inputs.to(device), targets.to(device), [item.to(device) for item in neighbor]
            outputs = net(inputs, neighbor)
            _, predicted = torch.max(outputs.data, 1)
            total += (targets == classes).sum(1)
            corrected = predicted==targets
            correct += torch.stack([corrected[targets==i].sum() for i in range(num_class)])
        acc = correct/total
    return acc


if __name__ == "__main__":

    # Arguements
    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="dataset location")
    parser.add_argument("--dataset", type=str, default='cora', help="cora, citeseer, or pubmed")
    parser.add_argument("--model", type=str, default='LGL', help="LGL or SAGE")
    parser.add_argument("--load", type=str, default=None, help="load pretrained model file")
    parser.add_argument("--save", type=str, default='accuracy/cora-lgl-test', help="model file to save")
    parser.add_argument("--optm", type=str, default='SGD', help="SGD or Adam")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=5, help="minibatch size")
    parser.add_argument("--jump", type=int, default=1, help="reply samples")
    parser.add_argument("--iteration", type=int, default=10, help="number of training iteration")
    parser.add_argument("--memory-size", type=int, default=500, help="number of samples")
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    parser.add_argument("-p", "--plot", action="store_true", help="increase output verbosity")
    parser.add_argument("--eval", type=str, default=None, help="the path to eval the acc")
    parser.add_argument("--sample-rate", type=int, default=50, help="sampling rate for test acc, if ogb datasets please set it to 200")
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)

    test_data = continuum(root=args.data_root, name=args.dataset, data_type='test', download=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate, drop_last=True)

    # for ogn dataset
    if not args.dataset in ["cora", "citeseer", "pubmed"]:
        valid_data = continuum(root=args.data_root, name=args.dataset, data_type='test', download=True)
        valid_loader = Data.DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate, drop_last=True)

    if args.eval:
        with open(args.eval+'.txt','w') as file:
            file.write(str(args)+"\n")

    if args.load is not None:
        train_data = continuum(root=args.data_root, name=args.dataset, data_type='train', download=True)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate, drop_last=True)
        net = torch.load(args.load, map_location=args.device)
        train_acc, test_acc = performance(train_loader, net, args.device),  performance(test_loader, net, args.device)
        print("Train Acc: %.3f, Test Acc: %.3f"%(train_acc, test_acc))
        exit()

    Model = Net if args.dataset.lower() in ['cora', 'citeseer', 'pubmed'] else LifelongLGL
    nets = {'sage':LifelongSAGE, 'lgl': Model, 'plain': Net}
    Net = nets[args.model.lower()]
    net = Net(args, feat_len=test_data.feat_len, num_class=test_data.num_class).to(args.device)
    evaluation_metrics = []

    for i in range(test_data.num_class):
        # hack here to check 18
        incremental_data = continuum(root=args.data_root, name=args.dataset, data_type='incremental', download=True, task_type = i)
        incremental_loader = Data.DataLoader(dataset=incremental_data, batch_size=args.batch_size, shuffle=True, collate_fn=graph_collate, drop_last=True)
        for batch_idx, (inputs, targets, neighbor) in enumerate(tqdm.tqdm(incremental_loader)):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            neighbor = [item.to(args.device) for item in neighbor]
            net.observe(inputs, targets, neighbor, batch_idx%args.jump==0)

        train_acc, test_acc = performance(incremental_loader, net, args.device), performance(test_loader, net, args.device)
        evaluation_metrics.append([i, len(incremental_data), train_acc, test_acc])
        
        if args.eval: 
            incre_acc =  performance(incremental_loader, net, args.device)
            with open(args.eval+'-acc.txt','a') as file:
                file.write((str([i, incre_acc])+'\n').replace('[','').replace(']',''))
        if args.save is not None:
            torch.save(net, args.save)

    evaluation_metrics = torch.Tensor(evaluation_metrics)
    print('        | task | sample | train_acc | test_acc |')
    print(evaluation_metrics)
    num_parameters = count_parameters(net)
    print('number of parameters:', num_parameters)

    if args.save is not None:
        torch.save(net, args.save)

    if args.plot:
        import matplotlib.pyplot as plt
        tasks = evaluation_metrics[:,0]+1
        plt.plot(tasks, evaluation_metrics[:,2],"b-o", label = "train acc")
        plt.plot(tasks, evaluation_metrics[:,3],"r-o", label = "test acc")
        plt.title("datasets: %s memory size: %s lr: %s batch_size: %s"%(args.dataset,args.memory_size, args.lr, args.batch_size))
        plt.legend()
        plt.xlabel("task")
        plt.ylabel("accuracy (%)")
        for i, txt in enumerate(evaluation_metrics[:,1]):
            plt.annotate(int(txt),(tasks[i], evaluation_metrics[:,3][i]))
        plt.savefig("doc/plt.png")

    if args.eval: 
        test_acc, incre_acc = performance(test_loader, net, args.device), performance(incremental_loader, net, args.device)
        valid_acc = performance(valid_loader, net, args.device), performance(incremental_loader, net, args.device)
        with open(args.eval+'-acc.txt','a') as file:
            file.write('number of parameters:%i\n'%num_parameters)
            file.write('| task | train_acc | test_acc | valid_acc |\n')
            file.write((str([i, incre_acc, test_acc, valid_acc])).replace('[','').replace(']',''))
