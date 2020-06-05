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
import tqdm
import copy
import torch
import os.path
import argparse
import numpy as np
import torch.nn as nn
from models import Net
import torch.utils.data as Data
from continuum import Continuum
from datasets import Citation, citation_collate


def performance(loader, net):
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, neighbor) in enumerate(tqdm.tqdm(loader)):
            if torch.cuda.is_available():
                inputs, targets, neighbor = inputs.to(args.device), targets.to(args.device), [item.to(args.device) for item in neighbor]
            outputs = net(inputs, neighbor)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()
        acc = 100.*correct/total
    return acc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":

    # Arguements
    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="dataset location")
    parser.add_argument("--dataset", type=str, default='cora', help="cora, citeseer, or pubmed")
    parser.add_argument("--pretrained", type=str, default=None, help="pretrained model file")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate, 0.01 for citeseer")
    parser.add_argument("--batch-size", type=int, default=10, help="minibatch size")
    parser.add_argument("--iteration", type=int, default=5, help="number of training iteration")
    parser.add_argument("--memory-size", type=int, default=500, help="number of samples")
    parser.add_argument("--momentum", type=float, default=0, help="momentum of SGD optimizer")
    parser.add_argument("--adj-momentum", type=float, default=0.9, help="momentum of the feature adjacency")
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument("-p", "--plot", action="store_true", help="increase output verbosity")
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)

    test_data = Continuum(root=args.data_root, name=args.dataset, data_type='test', download=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=citation_collate)

    if args.pretrained is not None:
        net = torch.load(args.pretrained, map_location=args.device)
        test_acc = performance(test_loader, net)
        print("Test Accuracy", test_acc)
        exit()

    net = Net(args, feat_len=test_data.feat_len, num_class=test_data.num_class).to(args.device)
    evaluation_metrics = []
    for i in range(test_data.num_class):
        incremental_data = Continuum(root=args.data_root, name=args.dataset, data_type='incremental', download=True, task_type = i)
        incremental_loader = Data.DataLoader(dataset=incremental_data, batch_size=args.batch_size, shuffle=True, collate_fn=citation_collate)
        for batch_idx, (inputs, targets, neighbor) in enumerate(tqdm.tqdm(incremental_loader)):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            neighbor = [item.to(args.device) for item in neighbor]
            net.observe(inputs, targets, neighbor)

        train_acc, test_acc = performance(incremental_loader, net), performance(test_loader, net)
        evaluation_metrics.append([i, len(incremental_data), train_acc, test_acc])

    evaluation_metrics = torch.Tensor(evaluation_metrics)
    print('        | task | sample | train_acc | test_acc |')
    print(evaluation_metrics)
    print('number of parameters:', count_parameters(net))

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
