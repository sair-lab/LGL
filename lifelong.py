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
import configargparse
import warnings
import numpy as np
import torch.nn as nn
import torch.utils.data as Data

from models import SAGE, GCN, MLP, GAT, APP, APPNP
from models import LGL, AFGN, PlainNet, AttnPlainNet, KTransCAT, AttnKTransCAT
from models import LifelongRehearsal
from datasets import continuum, graph_collate
from torch_util import count_parameters, Timer, accuracy, performance

sys.path.append('models')
warnings.filterwarnings("ignore")

nets = {'sage':SAGE, 'lgl': LGL, 'afgn': AFGN, 'ktranscat':KTransCAT, 'attnktranscat':AttnKTransCAT, 'gcn':GCN, 'appnp':APPNP, 'app':APP, 'mlp':MLP, 'gat':GAT, 'plain':PlainNet, 'attnplain':AttnPlainNet}

if __name__ == "__main__":

    # Arguements
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
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
    parser.add_argument("--seed", type=int, default=1, help='Random seed.')
    parser.add_argument("-p", "--plot", action="store_true", help="increase output verbosity")
    parser.add_argument("--eval", type=str, default=None, help="the path to eval the acc")
    parser.add_argument("--sample-rate", type=int, default=50, help="sampling rate for test acc, if ogb datasets please set it to 200")
    parser.add_argument("--k", type=int, default=None, help='the level of k hop.')
    parser.add_argument("--hidden", type=int, nargs="+", default=[64,32])
    parser.add_argument("--drop", type=float, nargs="+", default=[0,0])
    parser.add_argument("--merge", type=int, default=1, help='Merge some class if needed.')
    args = parser.parse_args(); print(args)
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(args.seed)

    train_data = continuum(root=args.data_root, name=args.dataset, data_type='train', download=True, k_hop = args.k)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate, drop_last=True)

    test_data = continuum(root=args.data_root, name=args.dataset, data_type='test', download=True ,k_hop = args.k)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate, drop_last=True)

    valid_data = continuum(root=args.data_root, name=args.dataset, data_type='valid', download=True, k_hop = args.k)
    valid_loader = Data.DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate, drop_last=True)

    Net = nets[args.model.lower()]
    if args.model.lower() in ['ktranscat', 'ktranscat']:
        net = LifelongRehearsal(args, Net, feat_len=test_data.feat_len, num_class=test_data.num_class, k = args.k, hidden = args.hidden, drop = args.drop)
    else:
        net = LifelongRehearsal(args, Net, feat_len=test_data.feat_len, num_class=test_data.num_class, hidden = args.hidden, drop = args.drop)
    evaluation_metrics = []
    num_parameters = count_parameters(net)
    print('number of parameters:', num_parameters)

    if args.load is not None:
        net.backbone.load_state_dict(torch.load(args.load, map_location=args.device))
        train_acc, test_acc, valid_acc = performance(train_loader, net, args.device, k=args.k),  performance(test_loader, net, args.device, k=args.k), performance(valid_loader, net, args.device, k=args.k)
        print("Train Acc: %.3f, Test Acc: %.3f, Valid Acc: %.3f"%(train_acc, test_acc, valid_acc))
        exit()

    if args.eval:
        with open(args.eval+'-acc.txt','a') as file:
            file.write(str(args) + " number of prarams " + str(num_parameters) + "\n")
            file.write("epoch | train_acc | test_acc | valid_acc |\n")

    task_ids = [i for i in range(test_data.num_class)]
    for i in range(0, test_data.num_class, args.merge):
        ## merge the class if needed
        if (i+args.merge > test_data.num_class):
            tasks_list = task_ids[i:test_data.num_class]
        else:
            tasks_list = task_ids[i:i+args.merge]

        incremental_data = continuum(root=args.data_root, name=args.dataset, data_type='incremental', download=True, task_type = tasks_list, k_hop = args.k)
        incremental_loader = Data.DataLoader(dataset=incremental_data, batch_size=args.batch_size, shuffle=True, collate_fn=graph_collate, drop_last=True)

        for batch_idx, (inputs, targets, neighbor) in enumerate(tqdm.tqdm(incremental_loader)):
            net.observe(inputs, targets, neighbor, batch_idx%args.jump==0)

        train_acc, test_acc = performance(incremental_loader, net, args.device, k=args.k), performance(test_loader, net, args.device, k=args.k)
        evaluation_metrics.append([i, len(incremental_data), train_acc, test_acc])
        print("Train Acc: %.3f, Test Acc: %.3f"%(train_acc, test_acc))

        if args.save is not None:
            torch.save(net.backbone.state_dict(), args.save)

        if args.eval: 
            with open(args.eval+'-acc.txt','a') as file:
                file.write((str([i, train_acc, test_acc])+'\n').replace('[','').replace(']',''))

    evaluation_metrics = torch.Tensor(evaluation_metrics)
    print('        | task | sample | train_acc | test_acc |')
    print(evaluation_metrics)

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
        train_data = continuum(root=args.data_root, name=args.dataset, data_type='train', download=True, k_hop = args.k)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate, drop_last=True)
        test_acc, train_acc = performance(test_loader, net, args.device, k = args.k), performance(train_loader, net, args.device, k = args.k)
        valid_acc = performance(valid_loader, net, args.device, k=args.k)
        with open(args.eval+'-acc.txt','a') as file:
            file.write('number of parameters:%i\n'%num_parameters)
            file.write('| task | train_acc | test_acc | valid_acc |\n')
            file.write((str([i, train_acc, test_acc, valid_acc])).replace('[','').replace(']',''))
