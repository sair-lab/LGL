

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
from models import EWCLoss
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


if __name__ == "__main__":

    # Arguements
    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="dataset location")
    parser.add_argument("--dataset", type=str, default='cora', help="cora, citeseer, or pubmed")
    parser.add_argument("--model", type=str, default='LGL', help="LGL or SAGE")
    parser.add_argument("--load", type=str, default=None, help="load pretrained model file")
    parser.add_argument("--save", type=str, default=None, help="model file to save")
    parser.add_argument("--optm", type=str, default='SGD', help="SGD or Adam")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--alpha", type=int, default=1000, help="importance to the ewc")
    parser.add_argument("--batch-size", type=int, default=10, help="minibatch size")
    parser.add_argument("--iteration", type=int, default=5, help="number of training iteration")
    parser.add_argument("--memory-size", type=int, default=500, help="number of samples")
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    parser.add_argument("-p", "--plot", action="store_true", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print out loss")
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)

    test_data = continuum(root=args.data_root, name=args.dataset, data_type='test', download=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate)

    if args.load is not None:
        train_data = continuum(root=args.data_root, name=args.dataset, data_type='train', download=True)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate)
        net = torch.load(args.load, map_location=args.device)
        train_acc, test_acc = performance(train_loader, net, args.device),  performance(test_loader, net, args.device)
        print("Train Acc: %.3f, Test Acc: %.3f"%(train_acc, test_acc))
        exit()

    Model = Net if args.dataset.lower() in ['cora', 'citeseer', 'pubmed'] else LifelongLGL
    nets = {'sage':LifelongSAGE, 'lgl': Model}
    Net = nets[args.model.lower()]
    net = Net(args, feat_len=test_data.feat_len, num_class=test_data.num_class).to(args.device)
    evaluation_metrics = []
    ewc = EWCLoss(net)

    for i in range(test_data.num_class):
        incremental_data = continuum(root=args.data_root, name=args.dataset, data_type='incremental', download=True, task_type = i)
        incremental_loader = Data.DataLoader(dataset=incremental_data, batch_size=args.batch_size, shuffle=True, collate_fn=graph_collate)
        for batch_idx, (inputs, targets, neighbor) in enumerate(tqdm.tqdm(incremental_loader)):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            neighbor = [item.to(args.device) for item in neighbor]

            # the training process
            net.train()
            for itr in range(args.iteration):
                net.optimizer.zero_grad()
                outputs = net(inputs, neighbor)
                ewc_loss = args.alpha * ewc(net, [inputs, neighbor])
                loss = net.criterion(outputs, targets) + ewc_loss
                loss.backward()
                net.optimizer.step()
            if args.debug: print("ewc loss: %f lgl loss: %f"%(ewc_loss.item(), loss.item()-ewc_loss.item()))

        ewc.update(net)# update after each task

        train_acc, test_acc = performance(incremental_loader, net, args.device), performance(test_loader, net, args.device)
        evaluation_metrics.append([i, len(incremental_data), train_acc, test_acc])

    evaluation_metrics = torch.Tensor(evaluation_metrics)
    print('        | task | sample | train_acc | test_acc |')
    print(evaluation_metrics)
    print('number of parameters:', count_parameters(net))

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
