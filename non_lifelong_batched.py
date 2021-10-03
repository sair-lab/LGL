import os
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
from models import KTransCAT, AttnKTransCAT, AttnAPPTrans, AttnAPPNPTrans
from torch_util import count_parameters, EarlyStopScheduler
from torchvision.datasets import VisionDataset

import time, csv

class BathedDataset(VisionDataset):
    def __init__(self, data_root='data/T2D_sample_v2.csv', data_type='train'):
        super(BathedDataset, self).__init__(data_root)
        
        data = csv.reader(open(data_root, "r"))
        self.data = np.array(list(data))

        self.features = torch.FloatTensor(self.data[1:,1:-1].astype("float"))
        self.labels = torch.LongTensor(self.data[1:,-1].astype("int"))
        self.ids = self.data[1:,0]
        self.variables = self.data[0,1:-1]
        
        self.num_class = len(np.unique(self.labels))
        self.feat_len = len(self.variables)
        
        num_dataset = len(self.labels)
        
        ### TODO define your own normalization method
        self.features -= self.features.min(1, keepdim=True)[0]
        self.features /= self.features.max(1, keepdim=True)[0]
        
        ### TODO define your own mask for train, test and val
        self.mask = np.zeros(num_dataset, dtype = bool)
        num_train = int(num_dataset * 0.5)
        num_test = int(num_dataset *0.25)
        num_valid = num_dataset - num_train - num_test

        if data_type == 'train': # data incremental; use test and train as train
            self.mask[:num_train] = True
        elif data_type == 'test':
            self.mask[num_train: num_train + num_test] = True
        elif data_type == 'valid':
            self.mask[-num_valid:] = True
        else:
            raise RuntimeError('data type {} wrong'.format(data_type))

    def __len__(self):
        return len(self.labels[self.mask])

    def __getitem__(self, index):
        '''
         a naive implementation for the dataset 
        '''
        neighbor = self.features[self.mask].unsqueeze(-2) ## fully connected graph
        return self.features[self.mask][index].unsqueeze(-2), self.labels[self.mask][index], neighbor

    def get_neighbor(self, ids):
        return self.features[self.mask].unsqueeze(-2)

## AFGN is LGL with attention; AttnPlainNet is the PlainNet with attention
nets = {'sage':SAGE, 'lgl': LGL, 'ktranscat':KTransCAT, 'attnktranscat':AttnKTransCAT, 'gcn':GCN, 'appnp':APPNP, 'app':APP, 'mlp':MLP, 'gat':GAT, 'afgn':AFGN, 'plain':PlainNet, 'attnplain':AttnPlainNet}

def graph_collate(batch):
    feature = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    neighbor = [item[2] for item in batch]
    return [feature, labels, neighbor]

def performance(loader, net, device):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, neighbor) in enumerate(loader):
            if torch.cuda.is_available():
                inputs, targets = inputs.to(device), targets.to(device)
                if not args.k:
                    neighbor = [element.to(device) for element in neighbor]
                else:
                    neighbor = [[element.to(device) for element in item]for item in neighbor]

            outputs = net(inputs, neighbor)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()
        acc = correct/total
    return acc

def train(loader, net, criterion, optimizer, device):
    net.train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets, neighbor) in enumerate(loader):
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
    parser.add_argument("--data-root", type=str, default='data/T2D_sample_v2.csv', help="learning rate")
    parser.add_argument("--model", type=str, default='GCN', help="LGL or SAGE")
    parser.add_argument("--load", type=str, default=None, help="load pretrained model file")
    parser.add_argument("--save", type=str, default=None, help="model file to save")
    parser.add_argument("--optm", type=str, default='SGD', help="SGD or Adam")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    parser.add_argument("--min-lr", type=float, default=0.00001, help="minimum lr for ReduceLROnPlateau")
    parser.add_argument("--patience", type=int, default=20, help="patience for Early Stop")
    parser.add_argument("--batch-size", type=int, default=10, help="number of minibatch size")
    parser.add_argument("--milestones", type=int, default=15, help="milestones for applying multiplier")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--early-stop", type=int, default=100, help="number of epochs for early stop training")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum of the optimizer")
    parser.add_argument("--gamma", type=float, default=0.1, help="learning rate multiplier")
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    parser.add_argument("--eval", type=str, default=None, help="the path to eval the acc")
    parser.add_argument("--k", type=int, default=None, help='khop.')
    parser.add_argument("--hidden", type=int, nargs="+", default=[128,128])
    parser.add_argument("--drop", type=float, nargs="+", default=[0.2,0.1])
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    # Datasets
    train_data = BathedDataset(data_root=args.data_root, data_type='train')
    train_loader = Data.DataLoader(dataset=train_data, batch_size=train_data.__len__(), shuffle=True, collate_fn=graph_collate)
    test_data = BathedDataset(data_root=args.data_root, data_type='test')
    test_loader = Data.DataLoader(dataset=test_data, batch_size=test_data.__len__(), shuffle=False, collate_fn=graph_collate)
    valid_data = BathedDataset(data_root=args.data_root, data_type='valid')
    valid_loader = Data.DataLoader(dataset=valid_data, batch_size=valid_data.__len__(), shuffle=False, collate_fn=graph_collate)
    
    if args.load is not None:
        net = torch.load(args.load, map_location=args.device)
        train_acc, test_acc, valid_acc = performance(train_loader, net, args.device),  performance(test_loader, net, args.device), performance(valid_loader, net, args.device)
        print("Train Acc: %.5f, Test Acc: %.5f, Valid Acc: %.5f"%(train_acc, test_acc, valid_acc))
        exit()
    else:
        Net = nets[args.model.lower()]
        
        if args.model.lower() in ['ktranscat', 'attnktranscat']:
            net = Net(feat_len=train_data.feat_len, k=args.k, num_class=train_data.num_class).to(args.device)
        else:
            assert(args.k == None)
            net = Net(feat_len=train_data.feat_len, num_class=train_data.num_class, hidden = args.hidden, dropout = args.drop).to(args.device)

    print(net)

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
        test_acc = performance(test_loader, net, args.device) # validate
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

    train_acc, test_acc, valid_acc = performance(train_loader, best_net, args.device), performance(test_loader, best_net, args.device), performance(valid_loader, best_net, args.device)
    print('train_acc: %.4f, test_acc: %.4f, valid_acc: %.4f'%(train_acc, test_acc, valid_acc))
        
    if args.eval:
        with open(args.eval+'-acc.txt','a') as file:
            file.write((str([epoch, train_acc, test_acc, valid_acc])+'\n').replace('[','').replace(']',''))
