#!/usr/bin/env python3

import copy
import torch
from torch import nn


class EWCLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fisher = [0 for p in model.parameters() if p.requires_grad]
        self.criterion = nn.CrossEntropyLoss()
        self.update(model)

    def update(self, model):
        self.model = copy.deepcopy(model)
        self.weights = copy.deepcopy(self.fisher)
        self.num, self.fisher = 0, [0 for p in model.parameters() if p.requires_grad]

    def diag_fisher(self, inputs:list):
        self.model.zero_grad()
        output = self.model(*inputs)
        label = output.max(1)[1]
        loss = self.criterion(output, label)
        loss.backward()
        fisher = [p.grad.data**2 for p in self.model.parameters()]
        num = self.num
        self.num += inputs[0].size(0) # batch-size
        self.fisher = [(self.fisher[n]*num+w)/self.num for n, w in enumerate(fisher)]

    def forward(self, model, inputs:list):
        self.diag_fisher(inputs)
        return sum([(self.weights[n] * ((p1-p2)**2)).sum()
                        for n, (p1, p2) in enumerate(zip(model.parameters(), self.model.parameters()))
                            if p1.requires_grad and p2.requires_grad])


if __name__ == "__main__":
    '''
    Usage Sample
    '''
    from torchvision import models
    from torchvision import datasets
    from torchvision import transforms
    import torch.utils.data as Data

    class LeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Sequential(nn.Conv2d(1,  6, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
            self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5, 1, 0), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten())
            self.fc = nn.Linear(16 * 5 * 5, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            return self.fc(x)

    device, alpha = 'cuda:0', 5000
    net = LeNet().to(device)
    train_data = datasets.MNIST('/data/datasets', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=10, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.003, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15], gamma=0.1)
    ewcloss = EWCLoss(net)

    train_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        ewc = ewcloss(net, [inputs])
        loss = criterion(outputs, targets) + alpha * ewc
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        if batch_idx % 100 == 0:
            ewcloss.update(net)

    print(train_loss/(batch_idx+1), 100.*correct/total)
