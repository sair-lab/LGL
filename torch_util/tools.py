#!/usr/bin/env python3

# Copyright <2020> <Chen Wang [https://chenwang.site], Carnegie Mellon University>

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

import time
import torch
import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Timer:
    def __init__(self):
        torch.cuda.synchronize()
        self.start_time = time.time()

    def tic(self):
        self.start()

    def show(self, prefix="", output=True):
        torch.cuda.synchronize()
        duration = time.time()-self.start_time
        if output:
            print(prefix+"%fs" % duration)
        return duration

    def toc(self, prefix=""):
        self.end()
        print(prefix+"%fs = %fHz" % (self.duration, 1/self.duration))
        return self.duration

    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.time()

    def end(self):
        torch.cuda.synchronize()
        self.duration = time.time()-self.start_time
        self.start()
        return self.duration


def performance(loader, net, device, k):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, neighbor) in enumerate(tqdm.tqdm(loader)):
            if torch.cuda.is_available():
                inputs, targets = inputs.to(device), targets.to(device)
                if not k:
                    neighbor = [element.to(device) for element in neighbor]
                else:
                    neighbor = [[element.to(device) for element in item]for item in neighbor]

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
                inputs, targets = inputs.to(device), targets.to(device)
                if not k:
                    neighbor = [element.to(device) for element in neighbor]
                else:
                    neighbor = [[item.to(device) for item in element] for element in neighbor]
            outputs = net(inputs, neighbor)
            _, predicted = torch.max(outputs.data, 1)
            total += (targets == classes).sum(1)
            corrected = predicted==targets
            correct += torch.stack([corrected[targets==i].sum() for i in range(num_class)])
        acc = correct/total
    return acc