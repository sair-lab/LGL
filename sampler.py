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

import torch
import torch.utils.data as Data
from torch.utils.data import RandomSampler, SequentialSampler, Sampler, BatchSampler


class PlainSampler(BatchSampler):
    def __init__(self, data, batch_size, drop_last=False, shuffle=True):
        self.shuffle = shuffle
        sampler = RandomSampler(data) if shuffle is True else SequentialSampler(data)
        super().__init__(sampler, batch_size, drop_last)

    def step(self, score):
        pass

    def __repr__(self):
        return 'PlainSampler(batch_size={}, drop_last={}, shuffle={})'.format(self.batch_size, self.drop_last, self.shuffle)


if __name__ == '__main__':
    from torch.utils.data import Dataset, DataLoader

    class MyData(Dataset):
        def __init__(self, length=20):
            super().__init__()
            self.length = length
            self.data = torch.LongTensor(list(range(length)))

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return self.data[idx], 1

    data = MyData()
    batch_sampler = PlainSampler(data, batch_size=7, shuffle=True)
    # batch_sampler = EarlySampler(data, epochs=[2, 6, 8], batch_size=10, shuffle=True)
    loader = DataLoader(dataset=data, batch_sampler=batch_sampler, num_workers=0)

    for i in range(20):
        for batch_idx, (inputs, targets) in enumerate(loader):
            loss = torch.rand(inputs.size(0))
            batch_sampler.step(loss)
            print('epoch %d batch %d'%(i,batch_idx), inputs)
