# Copyright <2019> <Chen Wang <https://chenwang.site>, Carnegie Mellon University>

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
import torch
import os.path
import numpy as np
import scipy.sparse as sp
from dgl import DGLGraph
from dgl.data import citegrh
from itertools  import compress
from torchvision.datasets import VisionDataset

class Citation(VisionDataset):
    """ `Citation Dataset`.
    Args:
        root (string): Root directory of dataset exist.
        name (string): only one of ``cora``, ``citeseer``, ``pubmed``.
        data_type (string): only one of ``train``, ``val``, and ``test``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    train_file = 'train.pt'
    val_file = 'validate.pt'
    test_file = 'test.pt'

    def __init__(self, root='~/.dgl', name = 'cora', data_type='train', download=True):
        super(Citation, self).__init__(root)
        self.name = name

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Data not found. Use download=True to download it')

        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'val':
                data_file = self.val_file
        elif data_type == 'test':
            data_file = self.test_file
        else:
            raise RuntimeError('data type {} wrong'.format(data_type))

        self.adjacency, (self.features, self.labels), fadjs = torch.load(os.path.join(self.processed_folder, data_file))
        self.fadjs = [torch.sparse_coo_tensor(**it) for it in fadjs]
        print('{} Dataset for {} Loaded.'.format(self.name, data_type))


    def __getitem__(self, index):
        return self.features[index,:], self.labels[index], self.fadjs[index]


    def feature_adjacency(self):
        return self.adjacency


    def download(self):
        """Download data if it doesn't exist in processed_folder already."""
        if self._check_exists():
            return
        
        print('Downloading and Preprocessing {} Dataset...'.format(self.name))
        os.makedirs(self.processed_folder, exist_ok=True)

        os.environ["DGL_DOWNLOAD_DIR"] = self.processed_folder

        if self.name == 'cora':
            data = citegrh.load_cora()
        elif self.name == 'citeseer':
            data = citegrh.load_citeseer()
        elif self.name == 'pubmed':
            data = citegrh.load_pubmed()
        else:
            raise RuntimeError('Citation dataset name {} wrong'.format(self.name))

        adjacency, train_data, val_data, test_data, adjs = self.preprocessing(data)

        with open(os.path.join(self.processed_folder, self.train_file), 'wb') as f:
            torch.save((adjacency, train_data, [{'indices':it.indices(), 'values':it.values(), 'size':it.size()} for it in adjs['train']]), f)
        with open(os.path.join(self.processed_folder, self.val_file), 'wb') as f:
            torch.save((adjacency, val_data, [{'indices':it.indices(), 'values':it.values(), 'size':it.size()} for it in adjs['val']]), f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save((adjacency, test_data, [{'indices':it.indices(), 'values':it.values(), 'size':it.size()} for it in adjs['test']]), f)
        print('Done.')


    def preprocessing(self, data):

        features = torch.FloatTensor(data.features)

        data.graph.remove_edges_from(data.graph.selfloop_edges())
        # data.graph.add_edges_from(zip(data.graph.nodes(), data.graph.nodes()))
        
        fadj, fadjs = self.feature_adjacency(DGLGraph(data.graph), features)

        labels = torch.LongTensor(data.labels)
        features = features.unsqueeze(1) # for multi-channel features compatibility, (batch, channel, feature)
        train_mask, val_mask, test_mask = torch.ByteTensor([data.train_mask, data.val_mask, data.test_mask])
        train_data, val_data, test_data = features[train_mask,:,:], features[val_mask,:,:], features[test_mask,:,:]
        train_label, val_label, test_label = labels[train_mask], labels[val_mask], labels[test_mask]
        train_fadj, val_fadj, test_fadj = list([compress(fadjs,train_mask), compress(fadjs,val_mask), compress(fadjs, test_mask)])

        return fadj, (train_data, train_label), (val_data, val_label), (test_data, test_label), {'train':train_fadj, 'val':val_fadj, 'test':test_fadj}


    def feature_adjacency(self, graph, features):
        n_edges = graph.number_of_edges()
        n_nodes, n_features = features.shape
        src, dst = graph.edges()

        fadj = torch.zeros(n_features, n_features) # feature adjacency
        for i in tqdm.tqdm(range(n_edges)):
            x = (features[src[i]].view(-1, 1) * features[dst[i]].view(1, -1))/graph.in_degree(src[i])
            fadj += (x+x.transpose(1,0))/graph.in_degree(dst[i])
        fadj = self.row_normalize(fadj.sqrt()) + torch.eye(n_features)

        fadjs = []
        for i in tqdm.tqdm(range(n_nodes)):
            x = (features[i] * features[dst[src==i]].unsqueeze(-1)).sum(0)
            x += x.transpose(1,0)
            x = self.row_normalize(x.sqrt()) + torch.eye(n_features)
            fadjs.append(x.to_sparse()) # better to save instead of appending for larger dataset

        return fadj, fadjs


    def row_normalize(self, x):
        x = x / x.sum(1, keepdim=True)
        x[torch.isinf(x)] = 0
        x[torch.isnan(x)] = 0
        return x

    def __len__(self):
        return len(self.labels)

    def adjacency_matrix(self):
        return self.adjacency_matrix

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.name)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.train_file)) 
            and os.path.exists(os.path.join(self.processed_folder, self.val_file)) 
            and os.path.exists(os.path.join(self.processed_folder, self.test_file)))


if __name__ == "__main__":
    '''
    Use example for Citation Dataset 
    '''
    import torch.utils.data as Data

    train_data = Citation(root='/data/datasets', name='cora', data_type='train', download=True)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=20, shuffle=True, num_workers=0)

    for batch_idx, (inputs, targets, adj) in enumerate(train_loader):
        if torch.cuda.is_available():
            inputs, targets, adj = inputs.cuda(), targets.cuda(), adj.cuda()
        print(batch_idx, inputs.shape, targets.shape, adj.shape)
