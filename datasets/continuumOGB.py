import os
import tqdm
import torch
import os.path
import numpy as np
import scipy.sparse as sp
from itertools  import compress
from torchvision.datasets import VisionDataset
from ogb.nodeproppred import NodePropPredDataset


class ContinuumOGB(VisionDataset):
    def __init__(self, root='~/.dgl', name='"ogbn-arxiv"', data_type='train', download=True, task_type=0):
        super(ContinuumOGB, self).__init__(root)
        self.name = name

        self.download()
        self.features = torch.FloatTensor(self.data['node_feat'])

        ## add self loop
        self_loop = torch.LongTensor(list(range(len(self.labels))))
        self.src, self.dst = torch.LongTensor(self.data["edge_index"])
        self.src = torch.cat((self.src, self_loop), 0)
        self.dst = torch.cat((self.dst, self_loop), 0)

        if data_type == 'incremental':
            mask = self.idx_split[data_type]
            self.mask = (torch.logical_and((self.labels==task_type),mask)).type(torch.bool)
        elif data_type in ['train','test','valid']:
            self.mask = torch.LongTensor(self.idx_split[data_type])
        else:
            raise RuntimeError('data type {} wrong'.format(data_type))

        print('{} Dataset for {} Loaded.'.format(self.name, data_type))

    def __len__(self):
        return len(self.labels[self.mask])

    def __getitem__(self, index):
        neighbor = self.features[self.dst[self.src==self.mask[index]]]
        return self.features[self.mask[index]].unsqueeze(-2), self.labels[self.mask[index]], neighbor.unsqueeze(-2)
    
    def download(self):
        """Download data if it doesn't exist in processed_folder already."""
        print('Loading {} Dataset...'.format(self.name))
        os.makedirs(self.root, exist_ok=True)
        os.environ["OGB_DOWNLOAD_DIR"] = self.root
        dataset = NodePropPredDataset(self.name,self.root)
        self.data = dataset.graph # the graph
        self.labels = torch.LongTensor(dataset.labels).squeeze()
        self.feat_len, self.num_class = self.data["node_feat"].shape[1], dataset.num_classes
        self.idx_split = dataset.get_idx_split()
