import os
import dgl
import tqdm
import torch
import os.path
import numpy as np
import scipy.sparse as sp
from dgl import DGLGraph
from dgl.data import citegrh
from itertools  import compress
from torchvision.datasets import VisionDataset
from .continuumLS import ContinuumLS


def graph_collate(batch):
    feature = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    neighbor = [item[2] for item in batch]
    return [feature, labels, neighbor]


def continuum(root='/data/', name='reddit', data_type='train', task_type = 0, download=True):
    name = name.lower()
    if name in ['reddit', 'flickr']:
        return ContinuumLS(root=root, name=name, data_type=data_type, task_type = task_type, download=download)
    elif name in ['cora', 'citeseer', 'pubmed']:
        return Continuum(root=root, name=name, data_type=data_type, task_type = task_type, download=download)
    else:
        raise RuntimeError('data type {} wrong'.format(data_type))


class Continuum(VisionDataset):
    def __init__(self, root='~/.dgl', name='cora', data_type='train', download=True, task_type=0):
        super(Continuum, self).__init__(root)
        self.name = name

        self.download()
        self.features = torch.FloatTensor(self.data.features)
        self.ids = torch.LongTensor(list(range(self.features.size(0))))
        graph = DGLGraph(self.data.graph)
        graph = dgl.transform.add_self_loop(graph)
        self.src, self.dst = graph.edges()
        self.labels = torch.LongTensor(self.data.labels)

        if data_type == 'train': # data incremental; use test and train as train
            self.mask = np.logical_or(self.data.test_mask, self.data.train_mask)
        elif data_type == 'incremental': # class incremental; use test and train as train
            mask = np.logical_or(self.data.test_mask, self.data.train_mask)
            self.mask = (np.logical_and((self.labels==task_type),mask)).type(torch.bool) # low efficient
        elif data_type == 'test':
            self.mask = torch.BoolTensor(self.data.val_mask) # use val as test, since val is larger than test
        else:
            raise RuntimeError('data type {} wrong'.format(data_type))

        print('{} Dataset for {} Loaded.'.format(self.name, data_type))

    def __len__(self):
        return len(self.labels[self.mask])

    def __getitem__(self, index):
        neighbor = self.features[self.dst[self.src==self.ids[self.mask][index]]]
        return self.features[self.mask][index].unsqueeze(-2), self.labels[self.mask][index], neighbor.unsqueeze(-2)
    
    def download(self):
        """Download data if it doesn't exist in processed_folder already."""
        print('Loading {} Dataset...'.format(self.name))
        processed_folder = os.path.join(self.root, self.name)
        os.makedirs(processed_folder, exist_ok=True)
        os.environ["DGL_DOWNLOAD_DIR"] = processed_folder
        data_file = os.path.join(processed_folder, 'data.pt')
        if os.path.exists(data_file):
            self.data = torch.load(data_file)
        else:
            if self.name.lower() == 'cora':
                self.data = citegrh.load_cora()
            elif self.name.lower() == 'citeseer':
                self.data = citegrh.load_citeseer()
            elif self.name.lower() == 'pubmed':
                self.data = citegrh.load_pubmed()
            else:
                raise RuntimeError('Citation dataset name {} wrong'.format(self.name))
            with open(data_file, 'wb') as f:
                torch.save(self.data, data_file)
        self.feat_len, self.num_class = self.data.features.shape[1], self.data.num_labels
