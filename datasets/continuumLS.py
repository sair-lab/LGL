import os
import dgl
import tqdm
import torch
import json
import os.path
import numpy as np
import scipy.sparse
from dgl import DGLGraph
from dgl.data import citegrh
from itertools  import compress
from torchvision.datasets import VisionDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ContinuumLS(VisionDataset):
    def __init__(self, root='/data/', name='reddit', data_type='train', task_type = 0, download=None):
        super(ContinuumLS, self).__init__(root)
        self.name = name
        adj_full, adj_train, feats, class_map, role = self.load_data(os.path.join(root,name))

        self.adj_train = adj_train
        self.adj_full = adj_full

        self.features = feats
        self.feat_len = feats.shape[1]
        self.labels = torch.LongTensor(list(class_map.values()))
        self.num_class = int(torch.max(self.labels) - torch.min(self.labels))+1

        if data_type == 'train':
            self.mask = role["tr"]
        elif data_type == 'mini':
            self.mask = role["tr"][:100]
        elif data_type == 'incremental':
            self.mask = role["tr"]
            self.mask = list((np.array(self.labels)[self.mask]==task_type).nonzero()[0])
        elif data_type == 'val':
            self.mask = role["va"]
        elif data_type == 'test':
            self.mask = role["te"]
        else:
            raise RuntimeError('data type {} wrong'.format(data_type))
        
        print('{} Dataset for {} Loaded with featlen {} and size {}.'.format(self.name, data_type, self.feat_len, len( self.mask)))

    def __len__(self):
        return len(self.labels[self.mask])

    def __getitem__(self, index):
        neighbor_mask = np.append(self.adj_full[self.mask[index]].nonzero()[1],index)
        neighbor = self.features[neighbor_mask]
        return torch.FloatTensor(self.features[self.mask[index]]).unsqueeze(-2), self.labels[self.mask[index]], torch.FloatTensor(neighbor).unsqueeze(-2)

    def load_data(self, prefix, normalize=True):
        adj_full = scipy.sparse.load_npz('{}/adj_full.npz'.format(prefix)).astype(np.bool)
        adj_train = scipy.sparse.load_npz('{}/adj_train.npz'.format(prefix)).astype(np.bool)
        role = json.load(open('{}/role.json'.format(prefix)))
        feats = np.load('{}/feats.npy'.format(prefix))
        class_map = json.load(open('{}/class_map.json'.format(prefix)))
        class_map = {int(k):v for k,v in class_map.items()}
        assert len(class_map) == feats.shape[0]
        train_nodes = np.array(list(set(adj_train.nonzero()[0])))
        train_feats = feats[train_nodes]
        scaler = MinMaxScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
        return adj_full, adj_train, feats, class_map, role
