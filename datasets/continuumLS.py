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
    def __init__(self, root='/data/', name='reddit', data_type='train', task_type = 0, download=None, k_hop=1, thres_nodes = 50):
        super(ContinuumLS, self).__init__(root)
        self.name = name
        self.k_hop = k_hop; self.thres_nodes = thres_nodes

        adj_full, adj_train, feats, class_map, role = self.load_data(os.path.join(root,name))

        self.adj_train = adj_train
        self.adj_full = adj_full

        self.features = torch.FloatTensor(feats)
        self.feat_len = feats.shape[1]

        self.labels = torch.LongTensor(list(class_map.values()))
        if name in ["amazon"]:
            self.num_class = self.labels.shape[1]
            _, self.labels = self.labels.max(dim = 1)
        else:
            self.num_class = int(torch.max(self.labels) - torch.min(self.labels))+1
        print("num_class", self.num_class)

        if data_type == 'train':
            self.mask = role["tr"]
        elif data_type == 'mini':
            self.mask = role["tr"][:100]
        elif data_type == 'incremental':
            self.mask = role["tr"]
            self.mask = list((np.array(self.labels)[self.mask]==task_type).nonzero()[0])
        elif data_type == 'valid':
            self.mask = role["va"]
        elif data_type == 'test':
            self.mask = role["te"]
        else:
            raise RuntimeError('data type {} wrong'.format(data_type))
        
        print('{} Dataset for {} Loaded with featlen {} and size {}.'.format(self.name, data_type, self.feat_len, len( self.mask)))

    def __len__(self):
        return len(self.labels[self.mask])

    def __getitem__(self, index):
        if self.k_hop == None:
            k_hop = 1
        else:
            k_hop = self.k_hop

        neighbors_khop = list()
        ids_khop = [self.mask[index]]

        for k in range(k_hop):
            ids = torch.LongTensor()
            neighbor = torch.FloatTensor()
            for i in ids_khop:
                ids = torch.cat((ids, self.get_neighborId(i)),0)
                neighbor = torch.cat((neighbor, self.get_neighbor(i)),0)
            ## TODO random selection in pytorch is tricky
            if ids.shape[0]>self.thres_nodes:
                indices = torch.randperm(ids.shape[0])[:self.thres_nodes]
                ids = ids[indices]
                neighbor = neighbor[indices]
            ids_khop = ids ## temp ids for next level
            neighbors_khop.append(neighbor) ## cat different level neighbor

        if self.k_hop == None:
            neighbors_khop = neighbors_khop[0]
        return torch.FloatTensor(self.features[self.mask[index]]).unsqueeze(-2), self.labels[self.mask[index]], neighbors_khop

    def get_neighbor(self, i):
        return self.features[self.get_neighborId(i)].unsqueeze(-2)

    def get_neighborId(self, i):
        return torch.LongTensor(self.adj_full[i].nonzero()[1])

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
