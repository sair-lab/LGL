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
    def __init__(self, root='~/.dgl', name='"ogbn-arxiv"', data_type='train', download=True, task_type=0, thres_nodes = 50, k_hop = 1):
        super(ContinuumOGB, self).__init__(root)
        self.name = name
        self.k_hop = k_hop; self.thres_nodes = thres_nodes

        self.download()
        self.features = torch.FloatTensor(self.data['node_feat'])

        ## add self loop
        self_loop = torch.LongTensor(list(range(len(self.labels))))
        self.src, self.dst = torch.LongTensor(self.data["edge_index"])
        self.src = torch.cat((self.src, self_loop), 0)
        self.dst = torch.cat((self.dst, self_loop), 0)

        self.process_check_list()

        if data_type == 'incremental':
            mask = torch.LongTensor(self.idx_split["train"])#TODO what if we want testing with certain task
            if type(task_type)==list:
                self.mask = torch.LongTensor()
                for i in task_type:
                    self.mask =torch.cat([self.mask,mask[self.labels[mask]==i]],0)
            else:
                self.mask = mask[self.labels[mask]==task_type]
        elif data_type == 'incremental_test':
            mask = torch.LongTensor(self.idx_split["valid"])#TODO what if we want testing with certain task
            if type(task_type)==list:
                self.mask = torch.LongTensor()
                for i in task_type:
                    self.mask =torch.cat([self.mask,mask[self.labels[mask]==i]],0)
            else:
                self.mask = mask[self.labels[mask]==task_type]
        elif data_type in ['train','test','valid']:
            self.mask = torch.LongTensor(self.idx_split[data_type])
        else:
            raise RuntimeError('data type {} wrong'.format(data_type))

        print('{} Dataset for {} Loaded.'.format(self.name, data_type))

    def __len__(self):
        return len(self.labels[self.mask])

    def __getitem__(self, index):
        if self.k_hop == None:
            k_hop = 1
        else:
            k_hop = self.k_hop

        neighbors_khop = list()
        ids_khop = [self.mask[index]]
        ## TODO: simplify this process
        for k in range(k_hop):
            ids = torch.LongTensor()
            neighbor = torch.FloatTensor()
            for i in ids_khop:
                ## save the index of neighbors
                ids = torch.cat((ids, torch.LongTensor(self.check_list[i])),0)
                ids = torch.cat((ids,torch.tensor(i).unsqueeze(0)), 0)
                neighbor = torch.cat((neighbor, self.get_neighbor(ids)),0)
            ## TODO random selection in pytorch is tricky
            if ids.shape[0]>self.thres_nodes:
                indices = torch.randperm(ids.shape[0])[:self.thres_nodes]
                ids = ids[indices]
                neighbor = neighbor[indices]
            ids_khop = ids ## temp ids for next level
            neighbors_khop.append(neighbor) ## cat different level neighbor
        if self.k_hop == None:
            neighbors_khop = neighbors_khop[0]
        return self.features[self.mask][index].unsqueeze(-2), self.labels[self.mask][index], neighbors_khop

    def get_neighbor(self, ids):
        return self.features[ids].unsqueeze(-2)

    def process_check_list(self):
        if os.path.isfile(os.path.join(self.root, self.name+"check-list.pt")):
            self.check_list = torch.load(os.path.join(self.root, self.name+"check-list.pt"))
        else:
            self.check_list = [self.dst[self.src==i] for i in range(self.data["node_feat"].shape[0])]
            torch.save(self.check_list, os.path.join(self.root, self.name+"check-list.pt"))
        
    
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
