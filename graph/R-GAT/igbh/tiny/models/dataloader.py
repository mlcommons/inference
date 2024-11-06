import torch
from torch_geometric.data import InMemoryDataset, Data
from dgl.data import DGLDataset

from utils import IGL260MDataset

#TODO: Make a PyG dataloader for large datasets
class IGL260M_PyG(InMemoryDataset):
    def __init__(self, args):
        super().__init__(root, transform, pre_transform, pre_filter)

    def process(self):
        dataset = IGL260MDataset(root=self.dir, size=args.dataset_size, \
            in_memory=args.in_memory, classes=args.type_classes, synthetic=args.synthetic)
        node_features = torch.from_numpy(dataset.paper_feat)
        node_edges = torch.from_numpy(dataset.paper_edge).T
        node_labels = torch.from_numpy(dataset.paper_label).to(torch.long)
        data = Data(x=node_features, edge_index=node_edges, y=node_labels)

        n_nodes = node_features.shape[0]

        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask


class IGL260M_DGL(DGLDataset):
    def __init__(self, args):
        self.dir = args.path
        super().__init__(name='IGB260M')

    def process(self):
        dataset = IGL260MDataset(root=self.dir, size=args.dataset_size, \
            in_memory=args.in_memory, classes=args.type_classes, synthetic=args.synthetic)
        node_features = torch.from_numpy(dataset.paper_feat)
        node_edges = torch.from_numpy(dataset.paper_edge)
        node_labels = torch.from_numpy(dataset.paper_label).to(torch.long)

        self.graph = dgl.graph((node_edges[:, 0],node_edges[:, 1]), num_nodes=node_features.shape[0])

        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        
        self.graph = dgl.remove_self_loop(self.graph)
        self.graph = dgl.add_self_loop(self.graph)
        
        n_nodes = node_features.shape[0]

        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1