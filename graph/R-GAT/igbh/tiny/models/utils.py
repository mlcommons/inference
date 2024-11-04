import numpy as np
import torch

class IGL260MDataset(object):
    def __init__(self, root: str, size: str, in_memory: int, classes: int, synthetic: int):
        self.dir = root
        self.size = size
        self.synthetic = synthetic
        self.in_memory = in_memory
        self.num_classes = classes
        self.__meta__ = torch.load(osp.join(self.dir, self.size, 'meta.pt'))

        self.num_features = self.__meta__['paper']['emb_dim']
        self.num_nodes = self.__meta__['paper']['num_node']
        self.num_edges = self.__meta__['cites']['num_edge']

    @property
    def paper_feat(self) -> np.ndarray:
        if self.synthetic:
            return np.random((self.num_nodes, self.num_edges))
            
        path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_feat.npy')
        if self.in_memory:
            return np.load(path)
        else:
            return np.load(path, mmap_mode='r')

    @property
    def paper_label(self) -> np.ndarray:
        if self.num_classes == 19:
            path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_19.npy')
        else:
            path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_2K.npy')
        if self.in_memory:
            return np.load(path)
        else:
            return np.load(path, mmap_mode='r')

    @property
    def paper_edge(self) -> np.ndarray:
        path = osp.join(self.dir, self.size, 'processed', 'paper__cites__paper', 'edge_index.npy')
        if self.in_memory:
            return np.load(path)
        else:
            return np.load(path, mmap_mode='r')



def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, inputs, labels, val_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid])

def load_subtensor(g, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_labels = g.ndata['labels'][seeds].to(device)
    return batch_inputs, batch_labels

def track_acc(g, args):
    train_accuracy = []
    test_accuracy = []
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    in_feats = g.ndata['features'].shape[1]
    n_classes = args.num_classes

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    g.create_formats_()

    num_epochs = args.epochs
    num_hidden = args.hidden_channels
    num_layers = args.num_layers
    fan_out = args.fan_out
    batch_size = args.batch_size
    lr = args.learning_rate
    dropout = args.dropout
    num_workers = args.num_workers

    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
                [int(fanout) for fanout in fan_out.split(',')])
    
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers)

    # Define model and optimizer
    if args.model_type == 'gcn':
        model = GCN(in_feats, num_hidden, n_classes, 1, F.relu, dropout)
    if args.model_type == 'sage':
        model = SAGE(in_feats, num_hidden, n_classes, num_layers, F.relu, dropout, 'gcn')
    if args.model_type == 'gat':
        model = GAT(in_feats, num_hidden, n_classes, num_layers, 2, F.relu)

    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.decay)

     # Training loop
    avg = 0
    best_test_acc = 0
    log_every = 1
    training_start = time.time()
    for epoch in (range(num_epochs)):
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        epoch_loss = 0
        gpu_mem_alloc = 0
        epoch_start = time.time()
        for step, (input_nodes, seeds, blocks) in (enumerate(dataloader)):
            # Load the input features as well as output labels
            #batch_inputs, batch_labels = load_subtensor(g, seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach()

            gpu_mem_alloc += (
                torch.cuda.max_memory_allocated() / 1000000
                if torch.cuda.is_available()
                else 0
            )

        train_g = g
        train_nid = torch.nonzero(
            train_g.ndata['train_mask'], as_tuple=True)[0]
        train_acc = evaluate(
            model, train_g, train_g.ndata['features'], train_g.ndata['labels'], train_nid, batch_size, device)
        
        test_g = g
        test_nid = torch.nonzero(
            test_g.ndata['test_mask'], as_tuple=True)[0]
        test_acc = evaluate(
            model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, batch_size, device)

        if test_acc.item() > best_test_acc:
            best_test_acc = test_acc.item()
        tqdm.write(
            "Epoch {:05d} | Loss {:.4f} | Train Acc {:.4f} | Test Acc {:.4f} | Time {:.2f}s | GPU {:.1f} MB".format(
                epoch,
                epoch_loss,
                train_acc.item(),
                test_acc.item(),
                time.time() - epoch_start,
                gpu_mem_alloc
            )
        )
        test_accuracy.append(test_acc.item())
        train_accuracy.append(train_acc.item())
        torch.save(model.state_dict(), args.modelpath)
    print()
    print("Total time taken: ", time.time() - training_start)

    return best_test_acc, train_accuracy, test_accuracy
