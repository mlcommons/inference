import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dgl_utilities.pyg_sampler import PyGSampler

DGL_AVAILABLE = True

try:
    import dgl
except ModuleNotFoundError:
    DGL_AVAILABLE = False
    dgl = None


def check_dgl_available():
    assert DGL_AVAILABLE, "DGL Not available in the container"


def build_graph(graph_structure, backend, features=None):
    assert graph_structure.separate_sampling_aggregation or (features is not None), \
        "Either we need a feature to build the graph, or \
            we should specify to separate sampling from aggregation"

    if backend.lower() == "dgl":
        check_dgl_available()

        graph = dgl.heterograph(graph_structure.edge_dict)
        graph.predict = "paper"

        if features is not None:
            for node, node_feature in features.feature.items():
                if graph.num_nodes(ntype=node) < node_feature.shape[0]:
                    graph.add_nodes(
                        node_feature.shape[0] -
                        graph.num_nodes(
                            ntype=node),
                        ntype=node)
                else:
                    assert graph.num_nodes(ntype=node) == node_feature.shape[0], f"\
                    Graph has more {node} nodes ({graph.num_nodes(ntype=node)}) \
                        than feature shape ({node_feature.shape[0]})"

                if not graph_structure.separate_sampling_aggregation:
                    for node, node_feature in features.feature.items():
                        graph.nodes[node].data['feat'] = node_feature
                        setattr(
                            graph,
                            f"num_{node}_nodes",
                            node_feature.shape[0])

        graph = dgl.remove_self_loop(graph, etype="cites")
        graph = dgl.add_self_loop(graph, etype="cites")

        graph.nodes['paper'].data['label'] = graph_structure.label

        return graph
    else:
        assert False, "Unrecognized backend " + backend


def get_sampler(use_pyg_sampler=False):
    if use_pyg_sampler:
        return PyGSampler
    else:
        return dgl.dataloading.MultiLayerNeighborSampler


def get_loader(graph, index, fanouts, backend, use_pyg_sampler=True, **kwargs):
    if backend.lower() == "dgl":
        check_dgl_available()
        fanouts = [int(fanout) for fanout in fanouts.split(",")]
        return dgl.dataloading.DataLoader(
            graph, {"paper": index},
            get_sampler(use_pyg_sampler=use_pyg_sampler)(fanouts),
            **kwargs
        )
    else:
        assert False, "Unrecognized backend " + backend


def glorot(value):
    if isinstance(value, torch.Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)


class GATPatched(dgl.nn.pytorch.GATConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        if hasattr(self, 'fc'):
            glorot(self.fc.weight)
        else:
            glorot(self.fc_src.weight)
            glorot(self.fc_dst.weight)
        glorot(self.attn_l)
        glorot(self.attn_r)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            glorot(self.res_fc.weight)


class RGAT_DGL(nn.Module):
    def __init__(
            self,
            etypes,
            in_feats, h_feats, num_classes,
            num_layers=2, n_heads=4, dropout=0.2,
            with_trim=None):
        super().__init__()
        self.layers = nn.ModuleList()

        # does not support other models since they are not used
        self.layers.append(dgl.nn.pytorch.HeteroGraphConv({
            etype: GATPatched(in_feats, h_feats // n_heads, n_heads)
            for etype in etypes}))

        for _ in range(num_layers - 2):
            self.layers.append(dgl.nn.pytorch.HeteroGraphConv({
                etype: GATPatched(h_feats, h_feats // n_heads, n_heads)
                for etype in etypes}))

        self.layers.append(dgl.nn.pytorch.HeteroGraphConv({
            etype: GATPatched(h_feats, h_feats // n_heads, n_heads)
            for etype in etypes}))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_feats, num_classes)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = dgl.apply_each(
                h, lambda x: x.view(
                    x.shape[0], x.shape[1] * x.shape[2]))
            if l != len(self.layers) - 1:
                h = dgl.apply_each(h, F.leaky_relu)
                h = dgl.apply_each(h, self.dropout)
        return self.linear(h['paper'])

    def extract_graph_structure(self, batch, device):
        # moves all blocks to device
        return [block.to(device) for block in batch[-1]]

    def extract_inputs_and_outputs(self, sampled_subgraph, device, features):
        # input to the batch argument would be a list of blocks
        # the sampled sbgraph is already moved to device in
        # extract_graph_structure

        # in case if the input feature is not stored on the graph,
        # but rather in shared memory: (separate_sampling_aggregation)
        # we use this method to extract them based on the blocks
        if features is None or features.feature == {}:
            batch_inputs = {
                key: value.to(torch.float32)
                for key, value in sampled_subgraph[0].srcdata['feat'].items()
            }
        else:
            batch_inputs = features.get_input_features(
                sampled_subgraph[0].srcdata[dgl.NID],
                device
            )
        batch_labels = sampled_subgraph[-1].dstdata['label']['paper']
        return batch_inputs, batch_labels


class RGAT(torch.nn.Module):
    def __init__(self, backend, device, graph, **model_kwargs):
        super().__init__()
        self.backend = backend.lower()
        if backend.lower() == "dgl":
            check_dgl_available()
            etypes = graph.etypes
            self.model = RGAT_DGL(etypes=etypes, **model_kwargs)
        else:
            assert False, "Unrecognized backend " + backend

        self.device = device
        self.layers = self.model.layers

    def forward(self, batch, device, features):
        # a general method to get the batches and move them to the
        # corresponding device
        batch = self.model.extract_graph_structure(batch, device)

        # a general method to fetch the features given the sampled blocks
        # and move them to corresponding device
        batch_inputs, batch_labels = self.model.extract_inputs_and_outputs(
            sampled_subgraph=batch,
            device=device,
            features=features,
        )
        return self.model.forward(batch, batch_inputs), batch_labels
