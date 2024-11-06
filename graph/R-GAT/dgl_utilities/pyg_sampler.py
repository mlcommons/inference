import dgl
import torch


class PyGSampler(dgl.dataloading.Sampler):
    r"""
    An example DGL sampler implementation that matches PyG/GLT sampler behavior.
    The following differences need to be addressed:
    1.  PyG/GLT applies conv_i to edges in layer_i, and all subsequent layers, while DGL only applies conv_i to edges in layer_i.
        For instance, consider a path a->b->c. At layer 0,
        DGL updates only node b's embedding with a->b, but
        PyG/GLT updates both node b and c's embeddings.
        Therefore, if we use h_i(x) to denote the hidden representation of node x at layer i, then the output h_2(c) is:
            DGL:     h_2(c) = conv_2(h_1(c), h_1(b)) = conv_2(h_0(c), conv_1(h_0(b), h_0(a)))
            PyG/GLT: h_2(c) = conv_2(h_1(c), h_1(b)) = conv_2(conv_1(h_0(c), h_0(b)), conv_1(h_0(b), h_0(a)))
    2.  When creating blocks for layer i-1, DGL not only uses the destination nodes from layer i,
        but also includes all subsequent i+1 ... n layers' destination nodes as seed nodes.
    More discussions and examples can be found here: https://github.com/alibaba/graphlearn-for-pytorch/issues/79.
    """

    def __init__(self, fanouts, num_threads=1):
        super().__init__()
        self.fanouts = fanouts
        self.num_threads = num_threads

    def sample(self, g, seed_nodes):
        if self.num_threads != 1:
            old_num_threads = torch.get_num_threads()
            torch.set_num_threads(self.num_threads)
        output_nodes = seed_nodes
        subgs = []
        previous_edges = {}
        previous_seed_nodes = seed_nodes
        input_nodes = seed_nodes

        device = None
        for key in seed_nodes:
            device = seed_nodes[key].device

        not_sampled = {
            ntype: torch.ones([g.num_nodes(ntype)], dtype=torch.bool, device=device) for ntype in g.ntypes
        }

        for fanout in reversed(self.fanouts):
            for node_type in seed_nodes:
                not_sampled[node_type][seed_nodes[node_type]] = 0

            # Sample a fixed number of neighbors of the current seed nodes.
            sg = g.sample_neighbors(seed_nodes, fanout)

            # Before we add the edges, we need to first record the source nodes (of the current seed nodes)
            # so that other edges' source nodes will not be included as next
            # layer's seed nodes.
            temp = dgl.to_block(sg, previous_seed_nodes,
                                include_dst_in_src=False)
            seed_nodes = temp.srcdata[dgl.NID]

            # GLT/PyG does not sample again on previously-sampled nodes
            # we mimic this behavior here
            for node_type in g.ntypes:
                seed_nodes[node_type] = seed_nodes[node_type][not_sampled[node_type]
                                                              [seed_nodes[node_type]]]

            # We add all previously accumulated edges to this subgraph
            for etype in previous_edges:
                sg.add_edges(*previous_edges[etype], etype=etype)

            # This subgraph now contains all its new edges
            # and previously accumulated edges
            # so we add them
            previous_edges = {}
            for etype in sg.etypes:
                previous_edges[etype] = sg.edges(etype=etype)

            # Convert this subgraph to a message flow graph.
            # we need to turn on the include_dst_in_src
            # so that we get compatibility with DGL's OOTB GATConv.
            sg = dgl.to_block(sg, previous_seed_nodes, include_dst_in_src=True)

            # for this layers seed nodes -
            # they will be our next layers' destination nodes
            # so we add them to the collection of previous seed nodes.
            previous_seed_nodes = sg.srcdata[dgl.NID]

            # we insert the block to our list of blocks
            subgs.insert(0, sg)
            input_nodes = seed_nodes
        if self.num_threads != 1:
            torch.set_num_threads(old_num_threads)
        return input_nodes, output_nodes, subgs
