from typing import Optional, List, Union
import os
import torch
import logging
import backend
from typing import Literal
from rgnn import RGNN
from igbh import IGBHeteroDataset, IGBH
import graphlearn_torch as glt

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backend-pytorch")

from graphlearn_torch.loader import NodeLoader

from graphlearn_torch.data import Dataset
from graphlearn_torch.sampler import NeighborSampler, NodeSamplerInput
from graphlearn_torch.typing import InputNodes, NumNeighbors


class CustomNeighborLoader(NodeLoader):
    # Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ==============================================================================
    r"""
    This class is a modified version of the NeighborLoader found in this link:
    https://github.com/alibaba/graphlearn-for-pytorch/blob/main/graphlearn_torch/python/loader/neighbor_loader.py

    A data loader that performs node neighbor sampling for mini-batch training
    of GNNs on large-scale graphs.

    Args:
      data (Dataset): The `graphlearn_torch.data.Dataset` object.
      num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
        number of neighbors to sample for each node in each iteration.
        In heterogeneous graphs, may also take in a dictionary denoting
        the amount of neighbors to sample for each individual edge type.
        If an entry is set to :obj:`-1`, all neighbors will be included.
      input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
        indices of nodes for which neighbors are sampled to create
        mini-batches.
        Needs to be either given as a :obj:`torch.LongTensor` or
        :obj:`torch.BoolTensor`.
        In heterogeneous graphs, needs to be passed as a tuple that holds
        the node type and node indices.
      batch_size (int): How many samples per batch to load (default: ``1``).
      shuffle (bool): Set to ``True`` to have the data reshuffled at every
        epoch (default: ``False``).
      drop_last (bool): Set to ``True`` to drop the last incomplete batch, if
        the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last
        batch will be smaller. (default: ``False``).
      with_edge (bool): Set to ``True`` to sample with edge ids and also include
        them in the sampled results. (default: ``False``).
      strategy: (str): Set sampling strategy for the default neighbor sampler
        provided by graphlearn-torch. (default: ``"random"``).
      as_pyg_v1 (bool): Set to ``True`` to return result as the NeighborSampler
        in PyG v1. (default: ``False``).
    """

    def __init__(
        self,
        data: Dataset,
        num_neighbors: NumNeighbors,
        input_nodes: InputNodes,
        neighbor_sampler: Optional[NeighborSampler] = None,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        with_edge: bool = False,
        with_weight: bool = False,
        strategy: str = "random",
        device: torch.device = torch.device(0),
        seed: Optional[int] = None,
        **kwargs,
    ):
        if neighbor_sampler is None:
            neighbor_sampler = NeighborSampler(
                data.graph,
                num_neighbors=num_neighbors,
                strategy=strategy,
                with_edge=with_edge,
                with_weight=with_weight,
                device=device,
                edge_dir=data.edge_dir,
                seed=seed,
            )
        self.edge_dir = data.edge_dir
        super().__init__(
            data=data,
            node_sampler=neighbor_sampler,
            input_nodes=input_nodes,
            device=device,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            **kwargs,
        )

    def get_neighbors(self, seeds: torch.Tensor):
        inputs = NodeSamplerInput(node=seeds, input_type=self._input_type)
        out = self.sampler.sample_from_nodes(inputs)
        result = self._collate_fn(out)

        return result


class BackendPytorch(backend.Backend):
    def __init__(
        self,
        model_type="rgat",
        type: Literal["fp16", "fp32"] = "fp16",
        device: Literal["cpu", "gpu"] = "gpu",
        ckpt_path: str = None,
        igbh_dataset: IGBHeteroDataset = None,
        batch_size: int = 1,
        layout: Literal["CSC", "CSR", "COO"] = "COO",
        edge_dir: str = "in",
    ):
        super(BackendPytorch, self).__init__()
        self.i = 0
        # Set device and type
        if device == "gpu":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if type == "fp32":
            self.type = torch.float32
        else:
            self.type = torch.float16
        # Create Node and neighbor loade
        self.glt_dataset = glt.data.Dataset(edge_dir=edge_dir)
        self.glt_dataset.init_node_features(
            node_feature_data=igbh_dataset.feat_dict, with_gpu=(device == "gpu"), dtype=self.type
        )
        self.glt_dataset.init_graph(
            edge_index=igbh_dataset.edge_dict,
            layout=layout,
            graph_mode="ZERO_COPY" if (device == "gpu") else "CPU",
        )
        self.glt_dataset.init_node_labels(node_label_data={"paper": igbh_dataset.label})
        self.neighbor_loader = CustomNeighborLoader(
            self.glt_dataset,
            [15, 10, 5],
            input_nodes=("paper", igbh_dataset.val_idx),
            shuffle=False,
            drop_last=False,
            device=self.device,
            seed=42,
        )

        self.model = RGNN(
            self.glt_dataset.get_edge_types(),
            self.glt_dataset.node_features["paper"].shape[1],
            512,
            2983,
            num_layers=3,
            dropout=0.2,
            model=model_type,
            heads=4,
            node_type="paper",
        ).to(self.type).to(self.device)
        self.model.eval()
        ckpt = None
        if ckpt_path is not None:
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
            except FileNotFoundError as e:
                print(f"Checkpoint file not found: {e}")
                return -1
        if ckpt is not None:
            self.model.load_state_dict(ckpt["model_state_dict"])

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-SUT"

    def image_format(self):
        return "NCHW"

    def load(self):
        return self

    def predict(self, inputs: torch.Tensor):
        with torch.no_grad():
            input_size = inputs.shape[0]
            batch = self.neighbor_loader.get_neighbors(inputs)
            out = self.model(
                {
                    node_name: node_feat.to(self.device)
                    for node_name, node_feat in batch.x_dict.items()
                },
                batch.edge_index_dict,
            )[:input_size]
        return out

