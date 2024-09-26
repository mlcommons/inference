from typing import Literal
import torch
import os, sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from rgnn import RGNN
from igbh import IGBHeteroDataset, IGBH
import graphlearn_torch as glt

from graphlearn_torch.loader import NodeLoader

from graphlearn_torch.data import Dataset
from graphlearn_torch.sampler import NeighborSampler, NodeSamplerInput
from graphlearn_torch.typing import InputNodes, NumNeighbors

from collections import OrderedDict



class Formatter:
    def __init__(self,
            model_type="rgat",
            type: Literal["fp16", "fp32"] = "fp16",
            device: Literal["cpu", "gpu"] = "gpu",
            ckpt_path: str = None,
            igbh_dataset: IGBHeteroDataset = None,
            batch_size: int = 1,
            layout: Literal["CSC", "CSR", "COO"] = "COO",
            edge_dir: str = "in",
        ) -> None:
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
            formatted_ckpt = OrderedDict()
            formatted_ckpt["model_state_dict"] = OrderedDict()
            for k in ckpt["model_state_dict"]:
                if "lin_dst" in k:
                    pass
                elif "lin_src" in k:
                    formatted_ckpt["model_state_dict"][str(k).replace("lin_src", "lin")] = ckpt["model_state_dict"][k]
                else:
                    formatted_ckpt["model_state_dict"][k] = ckpt["model_state_dict"][k]
            self.model.load_state_dict(formatted_ckpt["model_state_dict"])
            torch.save({'model_state_dict': self.model.state_dict()}, "model/FULL_model_seq_69294_formatted.ckpt")

if __name__ == "__main__":
    igbh = IGBHeteroDataset("igbh", use_label_2K=True)
    f = Formatter(igbh_dataset=igbh, ckpt_path="model/FULL_model_seq_69294.ckpt", device="cpu")
