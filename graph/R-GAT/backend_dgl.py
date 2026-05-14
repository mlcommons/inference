
from typing import Optional, List, Union, Any
from dgl_utilities.feature_fetching import IGBHeteroGraphStructure, Features, IGBH
from dgl_utilities.components import build_graph, get_loader, RGAT
from dgl_utilities.pyg_sampler import PyGSampler
import os
import torch
import logging
import backend
from typing import Literal

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backend-dgl")


class BackendDGL(backend.Backend):
    def __init__(
        self,
        model_type="rgat",
        type: Literal["fp16", "fp32"] = "fp16",
        device: Literal["cpu", "gpu"] = "gpu",
        ckpt_path: str = None,
        igbh: IGBH = None,
        batch_size: int = 1,
        layout: Literal["CSC", "CSR", "COO"] = "COO",
        edge_dir: str = "in",
    ):
        super(BackendDGL, self).__init__()
        # Set device and type
        if device == "gpu":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if type == "fp32":
            self.type = torch.float32
        else:
            self.type = torch.float16
        # Create Node and neighbor loader
        self.fan_out = [5, 10, 15]
        self.igbh_graph_structure = igbh.igbh_dataset
        self.feature_store = Features(
            self.igbh_graph_structure.dir,
            self.igbh_graph_structure.dataset_size,
            self.igbh_graph_structure.in_memory,
            use_fp16=self.igbh_graph_structure.use_fp16,
        )
        self.feature_store.build_features(use_journal_conference=True)
        self.graph = build_graph(
            self.igbh_graph_structure,
            "dgl",
            features=self.feature_store)
        self.neighbor_loader = PyGSampler([5, 10, 15])
        # Load model Architechture
        self.model = RGAT(
            backend="dgl",
            device=device,
            graph=self.graph,
            in_feats=1024,
            h_feats=512,
            num_classes=2983,
            num_layers=len(self.fan_out),
            n_heads=4
        ).to(self.type).to(self.device)
        self.model.eval()
        # Load model checkpoint
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
            # Get batch
            batch = self.neighbor_loader.sample(self.graph, {"paper": inputs})
            batch_preds, batch_labels = self.model(
                batch, self.device, self.feature_store)
        return batch_preds
