"""
pytoch native backend for dlrm
"""
import os
import torch
import backend
import numpy as np

from torchrec import EmbeddingBagCollection
from torchrec.models.dlrm import DLRM, DLRM_DCN, DLRMTrain
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.datasets.random import RandomRecDataset

# Modules for distributed running
from torch import distributed as dist
import torch.multiprocessing as mp
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
import torchrec.distributed as trec_dist


class BackendPytorchNative(backend.Backend):
    def __init__(
        self,
        num_embeddings_per_feature,
        embedding_dim=128,
        dcn_num_layers=3,
        dcn_low_rank_dim=512,
        dense_arch_layer_sizes=[512, 256, 128],
        over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
        use_gpu=False,
        debug=False,
    ):
        super(BackendPytorchNative, self).__init__()
        self.i = 0
        self.sess = None
        self.model = None

        self.embedding_dim = embedding_dim
        self.num_embeddings_per_feature = num_embeddings_per_feature
        self.dcn_num_layers = dcn_num_layers
        self.dcn_low_rank_dim = dcn_low_rank_dim
        self.dense_arch_layer_sizes = dense_arch_layer_sizes
        self.over_arch_layer_sizes = over_arch_layer_sizes
        self.debug = debug

        self.use_gpu = use_gpu and torch.cuda.is_available()
        ngpus = torch.cuda.device_count() if self.use_gpu else -1
        if self.use_gpu:
            print("Using {} GPU(s)...".format(ngpus))
        else:
            print("Using CPU...")

        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        if self.use_gpu:
            self.device: torch.device = torch.device(f"cuda:0")
            self.dist_backend = "nccl"
            # torch.cuda.set_device(self.device)
        else:
            #os.environ["WORLD_SIZE"] = "8"
            self.device: torch.device = torch.device("cpu")
            self.dist_backend = "gloo"

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-native-dlrm"

    def load(self, model_path, inputs=None, outputs=None):
        # debug prints
        # print(model_path, inputs, outputs)
        print(f"Loading model from {model_path}")
        
        print("Initializing embeddings...")
        dist.init_process_group(backend=self.dist_backend, rank=0, world_size=1)
        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=self.embedding_dim,
                num_embeddings=self.num_embeddings_per_feature[feature_idx],
                feature_names=[feature_name],
            )
            for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
        ]
        print("Initializing model...")
        dlrm_model = DLRM_DCN(
            embedding_bag_collection=EmbeddingBagCollection(
                tables=eb_configs, device=torch.device("meta")
            ),
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=self.dense_arch_layer_sizes,
            over_arch_layer_sizes=self.over_arch_layer_sizes,
            dcn_num_layers=self.dcn_num_layers,
            dcn_low_rank_dim=self.dcn_low_rank_dim,
            dense_device=self.device,
        )
        model = DLRMTrain(dlrm_model)

        print("Distributing the model...")
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                local_world_size=get_local_size(),
                world_size=dist.get_world_size(),
                compute_device=self.device.type,
            ),
            storage_reservation=HeuristicalStorageReservation(percentage=0.05),
        )
        plan = planner.collective_plan(
            model, get_default_sharders(), dist.GroupMember.WORLD
        )
        self.model = DistributedModelParallel(
            module=model,
            device=self.device,
            plan=plan
        )
        # path_to_sharded_weights should have 2 subdirectories - batched and sharded
        # If we need to load the weights on different device or world size, we would need to change the process
        # group accordingly. If we would want to load on 8 GPUs, the process group created above should be fine
        # to understand sharding, --print_sharding_plan flag should be used while running dlrm_main.py in
        # torcherec implementation
        if not self.debug:
            print("Loading model weights...")
            from torchsnapshot import Snapshot
            snapshot = Snapshot(path=model_path)
            snapshot.restore(app_state={"model": self.model})

            ### To understand the keys in snapshot, you can look at following code snippet.
            # d = snapshot.get_manifest()
            # for k, v in d.items():
            #     print(k, v)
        self.model.eval()
        return self

    def predict(self, samples, ids = None):
        outputs = []
        for batch in samples:
            batch_in = batch.to(self.device)
            with torch.no_grad():
                _, (_, out, _) = self.model(
                    batch_in
                )
                out = torch.sigmoid(out)
                out = torch.reshape(out, (-1,))
                outputs.append(out)
        return outputs

