"""
pytoch native backend for dlrm
"""
import os
import torch 
import backend
import numpy as np

from torchrec import EmbeddingBagCollection
from torchrec.models.dlrm import DLRM, DLRM_DCN
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.datasets.random import RandomRecDataset

# Modules for distributed running
from torch import distributed as dist
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.model_parallel import DistributedModelParallel, get_default_sharders
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)


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

        if self.debug:
            self.device = "cuda:0" if self.use_gpu else "cpu"
            self.device = torch.device(self.device)
        else:
            #assert ngpus == 8, "Reference implementation only supports ngpus = 8"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = str(ngpus)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"
            rank = int(os.environ["RANK"])
            if self.use_gpu:
                self.device: torch.device = torch.device(f"cuda:{rank}")
                self.dist_backend = "nccl"
                torch.cuda.set_device(self.device)
            else:
                self.device: torch.device = torch.device("cpu")
                self.dist_backend = "gloo"

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-native-dlrm"

    def load(self, model_path, inputs=None, outputs=None):
        # debug prints
        # print(model_path, inputs, outputs)

        if self.debug:
            eb_configs = [
                EmbeddingBagConfig(
                    name=f"t_{feature_name}",
                    embedding_dim=self.embedding_dim,
                    num_embeddings=self.num_embeddings_per_feature[feature_idx],
                    feature_names=[feature_name],
                )
                for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
            ]

            dlrm_model = DLRM_DCN(
                embedding_bag_collection=EmbeddingBagCollection(tables=eb_configs, device=self.device),
                dense_in_features=len(DEFAULT_INT_NAMES),
                dense_arch_layer_sizes=self.dense_arch_layer_sizes,
                over_arch_layer_sizes=self.over_arch_layer_sizes,
                dcn_num_layers=self.dcn_num_layers,
                dcn_low_rank_dim=self.dcn_low_rank_dim,
                dense_device=self.device,
            )
            dlrm_model.inter_arch.to(self.device)
            dlrm_model.eval()
            self.model = dlrm_model
            return self
        else:
            dist.init_process_group(backend=self.dist_backend)
            eb_configs = [
                EmbeddingBagConfig(
                    name=f"t_{feature_name}",
                    embedding_dim=self.embedding_dim,
                    num_embeddings=self.num_embeddings_per_feature[feature_idx],
                    feature_names=[feature_name],
                )
                for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
            ]

            dlrm_model = DLRM_DCN(
                embedding_bag_collection=EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta")),
                dense_in_features=len(DEFAULT_INT_NAMES),
                dense_arch_layer_sizes=self.dense_arch_layer_sizes,
                over_arch_layer_sizes=self.over_arch_layer_sizes,
                dcn_num_layers=self.dcn_num_layers,
                dcn_low_rank_dim=self.dcn_low_rank_dim,
                dense_device=self.device,
            )
            planner = EmbeddingShardingPlanner(
                topology=Topology(
                    local_world_size=get_local_size(),
                    world_size=dist.get_world_size(),
                    compute_device=self.device.type,
                ),
                storage_reservation=HeuristicalStorageReservation(percentage=0.05),
            )
            plan = planner.collective_plan(
                dlrm_model, get_default_sharders(), dist.GroupMember.WORLD
            )
            self.model = DistributedModelParallel(
                module=dlrm_model,
                device=self.device,
                plan=plan,
            )
            # TODO: Load model
            return self
            
        """
        if self.use_gpu:
            dlrm = dlrm.to(self.device)  # .cuda()
            if dlrm.ndevices > 1:
                dlrm.emb_l = dlrm.create_emb(self.m_spa, self.ln_emb)

        if self.use_gpu:
            if dlrm.ndevices > 1:
                # NOTE: when targeting inference on multiple GPUs,
                # load the model as is on CPU or GPU, with the move
                # to multiple GPUs to be done in parallel_forward
                ld_model = torch.load(model_path)
            else:
                # NOTE: when targeting inference on single GPU,
                # note that the call to .to(device) has already happened
                ld_model = torch.load(
                    model_path,
                    map_location=torch.device("cuda")
                    # map_location=lambda storage, loc: storage.cuda(0)
                )
        else:
            # when targeting inference on CPU
            ld_model = torch.load(model_path, map_location=torch.device("cpu"))
        """
        

    def predict(self, samples):
        outputs = []
        for batch in samples:
            sparse_features = batch.sparse_features.to(self.device)
            dense_features = batch.dense_features.to(self.device)
            with torch.no_grad():
                out = self.model(dense_features=dense_features, sparse_features=sparse_features)
                out = torch.reshape(out, (-1,))
                outputs.append(out)
        return outputs


if __name__ == "__main__":
    num_enum_embeddings_per_feature=[
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
    ]
    #num_enum_embeddings_per_feature = [40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36]
    backend = BackendPytorchNative(
        num_embeddings_per_feature = num_enum_embeddings_per_feature,
        embedding_dim=128,
        dcn_num_layers=3,
        dcn_low_rank_dim=512,
        dense_arch_layer_sizes=[512, 256, 128],
        over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
        use_gpu=True,
        debug=False
    )
    backend.load("")

    dataset = RandomRecDataset(
        keys=DEFAULT_CAT_NAMES,
        batch_size=2048,
        hash_sizes=num_enum_embeddings_per_feature,
        ids_per_feature=1,
        num_dense=len(DEFAULT_INT_NAMES),
    )
    sample = next(iter(dataset))
    print(sample)
    prediction = backend.predict(sample)
    print(prediction)
    