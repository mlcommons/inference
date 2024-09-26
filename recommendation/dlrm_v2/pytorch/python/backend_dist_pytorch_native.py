"""
pytoch native backend for dlrm
"""
import os
import torch
import backend
import numpy as np
import threading

from torchrec import EmbeddingBagCollection
from torchrec.models.dlrm import DLRMTrain, DLRM_DCN
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
from torchrec.distributed.types import ShardingEnv
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)


class BackendDistPytorchNative(backend.Backend):
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
        super(BackendDistPytorchNative, self).__init__()
        mp.set_start_method("spawn")
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

        
        # assert ngpus == 8, "Reference implementation only supports ngpus = 8"
        os.environ["RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        if self.use_gpu:
            os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", str(ngpus))
            self.device = "cuda"
            self.dist_backend = "nccl"
        else:
            self.device: torch.device = torch.device("cpu")
            self.dist_backend = "gloo"
        self.world_size = int(os.environ["WORLD_SIZE"])

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-native-dlrm"
    
    def load(self, model_path, inputs=None, outputs=None):
        # debug prints
        # print(model_path, inputs, outputs)
        print(f"Loading model from {model_path}")
        world_size = int(os.environ["WORLD_SIZE"])

        # Set multiprocessing variables
        manager = mp.Manager()
        self.samples_q = [manager.Queue() for _ in range(world_size)]
        self.dataset_cache = manager.dict()
        self.predictions_cache = [manager.dict() for _ in range(world_size)]
        self.main_lock = manager.Event()
        

        # Create processes to load model
        ctx = mp.get_context("spawn")
        processes = []
        for rank in range(world_size):
            p = ctx.Process(
                target=self.distributed_setup,
                args=(
                    rank, world_size, model_path,
                ),
            )
            p.start()
            processes.append(p)
        self.main_lock.wait()

        return self
        
    def distributed_setup(self, rank, world_size, model_path):
        print("Initializing process...")
        if self.use_gpu:
            self.device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(f"cuda:{rank}")
        dist.init_process_group(backend=self.dist_backend, rank=rank, world_size=world_size)
        pg = dist.group.WORLD
        print("Initializing embeddings...")
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
        dist_model = DistributedModelParallel(
            module=model, device=self.device, plan=plan, env=ShardingEnv.from_process_group(pg),
        )
        self.model = dist_model
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

        self.main_lock.set()

        # Main prediction loop
        while(True):
            item = self.samples_q[rank].get()
            # If -1 is received terminate all subprocesses
            if item == -1:
                break
            with torch.no_grad():
                batch_in = self.dataset_cache[item][rank].to(self.device)
                _, (_, out, _) = self.model(batch_in)
                out = torch.sigmoid(out)
                self.predictions_cache[rank][item] = out.detach().cpu()

    def capture_output(self, id):
        out = []
        rank = 0
        while rank < self.world_size:
            e = self.predictions_cache[rank].get(id, None)
            if e is not None:
                out.append(e)
                rank += 1
        out = torch.cat(out)
        out = torch.reshape(out, (-1,))
        for rank in range(self.world_size):
            self.predictions_cache[rank].pop(id)
        self.dataset_cache.pop(id)
        return out


    def predict(self, samples, ids):
        outputs = []
        # If none is received terminate all subprocesses
        if samples is None:
            for rank in range(self.world_size):
                self.samples_q[rank].put(-1)
            return -1
        self.main_lock.wait()
        self.main_lock.clear()
        for id, batch in zip(ids, samples):
            # Enqueue samples into the multiprocessing queue
            self.dataset_cache[id] = batch
            for rank in range(self.world_size):
                self.samples_q[rank].put(id)

            # Wait for output capture it
            out = self.capture_output(id)
            outputs.append(out)
        self.main_lock.set()
        return outputs
