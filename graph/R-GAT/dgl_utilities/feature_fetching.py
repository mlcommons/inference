import torch
import os
import concurrent.futures
import os.path as osp
import numpy as np
from typing import Literal


def float2half(base_path, dataset_size):
    paper_nodes_num = {
        "tiny": 100000,
        "small": 1000000,
        "medium": 10000000,
        "large": 100000000,
        "full": 269346174,
    }
    author_nodes_num = {
        "tiny": 357041,
        "small": 1926066,
        "medium": 15544654,
        "large": 116959896,
        "full": 277220883,
    }
    # paper node
    paper_feat_path = os.path.join(base_path, "paper", "node_feat.npy")
    paper_fp16_feat_path = os.path.join(
        base_path, "paper", "node_feat_fp16.pt")
    if not os.path.exists(paper_fp16_feat_path):
        if dataset_size in ["large", "full"]:
            num_paper_nodes = paper_nodes_num[dataset_size]
            paper_node_features = torch.from_numpy(
                np.memmap(
                    paper_feat_path,
                    dtype="float32",
                    mode="r",
                    shape=(num_paper_nodes, 1024),
                )
            )
        else:
            paper_node_features = torch.from_numpy(
                np.load(paper_feat_path, mmap_mode="r")
            )
        paper_node_features = paper_node_features.half()
        torch.save(paper_node_features, paper_fp16_feat_path)

    # author node
    author_feat_path = os.path.join(base_path, "author", "node_feat.npy")
    author_fp16_feat_path = os.path.join(
        base_path, "author", "node_feat_fp16.pt")
    if not os.path.exists(author_fp16_feat_path):
        if dataset_size in ["large", "full"]:
            num_author_nodes = author_nodes_num[dataset_size]
            author_node_features = torch.from_numpy(
                np.memmap(
                    author_feat_path,
                    dtype="float32",
                    mode="r",
                    shape=(num_author_nodes, 1024),
                )
            )
        else:
            author_node_features = torch.from_numpy(
                np.load(author_feat_path, mmap_mode="r")
            )
        author_node_features = author_node_features.half()
        torch.save(author_node_features, author_fp16_feat_path)

    # institute node
    institute_feat_path = os.path.join(base_path, "institute", "node_feat.npy")
    institute_fp16_feat_path = os.path.join(
        base_path, "institute", "node_feat_fp16.pt")
    if not os.path.exists(institute_fp16_feat_path):
        institute_node_features = torch.from_numpy(
            np.load(institute_feat_path, mmap_mode="r")
        )
        institute_node_features = institute_node_features.half()
        torch.save(institute_node_features, institute_fp16_feat_path)

    # fos node
    fos_feat_path = os.path.join(base_path, "fos", "node_feat.npy")
    fos_fp16_feat_path = os.path.join(base_path, "fos", "node_feat_fp16.pt")
    if not os.path.exists(fos_fp16_feat_path):
        fos_node_features = torch.from_numpy(
            np.load(fos_feat_path, mmap_mode="r"))
        fos_node_features = fos_node_features.half()
        torch.save(fos_node_features, fos_fp16_feat_path)

    # conference node
    conference_feat_path = os.path.join(
        base_path, "conference", "node_feat.npy")
    conference_fp16_feat_path = os.path.join(
        base_path, "conference", "node_feat_fp16.pt"
    )
    if not os.path.exists(conference_fp16_feat_path):
        conference_node_features = torch.from_numpy(
            np.load(conference_feat_path, mmap_mode="r")
        )
        conference_node_features = conference_node_features.half()
        torch.save(conference_node_features, conference_fp16_feat_path)

    # journal node
    journal_feat_path = os.path.join(base_path, "journal", "node_feat.npy")
    journal_fp16_feat_path = os.path.join(
        base_path, "journal", "node_feat_fp16.pt")
    if not os.path.exists(journal_fp16_feat_path):
        journal_node_features = torch.from_numpy(
            np.load(journal_feat_path, mmap_mode="r")
        )
        journal_node_features = journal_node_features.half()
        torch.save(journal_node_features, journal_fp16_feat_path)


class IGBH:
    def __init__(
        self,
        data_path,
        name="igbh",
        dataset_size="full",
        use_label_2K=True,
        in_memory=False,
        layout: Literal["CSC", "CSR", "COO"] = "COO",
        type: Literal["fp16", "fp32"] = "fp16",
        device="cpu",
        edge_dir="in",
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.name = name
        self.size = dataset_size
        self.igbh_dataset = IGBHeteroGraphStructure(
            data_path,
            dataset_size=dataset_size,
            in_memory=in_memory,
            use_label_2K=use_label_2K,
            layout=layout,
            use_fp16=(type == "fp16")
        )
        self.num_samples = len(self.igbh_dataset.val_idx)

    def get_samples(self, id_list):
        return self.igbh_dataset.val_idx[id_list]

    def get_labels(self, id_list):
        return self.igbh_dataset.label[self.get_samples(id_list)]

    def get_item_count(self):
        return len(self.igbh_dataset.val_idx)

    def load_query_samples(self, id):
        pass

    def unload_query_samples(self, sample_list):
        pass


class IGBHeteroGraphStructure:
    """
    Synchronously (optionally parallelly) loads the edge relations for IGBH.
    Current IGBH edge relations are not yet converted to torch tensor.
    """

    def __init__(
        self,
        data_path,
        dataset_size="full",
        use_label_2K=True,
        in_memory=False,
        use_fp16=True,
        # in-memory and memory-related optimizations
        separate_sampling_aggregation=False,
        # perf related
        multithreading=True,
        **kwargs,
    ):

        self.dir = data_path
        self.dataset_size = dataset_size
        self.use_fp16 = use_fp16
        self.in_memory = in_memory
        self.use_label_2K = use_label_2K
        self.num_classes = 2983 if not self.use_label_2K else 19
        self.label_file = "node_label_19.npy" if not self.use_label_2K else "node_label_2K.npy"

        self.num_nodes = {
            "full": {'paper': 269346174, 'author': 277220883, 'institute': 26918, 'fos': 712960, 'journal': 49052, 'conference': 4547},
            "small": {'paper': 1000000, 'author': 1926066, 'institute': 14751, 'fos': 190449, 'journal': 15277, 'conference': 1215},
            "medium": {'paper': 10000000, 'author': 15544654, 'institute': 23256, 'fos': 415054, 'journal': 37565, 'conference': 4189},
            "large": {'paper': 100000000, 'author': 116959896, 'institute': 26524, 'fos': 649707, 'journal': 48820, 'conference': 4490},
            "tiny": {'paper': 100000, 'author': 357041, 'institute': 8738, 'fos': 84220, 'journal': 8101, 'conference': 398}
        }[self.dataset_size]

        self.use_journal_conference = True
        self.separate_sampling_aggregation = separate_sampling_aggregation

        self.torch_tensor_input_dir = data_path
        self.torch_tensor_input = self.torch_tensor_input_dir != ""

        self.multithreading = multithreading

        # This class only stores the edge data, labels, and the train/val
        # indices
        self.edge_dict = self.load_edge_dict()
        self.label = self.load_labels()
        self.full_num_trainable_nodes = (
            227130858 if self.num_classes != 2983 else 157675969)
        self.train_idx, self.val_idx = self.get_train_val_test_indices()
        if self.use_fp16:
            float2half(
                os.path.join(
                    self.dir,
                    self.dataset_size,
                    "processed"),
                self.dataset_size)

    def load_edge_dict(self):
        mmap_mode = None if self.in_memory else "r"

        edges = [
            "paper__cites__paper",
            "paper__written_by__author",
            "author__affiliated_to__institute",
            "paper__topic__fos"]
        if self.use_journal_conference:
            edges += ["paper__published__journal", "paper__venue__conference"]

        loaded_edges = None

        def load_edge(edge, mmap=mmap_mode, parent_path=osp.join(
                self.dir, self.dataset_size, "processed")):
            return edge, torch.from_numpy(
                np.load(osp.join(parent_path, edge, "edge_index.npy"), mmap_mode=mmap))

        if self.multithreading:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                loaded_edges = executor.map(load_edge, edges)
            loaded_edges = {
                tuple(edge.split("__")): (edge_index[:, 0], edge_index[:, 1]) for edge, edge_index in loaded_edges
            }
        else:
            loaded_edges = {
                tuple(edge.split("__")): (edge_index[:, 0], edge_index[:, 1])
                for edge, edge_index in map(load_edge, edges)
            }

        return self.augment_edges(loaded_edges)

    def load_labels(self):
        if self.dataset_size not in ['full', 'large']:
            return torch.from_numpy(
                np.load(
                    osp.join(
                        self.dir,
                        self.dataset_size,
                        'processed',
                        'paper',
                        self.label_file)
                )
            ).to(torch.long)
        else:
            return torch.from_numpy(
                np.memmap(
                    osp.join(
                        self.dir,
                        self.dataset_size,
                        'processed',
                        'paper',
                        self.label_file
                    ),
                    dtype='float32',
                    mode='r',
                    shape=(
                        (269346174 if self.dataset_size == "full" else 100000000)
                    )
                )
            ).to(torch.long)

    def augment_edges(self, edge_dict):
        # Adds reverse edge connections to the graph
        # add rev_{edge} to every edge except paper-cites-paper
        edge_dict.update(
            {
                (dst, f"rev_{edge}", src): (dst_idx, src_idx)
                for (src, edge, dst), (src_idx, dst_idx) in edge_dict.items()
                if src != dst
            }
        )

        paper_cites_paper = edge_dict[("paper", 'cites', 'paper')]

        self_loop = torch.arange(self.num_nodes['paper'])
        mask = paper_cites_paper[0] != paper_cites_paper[1]

        paper_cites_paper = (
            torch.cat((paper_cites_paper[0][mask], self_loop.clone())),
            torch.cat((paper_cites_paper[1][mask], self_loop.clone()))
        )

        edge_dict[("paper", 'cites', 'paper')] = (
            torch.cat((paper_cites_paper[0], paper_cites_paper[1])),
            torch.cat((paper_cites_paper[1], paper_cites_paper[0]))
        )

        return edge_dict

    def get_train_val_test_indices(self):
        base_dir = osp.join(self.dir, self.dataset_size, "processed")
        assert osp.exists(osp.join(base_dir, "train_idx.pt")) and osp.exists(osp.join(base_dir, "val_idx.pt")), \
            "Train and validation indices not found. Please run GLT's split_seeds.py first."

        return (
            torch.load(
                osp.join(
                    self.dir,
                    self.dataset_size,
                    "processed",
                    "train_idx.pt")),
            torch.load(
                osp.join(
                    self.dir,
                    self.dataset_size,
                    "processed",
                    "val_idx.pt"))
        )


class Features:
    """
    Lazily initializes the features for IGBH.

    Features will be initialized only when *build_features* is called.

    Features will be placed into shared memory when *share_features* is called
    or if the features are built (either mmap-ed or loaded in memory)
    and *torch.multiprocessing.spawn* is called
    """

    def __init__(self, path, dataset_size, in_memory=True, use_fp16=True):
        self.path = path
        self.dataset_size = dataset_size
        self.in_memory = in_memory
        self.use_fp16 = use_fp16
        if self.use_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.feature = {}

    def build_features(self, use_journal_conference=False,
                       multithreading=False):
        node_types = ['paper', 'author', 'institute', 'fos']
        if use_journal_conference or self.dataset_size in ['large', 'full']:
            node_types += ['conference', 'journal']

        if multithreading:
            def load_feature(feature_store, feature_name):
                return feature_store.load(feature_name), feature_name

            with concurrent.futures.ThreadPoolExecutor() as executor:
                loaded_features = executor.map(
                    load_feature, [(self, ntype) for ntype in node_types])
                self.feature = {
                    node_type: feature_value for feature_value, node_type in loaded_features
                }
        else:
            for node_type in node_types:
                self.feature[node_type] = self.load(node_type)

    def share_features(self):
        for node_type in self.feature:
            self.feature[node_type] = self.feature[node_type].share_memory_()

    def load_from_tensor(self, node):
        return torch.load(osp.join(self.path, self.dataset_size,
                          "processed", node, "node_feat_fp16.pt"))

    def load_in_memory_numpy(self, node):
        return torch.from_numpy(np.load(
            osp.join(self.path, self.dataset_size, 'processed', node, 'node_feat.npy')))

    def load_mmap_numpy(self, node):
        """
        Loads a given numpy array through mmap_mode="r"
        """
        return torch.from_numpy(np.load(osp.join(
            self.path, self.dataset_size, "processed", node, "node_feat.npy"), mmap_mode="r"))

    def memmap_mmap_numpy(self, node):
        """
        Loads a given NumPy array through memory-mapping np.memmap.

        This is the same code as the one provided in IGB codebase.
        """
        shape = [None, 1024]
        if self.dataset_size == "full":
            if node == "paper":
                shape[0] = 269346174
            elif node == "author":
                shape[0] = 277220883
        elif self.dataset_size == "large":
            if node == "paper":
                shape[0] = 100000000
            elif node == "author":
                shape[0] = 116959896

        assert shape[0] is not None
        return torch.from_numpy(np.memmap(osp.join(self.path, self.dataset_size,
                                "processed", node, "node_feat.npy"), dtype="float32", mode='r', shape=tuple(shape)))

    def load(self, node):
        if self.in_memory:
            if self.use_fp16:
                return self.load_from_tensor(node)
            else:
                if self.dataset_size in [
                        'large', 'full'] and node in ['paper', 'author']:
                    return self.memmap_mmap_numpy(node)
                else:
                    return self.load_in_memory_numpy(node)
        else:
            if self.dataset_size in [
                    'large', 'full'] and node in ['paper', 'author']:
                return self.memmap_mmap_numpy(node)
            else:
                return self.load_mmap_numpy(node)

    def get_input_features(self, input_dict, device):
        # fetches the batch inputs
        # moving it here so so that future modifications could be easier
        return {
            key: self.feature[key][value.to(torch.device("cpu")), :].to(
                device).to(self.dtype)
            for key, value in input_dict.items()
        }
