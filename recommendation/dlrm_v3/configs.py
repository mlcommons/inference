# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
"""
Configuration module for DLRMv3 model.

This module provides configuration functions for the HSTU model architecture and embedding table configurations.
"""
from typing import Dict

from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from generative_recommenders.modules.multitask_module import (
    MultitaskTaskType,
    TaskConfig,
)
from torchrec.modules.embedding_configs import DataType, EmbeddingConfig

HSTU_EMBEDDING_DIM = 512  # final DLRMv3 model
HASH_SIZE = 1_000_000_000


def get_hstu_configs(dataset: str = "debug") -> DlrmHSTUConfig:
    """
    Create and return HSTU model configuration.

    Builds a complete DlrmHSTUConfig with default hyperparameters for the HSTU
    architecture including attention settings, embedding dimensions, dropout rates,
    and feature name mappings.

    Args:
        dataset: Dataset identifier (currently unused, reserved for dataset-specific configs).

    Returns:
        DlrmHSTUConfig: Complete configuration object for the HSTU model.
    """
    hstu_config = DlrmHSTUConfig(
        hstu_num_heads=4,
        hstu_attn_linear_dim=128,
        hstu_attn_qk_dim=128,
        hstu_attn_num_layers=5,
        hstu_embedding_table_dim=HSTU_EMBEDDING_DIM,
        hstu_preprocessor_hidden_dim=256,
        hstu_transducer_embedding_dim=512,
        hstu_group_norm=False,
        hstu_input_dropout_ratio=0.2,
        hstu_linear_dropout_rate=0.1,
        causal_multitask_weights=0.2,
    )
    hstu_config.user_embedding_feature_names = [
        "item_id",
        "user_id",
        "item_category_id",
    ]
    hstu_config.item_embedding_feature_names = [
        "item_candidate_id",
        "item_candidate_category_id",
    ]
    hstu_config.uih_post_id_feature_name = "item_id"
    hstu_config.uih_action_time_feature_name = "action_timestamp"
    hstu_config.candidates_querytime_feature_name = "item_query_time"
    hstu_config.candidates_weight_feature_name = "item_action_weights"
    hstu_config.uih_weight_feature_name = "item_weights"
    hstu_config.candidates_watchtime_feature_name = "item_rating"
    hstu_config.action_weights = [1, 2, 4, 8, 16]
    hstu_config.action_embedding_init_std = 5.0
    hstu_config.contextual_feature_to_max_length = {"user_id": 1}
    hstu_config.contextual_feature_to_min_uih_length = {"user_id": 20}
    hstu_config.merge_uih_candidate_feature_mapping = [
        ("item_id", "item_candidate_id"),
        ("item_rating", "item_candidate_rating"),
        ("action_timestamp", "item_query_time"),
        ("item_weights", "item_action_weights"),
        ("dummy_watch_time", "item_dummy_watchtime"),
        ("item_category_id", "item_candidate_category_id"),
    ]
    hstu_config.hstu_uih_feature_names = [
        "user_id",
        "item_id",
        "item_rating",
        "action_timestamp",
        "item_weights",
        "dummy_watch_time",
        "item_category_id",
    ]
    hstu_config.hstu_candidate_feature_names = [
        "item_candidate_id",
        "item_candidate_rating",
        "item_query_time",
        "item_action_weights",
        "item_dummy_watchtime",
        "item_candidate_category_id",
    ]
    hstu_config.max_num_candidates = 32
    hstu_config.max_num_candidates_inference = 2048
    hstu_config.multitask_configs = [
        TaskConfig(
            task_name="rating",
            task_weight=1,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        )
    ]
    return hstu_config


def get_embedding_table_config(
        dataset: str = "debug") -> Dict[str, EmbeddingConfig]:
    """
    Create and return embedding table configurations.

    Defines the embedding table configurations for item IDs, category IDs, and user IDs
    with their respective dimensions and data types.

    Args:
        dataset: Dataset identifier (currently unused, reserved for dataset-specific configs).

    Returns:
        Dict mapping table names to their EmbeddingConfig objects.
    """
    return {
        "item_id": EmbeddingConfig(
            num_embeddings=HASH_SIZE,
            embedding_dim=HSTU_EMBEDDING_DIM,
            name="item_id",
            data_type=DataType.FP16,
            feature_names=["item_id", "item_candidate_id"],
        ),
        "item_category_id": EmbeddingConfig(
            num_embeddings=128,
            embedding_dim=HSTU_EMBEDDING_DIM,
            name="item_category_id",
            data_type=DataType.FP16,
            weight_init_max=1.0,
            weight_init_min=-1.0,
            feature_names=["item_category_id", "item_candidate_category_id"],
        ),
        "user_id": EmbeddingConfig(
            num_embeddings=10_000_000,
            embedding_dim=HSTU_EMBEDDING_DIM,
            name="user_id",
            data_type=DataType.FP16,
            feature_names=["user_id"],
        ),
    }
