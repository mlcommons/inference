"""Core benchmark execution logic for the Qwen3-VL (Q3VL) benchmark."""

from __future__ import annotations

import mlperf_loadgen as lg
from loguru import logger

from .schema import Dataset, Endpoint, Settings
from .task import ShopifyGlobalCatalogue


def run_benchmark(
    settings: Settings,
    dataset: Dataset,
    endpoint: Endpoint,
    random_seed: int,
) -> None:
    """Run the Qwen3-VL (Q3VL) benchmark."""
    logger.info(
        "Running Qwen3-VL (Q3VL) benchmark with settings: {}",
        settings)
    logger.info("Running Qwen3-VL (Q3VL) benchmark with dataset: {}", dataset)
    logger.info(
        "Running Qwen3-VL (Q3VL) benchmark with OpenAI API endpoint: {}",
        endpoint,
    )
    logger.info(
        "Running Qwen3-VL (Q3VL) benchmark with random seed: {}",
        random_seed)
    test_settings, log_settings = settings.to_lgtype()
    task = ShopifyGlobalCatalogue(
        dataset=dataset,
        endpoint=endpoint,
        settings=settings.test,
        random_seed=random_seed,
    )
    sut = task.construct_sut()
    qsl = task.construct_qsl()
    logger.info("Starting the Qwen3-VL (Q3VL) benchmark with LoadGen...")
    lg.StartTestWithLogSettings(sut, qsl, test_settings, log_settings)
    logger.info("The Qwen3-VL (Q3VL) benchmark with LoadGen completed.")
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)
