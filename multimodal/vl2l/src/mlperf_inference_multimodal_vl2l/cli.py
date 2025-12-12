"""The CLI definition for the VL2L benchmark."""

from __future__ import annotations

from typing import Annotated

import mlperf_loadgen as lg
from loguru import logger
from pydantic import FilePath  # noqa: TC002
from pydantic_typer import Typer
from typer import Option

from .deploy import LocalVllmDeployer
from .evaluation import run_evaluation
from .log import setup_loguru_for_benchmark
from .schema import Dataset, Endpoint, Settings, Verbosity, VllmEndpoint
from .task import ShopifyGlobalCatalogue

app = Typer()
benchmark_app = Typer()
app.add_typer(
    benchmark_app,
    name="benchmark",
    help="Main CLI for running the VL2L benchmark.",
)


@app.command()
def evaluate(
    *,
    random_seed: Annotated[
        int,
        Option(help="The seed for the random number generator used by the benchmark."),
    ] = 12345,
    filename: Annotated[
        FilePath,
        Option(
            help="Location of the accuracy file.",
        ),
    ],
    dataset: Dataset,
) -> None:
    """Evaluate the accuracy of the VLM responses."""
    logger.info("Evaluating the accuracy file")
    run_evaluation(random_seed=random_seed, filename=filename, dataset=dataset)


@benchmark_app.command(name="endpoint")
def benchmark_endpoint(
    *,
    settings: Settings,
    dataset: Dataset,
    endpoint: Endpoint,
    random_seed: Annotated[
        int,
        Option(help="The seed for the random number generator used by the benchmark."),
    ] = 12345,
    verbosity: Annotated[
        Verbosity,
        Option(help="The verbosity level of the logger."),
    ] = Verbosity.INFO,
) -> None:
    """Benchmark an already deployed OpenAI API endpoint.

    This is suitable when you have already deployed an OpenAI API endpoint that is
    accessible via a URL (and an API key, if applicable).
    """
    setup_loguru_for_benchmark(settings=settings, verbosity=verbosity)
    _run_benchmark(
        settings=settings,
        dataset=dataset,
        endpoint=endpoint,
        random_seed=random_seed,
    )


def _run_benchmark(
    settings: Settings,
    dataset: Dataset,
    endpoint: Endpoint,
    random_seed: int,
) -> None:
    """Run the VL2L benchmark."""
    logger.info("Running VL2L benchmark with settings: {}", settings)
    logger.info("Running VL2L benchmark with dataset: {}", dataset)
    logger.info(
        "Running VL2L benchmark with OpenAI API endpoint: {}",
        endpoint)
    logger.info("Running VL2L benchmark with random seed: {}", random_seed)
    test_settings, log_settings = settings.to_lgtype()
    task = ShopifyGlobalCatalogue(
        dataset=dataset,
        endpoint=endpoint,
        settings=settings.test,
        random_seed=random_seed,
    )
    sut = task.construct_sut()
    qsl = task.construct_qsl()
    logger.info("Starting the VL2L benchmark with LoadGen...")
    lg.StartTestWithLogSettings(sut, qsl, test_settings, log_settings)
    logger.info("The VL2L benchmark with LoadGen completed.")
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)


@benchmark_app.command(name="vllm")
def benchmark_vllm(
    *,
    settings: Settings,
    dataset: Dataset,
    vllm: VllmEndpoint,
    random_seed: Annotated[
        int,
        Option(help="The seed for the random number generator used by the benchmark."),
    ] = 12345,
    verbosity: Annotated[
        Verbosity,
        Option(help="The verbosity level of the logger."),
    ] = Verbosity.INFO,
) -> None:
    """Deploy the endpoint using vLLM into a healthy state and then benchmark it.

    This is suitable when you have access to the `vllm serve` command in the local
    environment where this benchmarking CLI is running.
    """
    setup_loguru_for_benchmark(settings=settings, verbosity=verbosity)
    with LocalVllmDeployer(endpoint=vllm, settings=settings):
        _run_benchmark(
            settings=settings,
            dataset=dataset,
            endpoint=vllm,
            random_seed=random_seed,
        )
