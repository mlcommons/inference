"""The CLI definition for the VL2L benchmark."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import Annotated

import mlperf_loadgen as lg
from loguru import logger
from pydantic import FilePath  # noqa: TC002
from pydantic_typer import Typer
from typer import Option

from .evaluation import run_evaluation
from .schema import Dataset, Endpoint, Settings, Verbosity
from .task import ShopifyGlobalCatalogue

app = Typer()


@app.command()
def evaluate(
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
    run_evaluation(filename=filename, dataset=dataset)


@app.command()
def benchmark(
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
    """Main CLI for running the VL2L benchmark."""
    logger.remove()
    logger.add(sys.stdout, level=verbosity.value.upper())
    datetime_str_in_log_filename = (
        datetime.now(tz=UTC).astimezone().strftime("%FT%TZ_")
        if settings.logging.log_output.prefix_with_datetime
        else ""
    )
    logger.add(
        settings.logging.log_output.outdir
        / (
            f"{settings.logging.log_output.prefix}"
            f"{datetime_str_in_log_filename}"
            "benchmark"
            f"{settings.logging.log_output.suffix}"
            ".txt"
        ),
        level=verbosity.value.upper(),
    )
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
