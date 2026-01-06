"""The CLI definition for the Qwen3-VL (Q3VL) benchmark."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import entry_points
from typing import Annotated

from loguru import logger
from pydantic import FilePath  # noqa: TC002
from pydantic_typer import Typer
from typer import Option

from .benchmark import run_benchmark
from .deploy import LocalVllmDeployer
from .evaluation import run_evaluation
from .log import setup_loguru_for_benchmark
from .schema import Dataset, Endpoint, Settings, Verbosity, VllmEndpoint

app = Typer()
benchmark_app = Typer()
app.add_typer(
    benchmark_app,
    name="benchmark",
    help="Main CLI for running the Qwen3-VL (Q3VL) benchmark.",
)

_PLUGIN_RESULT_APP_AND_NAME = 2


def _load_benchmark_plugins() -> None:
    """Load and register benchmark plugins from third-party packages."""
    # Discover plugins from the entry point group
    discovered_plugins = entry_points(
        group="mlperf_inf_mm_q3vl.benchmark_plugins")

    for entry_point in discovered_plugins:
        try:
            # Load the plugin function
            plugin_func = entry_point.load()

            # Call the plugin function to get the command/typer app
            plugin_result = plugin_func()

            # Register it with the benchmark app
            if (
                isinstance(plugin_result, Sequence)
                and len(plugin_result) == _PLUGIN_RESULT_APP_AND_NAME
            ):
                # Plugin returns (typer_app, name)
                plugin_app, plugin_name = plugin_result
                benchmark_app.add_typer(plugin_app, name=plugin_name)
                logger.debug(
                    "Loaded benchmark plugin: {} from {}",
                    plugin_name,
                    entry_point.name,
                )
            elif callable(plugin_result):
                # Plugin returns just a command function
                benchmark_app.command(name=entry_point.name)(plugin_result)
                logger.debug("Loaded benchmark command: {}", entry_point.name)
            else:
                logger.warning(
                    "Unsupported plugin function return type {} for plugin {}",
                    type(plugin_result),
                    entry_point.name,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to load benchmark plugin {} with error: {}",
                entry_point.name,
                e,
            )


# Load plugins when the module is imported
_load_benchmark_plugins()


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
    run_benchmark(
        settings=settings,
        dataset=dataset,
        endpoint=endpoint,
        random_seed=random_seed,
    )


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
        run_benchmark(
            settings=settings,
            dataset=dataset,
            endpoint=vllm,
            random_seed=random_seed,
        )
