"""Endpoint deployers for deploying and managing the lifecycles of VLM endpoints."""

from __future__ import annotations

import os
import subprocess
import time
from abc import ABC, abstractmethod
from datetime import timedelta  # noqa: TC003
from typing import TYPE_CHECKING, Self
from urllib.parse import urlparse

import requests
from loguru import logger

from .log import get_log_file_path

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from .schema import EndpointToDeploy, Settings, VllmEndpoint


# HTTP status code constants
HTTP_OK = 200


class EndpointStartupTimeoutError(RuntimeError):
    """The exception raised when the endpoint fails to start within the timeout."""

    def __init__(self, timeout: timedelta) -> None:
        """Initialize the exception.

        Args:
            timeout: The timeout duration that was exceeded.
        """
        super().__init__(
            f"Endpoint failed to start within the timeout of {timeout}.",
        )


class EndpointDeployer(ABC):
    """Abstract base class for deploying and managing VLM endpoints.

    Subclasses should implement the deployment and cleanup logic for specific
    inference frameworks (e.g., vLLM, TensorRT-LLM, etc.).

    This class is designed to be used as a context manager:

    ```python
    with EndpointDeployer(...):
        # Endpoint is ready to use
        _run_benchmark(...)
    # Endpoint is shut down
    ```
    """

    def __init__(self, endpoint: EndpointToDeploy, settings: Settings) -> None:
        """Initialize the endpoint deployer.

        Args:
            endpoint: The endpoint configuration.
            settings: The benchmark settings.
        """
        self.endpoint = endpoint
        self.settings = settings

    def __enter__(self) -> Self:
        """Enter the context manager and deploy the endpoint.

        Returns:
            The deployer instance.
        """
        self._startup()
        self._wait_for_ready()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context manager and shut down the endpoint.

        Args:
            exc_type: The exception type if an exception was raised.
            exc_val: The exception value if an exception was raised.
            exc_tb: The exception traceback if an exception was raised.
        """
        logger.info("Shutting down endpoint: {}", self.endpoint)
        self._shutdown()
        logger.info("Endpoint shut down successfully")

    @abstractmethod
    def _startup(self) -> None:
        """Start up the endpoint.

        This method should start the endpoint.
        """
        raise NotImplementedError

    @abstractmethod
    def _failfast(self) -> None:
        """Raise an exception if the endpoint is already detected to be dead."""
        raise NotImplementedError

    def _wait_for_ready(self) -> None:
        """Wait for the endpoint to be ready."""
        health_url = self.endpoint.url.rstrip("/v1") + "/health"
        start_time = time.time()
        while time.time() - start_time < self.endpoint.startup_timeout.total_seconds():
            self._failfast()
            logger.info(
                "Waiting {:0.2f} seconds for endpoint to be ready...",
                time.time() - start_time,
            )
            try:
                response = requests.get(
                    health_url,
                    timeout=self.endpoint.healthcheck_timeout.total_seconds(),
                )
                if response.status_code == HTTP_OK:
                    logger.info("Endpoint is healthy and ready!")
                    return
            except requests.exceptions.RequestException:
                pass

            time.sleep(self.endpoint.poll_interval.total_seconds())

        raise EndpointStartupTimeoutError(self.endpoint.startup_timeout)

    @abstractmethod
    def _shutdown(self) -> None:
        """Shut down the endpoint and clean up resources.

        This method should gracefully terminate the endpoint and clean up
        any resources (processes, files, etc.).
        """
        raise NotImplementedError


class LocalProcessNotStartedError(RuntimeError):
    """The exception raised when the local process is not started yet."""

    def __init__(self) -> None:
        """Initialize the exception."""
        super().__init__("Local process is not started yet.")


class LocalProcessDeadError(RuntimeError):
    """The exception raised when the local process is already detected to be dead."""

    def __init__(
        self,
        returncode: int,
        stdout_file_path: Path,
        stderr_file_path: Path,
    ) -> None:
        """Initialize the exception."""
        super().__init__(
            f"Local process has already terminated with return code {returncode}. "
            f"Please check the logs in {stdout_file_path} and "
            f"{stderr_file_path} for more details.",
        )


class LocalProcessDeployer(EndpointDeployer):
    """Deploy and manage an endpoint that is powered by a local process."""

    def __init__(self, endpoint: EndpointToDeploy, settings: Settings) -> None:
        """Initialize the local process deployer.

        Args:
            endpoint: The endpoint configuration.
            settings: The benchmark settings.
        """
        super().__init__(endpoint=endpoint, settings=settings)
        self._process: subprocess.Popen | None = None
        self._stdout_file_path = get_log_file_path(
            key=self._stdout_log_file_key,
            settings=self.settings,
        )
        self._stderr_file_path = get_log_file_path(
            key=self._stderr_log_file_key,
            settings=self.settings,
        )

    @abstractmethod
    def _build_command(self) -> list[str]:
        """Build the command to start the local process."""
        raise NotImplementedError

    @property
    @abstractmethod
    def _stdout_log_file_key(self) -> str:
        """Get the log file key for the stdout log."""
        raise NotImplementedError

    @property
    @abstractmethod
    def _stderr_log_file_key(self) -> str:
        """Get the log file key for the stderr log."""
        raise NotImplementedError

    def _startup(self) -> None:
        """Start the local process."""
        cmd = self._build_command()
        logger.info("Starting local process with command: {}", cmd)
        logger.info(
            "Starting local process with environment variables: {}",
            os.environ)

        # Start the server
        process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=self._stdout_file_path.open("w"),
            stderr=self._stderr_file_path.open("w"),
            text=True,
        )

        logger.info("Started local process with PID: {}", process.pid)
        logger.info(
            "Local process stdout will be logged to: {}",
            self._stdout_file_path,
        )
        logger.info(
            "Local process stderr will be logged to: {}",
            self._stderr_file_path,
        )

        self._process = process

    def _failfast(self) -> None:
        """Raise an exception if the local process is already detected to be dead."""
        if self._process is None:
            raise LocalProcessNotStartedError
        returncode = self._process.poll()
        if returncode is not None:
            raise LocalProcessDeadError(
                returncode=returncode,
                stdout_file_path=self._stdout_file_path,
                stderr_file_path=self._stderr_file_path,
            )

    def _shutdown(self) -> None:
        """Shut down the local process gracefully."""
        if self._process is None:
            logger.warning("No local process to shut down")
            return

        # Try graceful termination first
        self._process.terminate()
        try:
            self._process.wait(
                timeout=self.endpoint.shutdown_timeout.total_seconds())
            logger.info("Local process terminated gracefully")
        except subprocess.TimeoutExpired:
            logger.warning(
                "Local process did not terminate within timeout, forcefully killing",
            )
            self._process.kill()
            self._process.wait()
            logger.info("Local process killed")


class LocalVllmDeployer(LocalProcessDeployer):
    """Deploy and manage an endpoint that is powered by a local vLLM server."""

    def __init__(self, endpoint: VllmEndpoint, settings: Settings) -> None:
        """Initialize the endpoint deployer.

        Args:
            endpoint: The endpoint configuration.
            settings: The benchmark settings.
        """
        super().__init__(endpoint=endpoint, settings=settings)
        self.endpoint: VllmEndpoint

    @property
    def _stdout_log_file_key(self) -> str:
        """Get the log file key for the stdout log."""
        return "vllm-stdout"

    @property
    def _stderr_log_file_key(self) -> str:
        """Get the log file key for the stderr log."""
        return "vllm-stderr"

    def _build_command(self) -> list[str]:
        """Build the command to start the vLLM server."""
        # Parse the URL to extract host and port
        parsed_url = urlparse(self.endpoint.url)
        host = parsed_url.hostname or "localhost"
        port = parsed_url.port or 8000

        # Build the command
        cmd = [
            "vllm",
            "serve",
            self.endpoint.model.repo_id,
            "--revision",
            self.endpoint.model.revision,
            "--host",
            host,
            "--port",
            str(port),
        ]

        if self.endpoint.model.token:
            cmd.extend(["--hf-token", self.endpoint.model.token])

        # Add API key if provided
        if self.endpoint.api_key:
            cmd.extend(["--api-key", self.endpoint.api_key])

        # Add any additional arguments from the VllmEndpoint.cli
        cmd.extend(self.endpoint.cli)

        return cmd
