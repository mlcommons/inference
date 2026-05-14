"""
Utility functions for MLPerf backends.
"""

import os
import random
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from transformers import set_seed


def get_cache_directory() -> Path:
    """
    Get the cache directory at /raid/data/$USER/.cache

    Returns:
        Path: The cache directory path
    """
    # Get the current user
    user = os.environ.get('USER', os.environ.get('USERNAME', 'unknown'))

    # Use /raid/data/$USER/.cache
    cache_dir = Path(f'/raid/data/{user}/.cache')

    # Create the cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir


def setup_huggingface_cache() -> Path:
    """
    Set up HuggingFace cache environment variables using the preferred cache directory.

    Returns:
        Path: The cache directory being used
    """
    cache_dir = get_cache_directory()

    # Set HuggingFace cache environment variables
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['HF_HUB_CACHE'] = str(cache_dir)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)

    return cache_dir


def find_free_port(start_port: int = 30000, max_attempts: int = 100) -> int:
    """
    Find a free port starting from start_port.

    Args:
        start_port: The port number to start searching from
        max_attempts: Maximum number of ports to try

    Returns:
        int: A free port number

    Raises:
        RuntimeError: If no free port is found after max_attempts
    """
    for i in range(max_attempts):
        port = start_port + i
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            continue
    raise RuntimeError(
        f"Could not find free port after {max_attempts} attempts starting from {start_port}")


def set_all_seeds(seed: int = 42) -> None:
    """
    Set seeds for all random number generators for reproducibility.

    Args:
        seed: The seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    set_seed(seed)


def validate_prompts(tokenized_prompts: Optional[list] = None,
                     text_prompts: Optional[list] = None,
                     backend_type: str = "") -> None:
    """
    Validate that at least one type of prompts is provided.

    Args:
        tokenized_prompts: List of tokenized prompts
        text_prompts: List of text prompts
        backend_type: Name of the backend for error messages

    Raises:
        ValueError: If neither prompt type is provided
    """
    if tokenized_prompts is None and text_prompts is None:
        raise ValueError(
            f"{backend_type + ' backend' if backend_type else 'Backend'} requires either text_prompts or tokenized_prompts")


# Terminal display utilities
class TerminalDisplay:
    """ANSI escape codes and utilities for terminal display formatting."""

    # ANSI escape codes for cursor control
    CLEAR_SCREEN = "\033[2J"
    MOVE_CURSOR_UP = "\033[{}A"
    CLEAR_LINE = "\033[K"
    SAVE_CURSOR = "\033[s"
    RESTORE_CURSOR = "\033[u"

    # Progress spinner characters
    PROGRESS_CHARS = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    @staticmethod
    def clear_lines(num_lines: int) -> None:
        """Clear the specified number of lines above the cursor."""
        print(TerminalDisplay.MOVE_CURSOR_UP.format(
            num_lines), end='', flush=True)
        for _ in range(num_lines):
            print(TerminalDisplay.CLEAR_LINE)
        print(TerminalDisplay.MOVE_CURSOR_UP.format(
            num_lines), end='', flush=True)

    @staticmethod
    def save_cursor_position() -> None:
        """Save the current cursor position."""
        print(TerminalDisplay.SAVE_CURSOR, end='', flush=True)

    @staticmethod
    def restore_cursor_position() -> None:
        """Restore the previously saved cursor position."""
        print(TerminalDisplay.RESTORE_CURSOR, end='', flush=True)

    @staticmethod
    def clear_current_line() -> None:
        """Clear the current line."""
        print("\r" + " " * 80 + "\r", end='', flush=True)

    @staticmethod
    def truncate_line(line: str, max_length: int = 110) -> str:
        """Truncate a line to fit within the specified length."""
        if len(line) <= max_length:
            return line
        return line[:max_length - 3] + "..."


class LogMonitor:
    """Real-time log file monitor with terminal display."""

    def __init__(self,
                 log_file_path: Union[str, Path],
                 prefix: str = "LOG",
                 max_lines: int = 5,
                 display_interval: float = 1.0,
                 header_text: Optional[str] = None):
        """
        Initialize the log monitor.

        Args:
            log_file_path: Path to the log file to monitor
            prefix: Prefix for display lines (e.g., "[SGLANG]")
            max_lines: Maximum number of log lines to display
            display_interval: How often to refresh the display (seconds)
            header_text: Optional custom header text
        """
        self.log_file_path = Path(log_file_path)
        self.prefix = prefix
        self.max_lines = max_lines
        self.display_interval = display_interval
        self.header_text = header_text or f"Server startup logs (last {max_lines} lines):"

        # Threading control
        self._monitor_thread = None
        self._stop_event = None
        self._ready_event = None

        # Display dimensions
        self.total_lines = max_lines + 3  # 2 header lines + 1 blank separator

    def start(self, wait_for_file: bool = True,
              file_wait_timeout: float = 30.0) -> bool:
        """
        Start the log monitor in a background thread.

        Args:
            wait_for_file: Whether to wait for the log file to exist
            file_wait_timeout: How long to wait for the file (seconds)

        Returns:
            bool: True if monitor started successfully
        """
        if self._monitor_thread is not None:
            return True  # Already running

        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(wait_for_file, file_wait_timeout),
            daemon=True
        )
        self._monitor_thread.start()

        # Wait for the monitor to set up its display area
        return self._ready_event.wait(timeout=2.0)

    def stop(self) -> None:
        """Stop the log monitor and clean up display."""
        if self._stop_event and self._monitor_thread:
            self._stop_event.set()
            self._monitor_thread.join(timeout=2)
            self._monitor_thread = None
            self._stop_event = None
            self._ready_event = None

    def _monitor_loop(self, wait_for_file: bool,
                      file_wait_timeout: float) -> None:
        """Main monitoring loop that runs in a separate thread."""
        # Wait for log file if requested
        if wait_for_file:
            start_time = time.time()
            while not self.log_file_path.exists():
                if time.time() - start_time > file_wait_timeout:
                    print(
                        f"[{self.prefix}] Warning: Log file not found after {file_wait_timeout}s: {self.log_file_path}")
                    self._ready_event.set()
                    return
                time.sleep(0.5)
        elif not self.log_file_path.exists():
            print(
                f"[{self.prefix}] Warning: Log file not found: {self.log_file_path}")
            self._ready_event.set()
            return

        print(f"\n[{self.prefix}] Monitoring logs: {self.log_file_path.name}")
        print(f"[{self.prefix}] " + "=" * 60)

        # Initialize display area
        self._setup_display_area()

        # Signal that we're ready
        self._ready_event.set()

        # Buffer for log lines
        line_buffer = []
        last_display_time = 0

        try:
            # Use tail -f to follow the log file
            process = subprocess.Popen(
                ["tail", "-f", str(self.log_file_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            while not self._stop_event.is_set():
                if process.poll() is not None:
                    break

                # Read available lines without blocking
                line_added = False
                try:
                    import select
                    while select.select([process.stdout], [], [], 0)[0]:
                        line = process.stdout.readline()
                        if line:
                            line_buffer.append(line.rstrip())
                            if len(line_buffer) > self.max_lines:
                                line_buffer.pop(0)
                            line_added = True
                        else:
                            break
                except BaseException:
                    # Fallback for systems without select
                    line = process.stdout.readline()
                    if line:
                        line_buffer.append(line.rstrip())
                        if len(line_buffer) > self.max_lines:
                            line_buffer.pop(0)
                        line_added = True

                # Update display if needed
                current_time = time.time()
                if line_added or (
                        current_time - last_display_time >= self.display_interval):
                    last_display_time = current_time
                    self._update_display(line_buffer)

                time.sleep(0.1)

            # Clean up
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

        except Exception as e:
            print(f"\n[{self.prefix}] Log monitor error: {e}")
        finally:
            self._cleanup_display()

    def _setup_display_area(self) -> None:
        """Reserve and initialize the display area."""
        # Reserve space
        for _ in range(self.total_lines):
            print()

        # Move back up to start of reserved area
        print(TerminalDisplay.MOVE_CURSOR_UP.format(
            self.total_lines), end='', flush=True)

        # Print initial display
        print(f"\r[{self.prefix}] {self.header_text}", end='')
        print(TerminalDisplay.CLEAR_LINE, flush=True)
        print(f"\r[{self.prefix}] " + "-" * 60, end='')
        print(TerminalDisplay.CLEAR_LINE, flush=True)

        # Print empty lines
        for _ in range(self.max_lines):
            print(f"\r[{self.prefix}] ", end='')
            print(TerminalDisplay.CLEAR_LINE, flush=True)

        # Print separator
        print(f"\r", end='')
        print(TerminalDisplay.CLEAR_LINE, flush=True)

    def _update_display(self, line_buffer: list) -> None:
        """Update the display with current log lines."""
        # Save cursor position
        print(TerminalDisplay.SAVE_CURSOR, end='', flush=True)

        # Move to start of reserved area (cursor is on progress line, 1 below
        # our area)
        print(TerminalDisplay.MOVE_CURSOR_UP.format(
            self.total_lines + 1), end='', flush=True)

        # Print header
        print(f"\r[{self.prefix}] {self.header_text}", end='')
        print(TerminalDisplay.CLEAR_LINE, flush=True)
        print(f"\r[{self.prefix}] " + "-" * 60, end='')
        print(TerminalDisplay.CLEAR_LINE, flush=True)

        # Print log lines
        for i in range(self.max_lines):
            if i < len(line_buffer):
                line = TerminalDisplay.truncate_line(line_buffer[i], 110)
                print(f"\r[{self.prefix}] {line}", end='')
            else:
                print(f"\r[{self.prefix}] ", end='')
            print(TerminalDisplay.CLEAR_LINE, flush=True)

        # Print separator
        print(f"\r", end='')
        print(TerminalDisplay.CLEAR_LINE, flush=True)

        # Restore cursor position
        print(TerminalDisplay.RESTORE_CURSOR, end='', flush=True)

    def _cleanup_display(self) -> None:
        """Clean up the display area on exit."""
        print(TerminalDisplay.SAVE_CURSOR, end='', flush=True)
        print(TerminalDisplay.MOVE_CURSOR_UP.format(
            self.total_lines + 1), end='', flush=True)

        # Clear all reserved lines
        for _ in range(self.total_lines):
            print(f"\r", end='')
            print(TerminalDisplay.CLEAR_LINE, flush=True)

        print(TerminalDisplay.RESTORE_CURSOR, end='', flush=True)
