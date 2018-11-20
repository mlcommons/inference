import os
import tempfile
from datetime import datetime

import tensorboardX

from deepspeech.utils.singleton import Singleton


class GlobalState(metaclass=Singleton, check_args=True):
    """Contains all the global state for a single experiment.

    Warning: Global state is an anti-pattern - think before adding attributes!

    Args:
        exp_dir (string, optional): path to directory where all experimental
            data will be stored. If `None` it defaults to the concatenation of
            `tempfile.gettempdir()`, 'experiments', plus the current date and
            time.
        log_frequency (int): Controls how frequent the `log_step` method will
            be `True`.

    Attributes:
        exp_dir: Path to global log directory.
        writer: A `tensorboardX.SummaryWriter` instance set to write it's files
            to `exp_dir`.
        step (int): An integer used to denote the current step during a
            training session.
    """
    def __init__(self, exp_dir=None, log_frequency=1):
        if exp_dir is None:
            parent = os.path.join(tempfile.gettempdir(), 'experiments')
            os.makedirs(parent, exist_ok=True)
            exp_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S_%f')
            exp_dir = os.path.join(parent, exp_time)

        os.makedirs(exp_dir, exist_ok=True)  # ensure it exists!

        self.exp_dir = exp_dir

        self._init_writer()

        self.log_frequency = log_frequency

        self.step = 0

    def _init_writer(self):
        self.writer = tensorboardX.SummaryWriter(log_dir=self.exp_dir)

    def log_step(self):
        """Returns `True` each time `self.step % log_frequency == 0`"""
        return self.step % self.log_frequency == 0

    def state_dict(self):
        state = {'log_frequency': self.log_frequency,
                 'step': self.step}
        return state

    def load_state_dict(self, state_dict):
        for name, value in state_dict.items():
            setattr(self, name, value)
        self._init_writer()
