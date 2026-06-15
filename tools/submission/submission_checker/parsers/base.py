
from abc import ABC, abstractmethod


class BaseParser:
    def __init__(self, log_path):
        """
        Helper class to parse the detail logs.
        log_path: path to the detail log.
        strict: whether to ignore lines with :::MLLOG prefix but with invalid JSON format.
        """
        self.path = log_path

    @abstractmethod
    def __getitem__(self, key):
        """
        Get the value of the message with the specific key. If a key appears multiple times, the first one is used.
        """
        pass

    @abstractmethod
    def get(self, key):
        """
        Get all the messages with specific key in the log.
        """
        pass

    @abstractmethod
    def get_messages(self):
        """
        Get all the messages in the log.
        """
        pass

    @abstractmethod
    def get_keys(self):
        """
        Get all the keys in the log.
        """
        pass
