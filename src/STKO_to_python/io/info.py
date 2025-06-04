from .time_utils import TimeUtils
from .utilities import Utilities

from typing import TYPE_CHECKING
import logging
import os
from datetime import datetime

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet


class Info:
    """
    Class to encapsulate general information about the dataset,
    such as runtime, folder size, and folder name.
    """
    def __init__(self, dataset: 'MPCODataSet'):

        self._time_utils = TimeUtils(dataset)
        self._utilities = Utilities(dataset)

        self.analysis_time: float = self._time_utils.get_time_STKO()
        self.size: float = self._utilities.get_dataset_folder_size(unit='GB')
        self.folder_name: str = self._utilities.get_dataset_folder_name()

        if dataset.name is None:
            self.name=self.folder_name
        else:
            self.name: str = dataset.name

    def print_info(self):
        """
        Prints the dataset information in a readable format.
        """
        print('MODEL INFO:')
        print(f'  Dataset Name: {self.name}')
        print(f'  Folder Name: {self.folder_name}')
        print(f"  Runtime: {self.analysis_time:.2f} minutes")
        print(f"  Folder Size: {self.size:.2f} GB")

    def __str__(self):
        return (f"Dataset Info:\n"
                f"  Runtime: {self.analysis_time:.2f} minutes\n"
                f"  Folder Size: {self.size:.2f} GB\n"
                f"  Folder Name: {self.folder_name}\n"
                f"  Dataset Name: {self.name}")

    def __repr__(self):
        return (f"Info(analysis_time={self.analysis_time:.2f}, "
                f"size={self.size:.2f}, folder_name='{self.folder_name}')")