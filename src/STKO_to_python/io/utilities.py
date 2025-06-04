from typing import TYPE_CHECKING
import os
import logging

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet

class Utilities:
    def __init__(self, dataset: 'MPCODataSet'):
        self.dataset = dataset

    def get_dataset_folder_size(self, unit: str = 'GB') -> float:
        """
        Calculate the size of the dataset folder.

        Parameters:
            unit (str): Unit to report size in ('B', 'KB', 'MB', 'GB'). Defaults to 'MB'.

        Returns:
            float: Folder size in specified unit.
        """
        folder = self.dataset.hdf5_directory

        if not os.path.exists(folder):
            logging.warning(f"Dataset directory does not exist: {folder}")
            return 0.0

        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except Exception as e:
                    logging.error(f"Error accessing file size: {filepath}, {e}")

        unit_factors = {
            'B': 1,
            'KB': 1 / 1024,
            'MB': 1 / (1024**2),
            'GB': 1 / (1024**3),
        }

        factor = unit_factors.get(unit.upper(), 1 / (1024**2))  # Default to MB
        size = total_size * factor
        logging.info(f"Dataset folder size: {size:.2f} {unit.upper()}")
        
        return size
    
    def get_dataset_folder_name(self) -> str:
        """
        Returns the name of the dataset folder (e.g., '1C' from 'D:\\STKO Models\\TH\\1C').

        Returns:
            str: Folder name.
        """
        return os.path.basename(os.path.normpath(self.dataset.hdf5_directory))
