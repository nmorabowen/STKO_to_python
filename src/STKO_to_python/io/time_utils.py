from typing import TYPE_CHECKING
import logging
import os
from datetime import datetime

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet

class TimeUtils:
    def __init__(self, dataset: 'MPCODataSet'):
        self.dataset = dataset

    def get_time_STKO(self) -> float:
        filepath = self.dataset.hdf5_directory 
        filename = os.path.join(filepath, 'STKO_time_monitor.tim')

        if not os.path.exists(filename):
            logging.warning(f"Time monitor file not found: {filename}")
            return 0.0

        try:
            with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            if len(lines) < 2:
                logging.warning("STKO time file does not contain both start and end timestamps.")
                return 0.0

            start = int(lines[0].strip())
            end = int(lines[1].strip())
            elapsed = end - start

            # Convert the time into minutes
            elapsed = elapsed / 60.0

            logging.info(f"STKO analysis started at {datetime.fromtimestamp(start)}")
            logging.info(f"STKO analysis ended at {datetime.fromtimestamp(end)}")
            logging.info(f"Elapsed time: {elapsed} seconds")

            return float(elapsed)

        except Exception as e:
            logging.error(f"Error reading STKO time monitor file: {e}")
            return 0.0
