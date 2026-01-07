from typing import TYPE_CHECKING
import os

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet


class TimeUtils:
    """
    Utility class to extract wall-clock execution time from STKO monitors.

    Returns
    -------
    float
        Elapsed time in minutes.
        Returns 0.0 if the monitor is missing, malformed, or unreadable.
    """

    def __init__(self, dataset: "MPCODataSet"):
        self.dataset = dataset

    def get_time_STKO(self) -> float:
        filepath = self.dataset.hdf5_directory
        filename = os.path.join(filepath, "STKO_time_monitor.tim")

        # Monitor not present → elapsed time unavailable
        if not os.path.exists(filename):
            return 0.0

        try:
            with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                lines = [l.strip() for l in f if l.strip()]

            # Expect start and end timestamps
            if len(lines) < 2:
                return 0.0

            start = int(lines[0])
            end = int(lines[1])

            # Guard against invalid timestamps
            if end <= start:
                return 0.0

            # Convert seconds → minutes
            return float((end - start) / 60.0)

        except Exception:
            return 0.0
