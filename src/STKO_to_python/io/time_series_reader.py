"""TimeSeriesReader — extract STEP/TIME pairs from MPCO DATA groups.

Today the logic to harvest ``STEP``/``TIME`` attrs off every step
subgroup lives inlined (and duplicated) in
:meth:`ModelInfoReader._get_time_series_on_nodes_for_stage` and
:meth:`ModelInfoReader._get_time_series_on_elements_for_stage`. Both
methods do the same thing: open the DATA group, walk its step children,
read the ``STEP``/``TIME`` attrs, unwrap single-element arrays, and
build a dict.

This class extracts the shared logic. It is stateless and pure-function
in spirit; ``__slots__ = ()`` makes that explicit.

MPCO format note
----------------
STKO writes ``STEP`` and ``TIME`` as length-1 numpy arrays on each
step-subgroup's attrs. ``np.asarray(x).item()`` unwraps both 1-element
arrays and 0-d arrays; on historical files (or a modern numpy < 1.25)
where attrs are scalars, it is a no-op.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Mapping, Optional

import numpy as np

if TYPE_CHECKING:
    import h5py

logger = logging.getLogger(__name__)


class TimeSeriesReader:
    """Stateless reader for MPCO STEP/TIME attrs.

    No state: the class exists to localize the read logic and make it
    testable without spinning up a full ``MPCODataSet``.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "TimeSeriesReader()"

    def read_step_time_pairs(
        self,
        data_group: "Optional[h5py.Group]",
    ) -> dict[int, float]:
        """Return ``{step_index: time}`` for every step subgroup of ``data_group``.

        Parameters
        ----------
        data_group:
            An opened ``h5py.Group`` whose children are step subgroups
            (e.g. ``<stage>/RESULTS/ON_NODES/<result>/DATA``). Passing
            ``None`` returns an empty dict — MPCO files do not always
            have every result populated.

        Notes
        -----
        Malformed step subgroups (missing ``STEP`` or ``TIME`` attr) are
        silently skipped. The caller is expected to validate that at
        least one of its result groups produced a non-empty result.
        """
        if data_group is None:
            return {}

        out: dict[int, float] = {}
        for step_name in data_group.keys():
            step_group = data_group[step_name]
            step_attr = step_group.attrs.get("STEP")
            time_attr = step_group.attrs.get("TIME")
            if step_attr is None or time_attr is None:
                continue
            out[int(np.asarray(step_attr).item())] = float(
                np.asarray(time_attr).item()
            )
        return out

    def read_step_time_pairs_multi(
        self,
        data_groups: "Mapping[int, Optional[h5py.Group]]",
    ) -> dict[int, float]:
        """Read + union step/time dicts across multiple partition groups.

        Parameters
        ----------
        data_groups:
            ``{partition_index: h5py.Group}`` mapping.

        Returns
        -------
        dict
            Unioned ``{step_index: time}``; later partitions win on
            duplicate STEP keys (they should all agree anyway).
        """
        out: dict[int, float] = {}
        for _, group in data_groups.items():
            out.update(self.read_step_time_pairs(group))
        return out
