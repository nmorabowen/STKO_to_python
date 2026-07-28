"""Container for section-cut resultant time histories."""

from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CutResultants:
    """Per-pier resultant time histories produced by :class:`SectionCuts`.

    Attributes
    ----------
    df : pd.DataFrame
        Index = step; MultiIndex columns ``(cut, pier, comp)`` with comp in
        {'N', 'V', 'Vn', 'M'} (wall frame: v-hat up, s-hat along the wall).
    time : np.ndarray
        Analysis time per step (same length as ``df``).
    info : pd.DataFrame
        Per-element band geometry (pier, fam, plane, s, w, z) — the audit
        trail of which elements built each pier.
    name : str
        Display name (typically the dataset/run name).
    """

    df: pd.DataFrame
    time: np.ndarray
    info: pd.DataFrame
    name: str = ""

    # ------------------------------------------------------------ access
    @property
    def cuts(self) -> tuple:
        return tuple(self.df.columns.get_level_values("cut").unique())

    @property
    def piers(self) -> tuple:
        return tuple(self.df.columns.get_level_values("pier").unique())

    def series(self, cut: str, pier: str, comp: str) -> pd.Series:
        """One resultant history, e.g. ``series('grade', 'X5750@22000', 'V')``."""
        return self.df[(cut, pier, comp)]

    def envelope(self, comp: str = "V") -> pd.DataFrame:
        """Peak |value| per (cut, pier) for one component."""
        sub = self.df.xs(comp, axis=1, level="comp")
        return sub.abs().max().unstack(level="pier")

    def peaks_table(self) -> pd.DataFrame:
        """Peak |value| for every (cut, pier, comp) — the reduction most EDP
        pipelines start from."""
        return self.df.abs().max().unstack(level="comp")

    # ------------------------------------------------------------ merge
    def join(self, other: "CutResultants") -> "CutResultants":
        """Concatenate cuts from another run of the same model."""
        return CutResultants(
            df=pd.concat([self.df, other.df], axis=1),
            time=self.time,
            info=pd.concat([self.info, other.info], ignore_index=True),
            name=self.name,
        )

    # ------------------------------------------------------------ serialize
    def save_pickle(self, path, compress: bool = True) -> None:
        p = Path(path)
        opener = gzip.open if (compress or str(p).endswith(".gz")) else open
        with opener(p, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_pickle(cls, path) -> "CutResultants":
        p = Path(path)
        opener = gzip.open if str(p).endswith(".gz") else open
        with opener(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a CutResultants object.")
        return obj

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<CutResultants '{self.name}': {len(self.cuts)} cut(s), "
            f"{len(self.piers)} pier(s), {len(self.df)} steps>"
        )
