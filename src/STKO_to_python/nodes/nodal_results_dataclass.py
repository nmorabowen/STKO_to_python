# nodal_results_dataclass.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Mapping, Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class NodalResults:
    """
    Container for generic nodal results.

    Attributes
    ----------
    df
        Results dataframe. For a single stage, index is typically (node_id, step).
        For multi-stage output, index is (stage, node_id, step).
    time
        Reference time table from MPCODataSet (whatever you already use).
    name
        Dataset name (from MPCODataSet.name).

    node_ids
        Optional tuple of unique node IDs involved in this result set
        (usually sorted in some consistent way).
    coords_map
        Optional mapping node_id -> {'x': float, 'y': float, 'z': float}.
    component_names
        Optional tuple of component/column names (e.g. ('UX','UY','UZ') or
        ('1','2','3') or 'ACCELERATION|1', etc.).
    stages
        Optional tuple of stages used in this result (for multi-stage cases).
    """
    df: pd.DataFrame
    time: Any
    name: str

    node_ids: Optional[Tuple[int, ...]] = None
    coords_map: Optional[Dict[int, Dict[str, float]]] = None
    component_names: Optional[Tuple[str, ...]] = None
    stages: Optional[Tuple[str, ...]] = None
