from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..utilities.attribute_dictionary_class import AttrDict

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet
    
@dataclass
class NodalResults:

    df: pd.DataFrame
    metadata: AttrDict = field(default_factory=AttrDict)

    def __repr__(self):
        return f"<NodalResults: shape={self.df.shape}, metadata_keys={list(self.metadata.keys())}>"