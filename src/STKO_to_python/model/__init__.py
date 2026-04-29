from .model_info_reader import ModelInfoReader
from .cdata_reader import CDataReader

# Back-compat aliases preserved on the package surface (quiet); the
# deep paths ``STKO_to_python.model.model_info.ModelInfo`` and
# ``STKO_to_python.model.cdata.CData`` emit ``DeprecationWarning`` via
# PEP 562 ``__getattr__``.
ModelInfo = ModelInfoReader
CData = CDataReader

__all__ = [
    "ModelInfo",
    "ModelInfoReader",
    "CData",
    "CDataReader",
]
