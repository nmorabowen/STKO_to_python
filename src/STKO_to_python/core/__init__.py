from .dataset import MPCODataSet
from .metadata import ModelMetadata

# Back-compat alias preserved on the package surface (quiet); the deep
# path ``STKO_to_python.core.dataclasses.MetaData`` emits a
# ``DeprecationWarning`` via PEP 562 ``__getattr__``.
MetaData = ModelMetadata

__all__ = [
    "MPCODataSet",
    "MetaData",
    "ModelMetadata",
]
