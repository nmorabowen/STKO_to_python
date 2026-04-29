from .node_manager import NodeManager

# Back-compat alias preserved on the package surface (quiet); the deep
# path ``STKO_to_python.nodes.nodes.Nodes`` emits a
# ``DeprecationWarning`` via PEP 562 ``__getattr__``.
Nodes = NodeManager

__all__ = [
    'NodeManager',
    'Nodes',
]
