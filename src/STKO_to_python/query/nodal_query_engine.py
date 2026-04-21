"""Concrete nodal results query engine.

Phase 2.6 scope: scaffold that mirrors the public ``Nodes.get_nodal_results``
signature, delegates to the existing manager path, and applies the
base engine's LRU cache so repeat fetches of identical selections are
served from memory.

Later phases move the read logic into this class (so ``Nodes`` becomes a
thin ``NodeManager``); for now the engine lives side-by-side with the
manager and does not change public behavior.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Hashable, Optional, Sequence, Union

import numpy as np

from .base_query_engine import BaseResultsQueryEngine

if TYPE_CHECKING:
    from ..results.nodal_results_dataclass import NodalResults

logger = logging.getLogger(__name__)


class NodalResultsQueryEngine(BaseResultsQueryEngine):
    """Query engine for nodal results.

    Parameters are inherited from :class:`BaseResultsQueryEngine`. The
    engine expects ``dataset`` to expose a ``.nodes`` attribute (the
    manager) — it delegates into the manager for the actual HDF5 read.

    Caching
    -------
    Fetches are cached by a key that covers every input that affects
    output shape / values:
      ``(tuple(sorted(results)), tuple(sorted(stages)), tuple(sorted(ids)))``
    The resolved ``ids`` array is used as the id portion so selection-set
    name / id inputs collapse onto the same cache entry as an explicit
    ``node_ids`` that resolves to the same set.
    """

    __slots__ = ()

    def fetch(
        self,
        *,
        results_name: Union[str, Sequence[str], None] = None,
        model_stage: Union[str, Sequence[str], None] = None,
        node_ids: Union[int, Sequence[int], np.ndarray, None] = None,
        selection_set_id: Union[int, Sequence[int], None] = None,
        selection_set_name: Union[str, Sequence[str], None] = None,
    ) -> "NodalResults":
        """Return a ``NodalResults`` view for the given selection.

        Signature mirrors :meth:`Nodes.get_nodal_results` verbatim.
        """
        manager = self._dataset.nodes

        resolved_ids = self._resolver.resolve_nodes(
            names=selection_set_name,
            ids=selection_set_id,
            explicit_ids=node_ids,
        )

        stages = _normalize_sequence(model_stage, default=tuple(self._dataset.model_stages))
        results = _normalize_sequence(results_name, default=None)

        cache_key = self._build_cache_key(
            results=results,
            stages=stages,
            ids=resolved_ids,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug("NodalResultsQueryEngine cache hit for key %r", cache_key)
            return cached  # type: ignore[return-value]

        # Delegate to the manager for the actual read — Phase 2.6 is a
        # side-by-side scaffold; the read itself migrates here later.
        result = manager.get_nodal_results(
            results_name=results_name,
            model_stage=model_stage,
            node_ids=node_ids,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
        )
        self._cache_put(cache_key, result)  # type: ignore[arg-type]
        return result

    # ------------------------------------------------------------------ #
    # Cache key construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_cache_key(
        *,
        results: Optional[tuple[str, ...]],
        stages: tuple[str, ...],
        ids: np.ndarray,
    ) -> Hashable:
        results_key = results if results is not None else ("__all__",)
        return (
            "nodal",
            tuple(sorted(results_key)),
            tuple(sorted(stages)),
            tuple(sorted(int(x) for x in ids)),
        )


def _normalize_sequence(
    value: Union[str, Sequence[str], None],
    default: Optional[tuple[str, ...]],
) -> tuple[str, ...]:
    if value is None:
        return tuple(default) if default is not None else ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)
