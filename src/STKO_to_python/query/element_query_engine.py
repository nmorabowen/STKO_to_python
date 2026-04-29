"""Concrete element results query engine.

Phase 2.7 scope: scaffold that mirrors the public ``Elements.get_element_results``
signature, delegates to the manager, and applies the base engine's LRU
cache. Read-logic migration comes later.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Hashable, Optional, Sequence, Union

import numpy as np

from .base_query_engine import BaseResultsQueryEngine

if TYPE_CHECKING:
    from ..results.element_results_dataclass import ElementResults

logger = logging.getLogger(__name__)


class ElementResultsQueryEngine(BaseResultsQueryEngine):
    """Query engine for element results.

    Unlike :class:`NodalResultsQueryEngine`, ``results_name`` and
    ``element_type`` are required — element results are stored by class
    tag in HDF5, so the engine needs to know which dataset to open.
    """

    __slots__ = ()

    def fetch(
        self,
        results_name: str,
        element_type: str,
        *,
        element_ids: Union[Sequence[int], np.ndarray, None] = None,
        selection_set_id: Union[int, Sequence[int], None] = None,
        selection_set_name: Union[str, Sequence[str], None] = None,
        model_stage: Union[str, Sequence[str], None] = None,
        verbose: bool = False,
    ) -> "ElementResults":
        """Return an ``ElementResults`` view for the given selection.

        Signature mirrors :meth:`Elements.get_element_results` verbatim.
        ``model_stage`` accepts a single stage name or a sequence of
        stages — the manager concatenates multi-stage fetches with a
        contiguous global step axis.
        """
        manager = self._dataset.elements

        # Only resolve when caller provided a selection — otherwise leave
        # ``resolved_ids`` as None so the cache key reflects "all".
        resolved_ids: Optional[np.ndarray] = None
        if (
            element_ids is not None
            or selection_set_id is not None
            or selection_set_name is not None
        ):
            resolved_ids = self._resolver.resolve_elements(
                names=selection_set_name,
                ids=selection_set_id,
                explicit_ids=element_ids,
            )

        stages = _normalize_stage_arg(
            model_stage, default_first=self._dataset.model_stages
        )

        cache_key = self._build_cache_key(
            results_name=results_name,
            element_type=element_type,
            stages=stages,
            ids=resolved_ids,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug("ElementResultsQueryEngine cache hit for key %r", cache_key)
            return cached  # type: ignore[return-value]

        result = manager._fetch_element_results_uncached(
            results_name=results_name,
            element_type=element_type,
            element_ids=element_ids,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            model_stage=model_stage,
            verbose=verbose,
        )
        self._cache_put(cache_key, result)  # type: ignore[arg-type]
        return result

    @staticmethod
    def _build_cache_key(
        *,
        results_name: str,
        element_type: str,
        stages: tuple[str, ...],
        ids: Optional[np.ndarray],
    ) -> Hashable:
        if ids is None:
            ids_key: Hashable = "__all__"
        else:
            ids_key = tuple(sorted(int(x) for x in ids))
        # Stages are kept in request order (not sorted) — different
        # orderings imply different concatenation order on the step
        # axis, so they must produce different cache entries.
        return ("element", results_name, element_type, stages, ids_key)


def _normalize_stage_arg(
    model_stage: Union[str, Sequence[str], None],
    default_first: Sequence[str],
) -> tuple[str, ...]:
    """Mirror :meth:`ElementManager._normalize_stages` for cache keying.

    Returns an empty tuple only when the dataset has no stages (so the
    underlying read raises consistently).
    """
    if model_stage is None:
        return (default_first[0],) if default_first else ()
    if isinstance(model_stage, str):
        return (model_stage,)
    return tuple(str(s) for s in model_stage)
