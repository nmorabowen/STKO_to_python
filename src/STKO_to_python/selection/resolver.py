"""SelectionSetResolver — centralized node/element selection-set lookup.

One source of truth for the ``{name -> ids, id -> name, id -> members}``
bidirectional maps built from a dataset's selection-set dict (today produced
by :class:`~STKO_to_python.model.cdata.CData`, tomorrow by a ``CDataReader``).

Before this class, three sites (``Nodes``, ``Elements``,
``NodalResultsInfo``) each carried a near-identical implementation of the
same name-resolution logic. Centralizing it here removes that duplication
and gives every future query engine a consistent resolver to depend on.

Thread-safety:
    Read-only after construction. Safe to share across threads / processes
    as long as callers do not mutate the backing ``selection_set`` dict
    while this resolver lives.

Performance:
    O(nSets) at construction to build the name -> ids bucket map. Every
    public method is then O(nQuery) + O(nMembers). Returned arrays are
    freshly allocated ``numpy.int64`` — callers may mutate freely.
"""
from __future__ import annotations

import logging
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


IntLike = Union[int, np.integer]
IdInput = Union[IntLike, Sequence[IntLike], np.ndarray, None]
NameInput = Union[str, Sequence[str], None]


class SelectionSetResolver:
    """Resolve selection-set names/ids into node or element id arrays.

    Parameters
    ----------
    selection_set:
        Mapping of ``set_id -> dict`` as produced by
        :meth:`STKO_to_python.model.cdata.CData._get_selection_set`.
        Each value is expected to hold at least a ``SET_NAME`` key and one
        or both of ``NODES`` / ``ELEMENTS`` (as sets or iterables of int).

    Notes
    -----
    Name matching is case-insensitive and strips surrounding whitespace.
    An empty or missing name disqualifies the set from name-based lookup
    (the id still works).
    """

    __slots__ = ("_by_name", "_by_id", "_node_ids", "_element_ids")

    def __init__(self, selection_set: Mapping[int, Mapping]) -> None:
        by_name: dict[str, list[int]] = {}
        by_id: dict[int, str] = {}
        node_ids: dict[int, np.ndarray] = {}
        element_ids: dict[int, np.ndarray] = {}

        for raw_sid, payload in selection_set.items():
            try:
                sid = int(raw_sid)
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, Mapping):
                continue

            name_raw = payload.get("SET_NAME", payload.get("name", payload.get("Name", "")))
            name = "" if name_raw is None else str(name_raw)
            by_id[sid] = name

            key = name.strip().lower()
            if key:
                by_name.setdefault(key, []).append(sid)

            nodes_raw = payload.get("NODES")
            if nodes_raw is not None:
                arr = _to_int64_array(nodes_raw)
                if arr.size:
                    node_ids[sid] = arr

            elems_raw = payload.get("ELEMENTS")
            if elems_raw is not None:
                arr = _to_int64_array(elems_raw)
                if arr.size:
                    element_ids[sid] = arr

        self._by_name: dict[str, list[int]] = by_name
        self._by_id: dict[int, str] = by_id
        self._node_ids: dict[int, np.ndarray] = node_ids
        self._element_ids: dict[int, np.ndarray] = element_ids

    def __repr__(self) -> str:
        return (
            f"SelectionSetResolver(n_sets={len(self._by_id)}, "
            f"n_node_sets={len(self._node_ids)}, "
            f"n_element_sets={len(self._element_ids)})"
        )

    def __len__(self) -> int:
        return len(self._by_id)

    # ------------------------------------------------------------------ #
    # Public queries
    # ------------------------------------------------------------------ #
    def resolve_nodes(
        self,
        *,
        names: NameInput = None,
        ids: IdInput = None,
        explicit_ids: IdInput = None,
    ) -> np.ndarray:
        """Resolve node IDs from names, set-ids, and/or explicit node IDs.

        Parameters
        ----------
        names:
            One selection-set name or a sequence of names. Case-insensitive.
        ids:
            One selection-set id or a sequence of ids (integers).
        explicit_ids:
            Node IDs to include verbatim (no selection-set lookup).

        Returns
        -------
        numpy.ndarray
            Sorted, deduplicated ``int64`` array of node IDs.

        Raises
        ------
        ValueError
            If no inputs resolve to any IDs, if a name is unknown or
            ambiguous, or if a named set contains no NODES payload.
        """
        return self._resolve("NODES", self._node_ids, names=names, ids=ids, explicit_ids=explicit_ids)

    def resolve_elements(
        self,
        *,
        names: NameInput = None,
        ids: IdInput = None,
        explicit_ids: IdInput = None,
    ) -> np.ndarray:
        """Resolve element IDs from names, set-ids, and/or explicit IDs.

        See :meth:`resolve_nodes` for parameter semantics — same behavior,
        element payload.
        """
        return self._resolve("ELEMENTS", self._element_ids, names=names, ids=ids, explicit_ids=explicit_ids)

    def list_node_sets(self) -> list[str]:
        """Return the (case-preserved) names of all sets that have NODES."""
        return sorted(
            {self._by_id[sid] for sid in self._node_ids if self._by_id.get(sid)},
            key=str.lower,
        )

    def list_element_sets(self) -> list[str]:
        """Return the (case-preserved) names of all sets that have ELEMENTS."""
        return sorted(
            {self._by_id[sid] for sid in self._element_ids if self._by_id.get(sid)},
            key=str.lower,
        )

    # ------------------------------------------------------------------ #
    # Name / id lookups (exposed for callers that only need the map)
    # ------------------------------------------------------------------ #
    def name_for(self, sid: int) -> str:
        """Return the name for a selection-set id, or ``''`` if unknown."""
        return self._by_id.get(int(sid), "")

    def ids_for_name(self, name: str) -> tuple[int, ...]:
        """Return all set IDs that share a (case-insensitive) name."""
        key = str(name).strip().lower()
        if not key:
            return ()
        return tuple(self._by_name.get(key, ()))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _resolve(
        self,
        payload_key: str,
        member_map: Mapping[int, np.ndarray],
        *,
        names: NameInput,
        ids: IdInput,
        explicit_ids: IdInput,
    ) -> np.ndarray:
        gathered: list[np.ndarray] = []

        name_list = _normalize_names(names)
        if name_list:
            for raw in name_list:
                sid = self._lookup_unique_sid(raw)
                members = member_map.get(sid)
                if members is None or members.size == 0:
                    raise ValueError(
                        f"Selection set {sid} ({raw!r}) empty or missing {payload_key}."
                    )
                gathered.append(members)

        id_list = _normalize_ids(ids)
        for sid in id_list:
            members = member_map.get(sid)
            if members is None or members.size == 0:
                raise ValueError(
                    f"Selection set {sid} empty or missing {payload_key}."
                )
            gathered.append(members)

        if explicit_ids is not None:
            gathered.append(_to_int64_array(explicit_ids))

        if not gathered:
            raise ValueError(
                "Provide names, ids, and/or explicit_ids — got none."
            )

        out = np.unique(np.concatenate(gathered))
        if out.size == 0:
            raise ValueError(f"Resolved {payload_key.lower()} set is empty.")
        return out

    def _lookup_unique_sid(self, raw_name: str) -> int:
        key = str(raw_name).strip().lower()
        hits = self._by_name.get(key, [])
        if not hits:
            available = sorted(self._by_name.keys())
            preview = ", ".join(available[:30]) + (" ..." if len(available) > 30 else "")
            raise ValueError(
                f"Selection set name not found: {raw_name!r}. "
                f"Available (normalized) names: {preview}"
            )
        if len(hits) > 1:
            raise ValueError(
                f"Ambiguous selection set name {raw_name!r}: matches IDs {sorted(hits)}. "
                f"Use an id instead."
            )
        return hits[0]


# ---------------------------------------------------------------------- #
# Module-private helpers
# ---------------------------------------------------------------------- #
def _normalize_names(names: NameInput) -> tuple[str, ...]:
    if names is None:
        return ()
    if isinstance(names, str):
        return (names,) if names.strip() else ()
    return tuple(n for n in names if str(n).strip())


def _normalize_ids(ids: IdInput) -> tuple[int, ...]:
    if ids is None:
        return ()
    if isinstance(ids, (int, np.integer)):
        return (int(ids),)
    arr = np.asarray(list(ids) if not isinstance(ids, np.ndarray) else ids).ravel()
    return tuple(int(x) for x in arr)


def _to_int64_array(obj: Union[Iterable[IntLike], np.ndarray, IntLike]) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj.astype(np.int64, copy=False).ravel()
    if isinstance(obj, (int, np.integer)):
        return np.asarray([int(obj)], dtype=np.int64)
    return np.fromiter((int(x) for x in obj), dtype=np.int64)
