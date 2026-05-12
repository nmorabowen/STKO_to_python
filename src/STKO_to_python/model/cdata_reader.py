"""Reader for MPCO ``.cdata`` sidecar files.

The ``.cdata`` text file is STKO's companion to ``.mpco`` results. It
carries model metadata that does not live in HDF5:

- ``*SELECTION_SET`` — named groups of nodes / elements (also exposed
  via :class:`SelectionSetResolver`).
- ``*LOCAL_AXES`` — per-element rotation quaternion (element-local
  frame relative to global).
- ``*SECTION_OFFSET`` — per-element section-centroid offset in
  element-local 2D coords.
- ``*ELEMENT_INFO`` — parent geometry / sub-geometry / physical and
  element property metadata per element. Enables "select by geometry"
  workflows without pre-defined selection sets.
- ``*BEAM_PROFILE`` and ``*BEAM_PROFILE_ASSIGNMENT`` — present in the
  file but not yet parsed by this reader.

``CDataReader`` does a single pass over every partition the first time
any accessor is touched. ``MPCODataSet`` constructs the reader eagerly
and queries ``_extract_selection_set_ids`` during ``__init__``, so in
practice the parse happens at dataset-construction time and every
accessor is then O(1).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Optional, Union

import numpy as np


if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet

logger = logging.getLogger(__name__)


SelectionSetIdsArg = Union[int, np.integer, list, None]


# ---------------------------------------------------------------------- #
# Structured records
# ---------------------------------------------------------------------- #

@dataclass(frozen=True)
class ElementInfo:
    """Parent-geometry and property metadata for one element.

    Mirrors a single ``*ELEMENT_INFO`` row. Names can contain spaces
    (e.g. ``"New physical property"``); STKO emits them with explicit
    length prefixes which the parser handles transparently.
    """
    element_id: int
    geom_id: int
    geom_name: str
    sub_geom_idx: int
    sub_geom_type: str           # "Edge" | "Face" | "Solid" | ...
    physical_property_id: int
    physical_property_name: str
    element_property_id: int
    element_property_name: str


# ---------------------------------------------------------------------- #
# Low-level parsing helpers (module-private; testable directly)
# ---------------------------------------------------------------------- #

def _read_length_prefixed(rest: str) -> tuple[str, str]:
    """Consume a ``LENGTH NAME`` pair from the start of *rest*.

    STKO encodes name fields as ``LENGTH NAME`` where ``LENGTH`` is the
    character count of ``NAME`` (which may itself contain spaces, e.g.
    ``"21 New physical property"``).

    Returns ``(name, remainder)`` where ``remainder`` has the
    name's trailing separator space stripped.
    """
    space_idx = rest.find(" ")
    if space_idx == -1:
        raise ValueError(f"Expected '<len> <name>' pair, got: {rest!r}")
    length = int(rest[:space_idx])
    name_start = space_idx + 1
    name_end = name_start + length
    name = rest[name_start:name_end]
    remaining = rest[name_end:]
    if remaining.startswith(" "):
        remaining = remaining[1:]
    return name, remaining


def _read_section_lines(lines: list[str], start: int) -> tuple[list[str], int]:
    """Return ``(data_lines, end_index)`` for the section beginning at *start*.

    Sections terminate at the first blank line, ``*`` marker, or EOF.
    Each data line is right-stripped of whitespace; the leading marker
    line itself is the caller's responsibility (not included in *start*).
    """
    data: list[str] = []
    i = start
    n = len(lines)
    while i < n:
        s = lines[i].rstrip()
        if not s:
            break
        if s.startswith("*"):
            break
        data.append(s)
        i += 1
    return data, i


# ---------------------------------------------------------------------- #
# CDataReader
# ---------------------------------------------------------------------- #

class CDataReader:
    """Canonical Layer 3 reader for MPCO ``.cdata`` sidecar files.

    Parsing happens lazily on first access to any accessor (or eagerly
    when ``MPCODataSet`` calls ``_extract_selection_set_ids`` during
    construction). All sections share one file pass per partition.

    The legacy name ``CData`` is preserved as a quiet alias on the
    package surface; the deep import path emits ``DeprecationWarning``.
    """

    def __init__(self, dataset: "MPCODataSet"):
        self.dataset = dataset
        # Single-pass cache. None until first access; populated by
        # `_parse_all_files`.
        self._parsed: Optional[dict] = None

    # ------------------------------------------------------------------ #
    # Public read-only views (lazy)
    # ------------------------------------------------------------------ #

    @cached_property
    def local_axes(self) -> dict[int, np.ndarray]:
        """``{elem_id -> [qw, qx, qy, qz]}`` per-element rotation quaternion.

        The quaternion rotates from element-local to global coordinates.
        Useful for interpreting beam force/moment results that STKO
        emits in the local frame.
        """
        return self._parse_all_files()["local_axes"]

    @cached_property
    def section_offsets(self) -> dict[int, np.ndarray]:
        """``{elem_id -> [yOffset, zOffset]}`` in element-local coords.

        Distance from the integration axis to the section centroid;
        relevant for moment computations about the geometric centerline.
        """
        return self._parse_all_files()["section_offsets"]

    @cached_property
    def element_info(self) -> dict[int, ElementInfo]:
        """``{elem_id -> ElementInfo}`` parent geometry & property metadata.

        Lets callers select elements by STKO geometry/property names
        rather than relying on pre-built selection sets.
        """
        return self._parse_all_files()["element_info"]

    # ------------------------------------------------------------------ #
    # Selection-set surface (eager from MPCODataSet.__init__)
    # ------------------------------------------------------------------ #

    def _extract_selection_set_ids(
        self,
        selection_set_ids: SelectionSetIdsArg = None,
    ) -> dict[int, dict]:
        """Aggregate ``*SELECTION_SET`` blocks across every partition.

        Args:
            selection_set_ids: Optional ID filter. If ``None``, every
                set from every partition is aggregated.

        Returns:
            A fresh dict mapping ``set_id -> {"SET_NAME": str,
            "NODES": list[int], "ELEMENTS": list[int]}``. Member lists
            are sorted and deduplicated across partitions. Safe to
            mutate without disturbing the reader's cache.
        """
        if isinstance(selection_set_ids, (int, np.integer)):
            selection_set_ids = [int(selection_set_ids)]
        if selection_set_ids is not None and not isinstance(selection_set_ids, list):
            raise ValueError("selection_set_ids must be a list of integers or None.")

        cached = self._parse_all_files()["selection_sets"]
        if selection_set_ids is None:
            wanted = cached.keys()
        else:
            wanted = set(selection_set_ids) & cached.keys()
        return {
            sid: {
                "SET_NAME": cached[sid]["SET_NAME"],
                "NODES": list(cached[sid]["NODES"]),
                "ELEMENTS": list(cached[sid]["ELEMENTS"]),
            }
            for sid in wanted
        }

    def print_selection_set_names(self) -> None:
        """Emit names of all available selection sets at INFO level."""
        sets = self.dataset.selection_set
        logger.info("Available selection sets:")
        for key, payload in sets.items():
            logger.info("  Set id: %s - Set name: %s", key, payload["SET_NAME"])

    # ------------------------------------------------------------------ #
    # Parsing internals
    # ------------------------------------------------------------------ #

    def _parse_all_files(self) -> dict:
        """Single-pass parse across every partition; cached on ``self``."""
        if self._parsed is not None:
            return self._parsed

        accum: dict = {
            "selection_sets": {},      # int -> {"SET_NAME": str, "NODES": set, "ELEMENTS": set}
            "local_axes": {},          # int -> ndarray(4,)
            "section_offsets": {},     # int -> ndarray(2,)
            "element_info": {},        # int -> ElementInfo
        }
        for file_path in self.dataset.cdata_partitions.values():
            self._parse_file_into(file_path, accum)

        # Finalize: convert member sets to sorted lists for the
        # selection-set payload. Done once, post-aggregation.
        for payload in accum["selection_sets"].values():
            payload["NODES"] = sorted(payload["NODES"])
            payload["ELEMENTS"] = sorted(payload["ELEMENTS"])

        self._parsed = accum
        return accum

    def _parse_file_into(self, file_path: str, accum: dict) -> None:
        """Single-pass parse of one ``.cdata`` file into *accum*.

        Re-raises after logging on any failure so the dataset fails
        loudly at construction; silent ``return``s used to mask broken
        cdata files until a downstream selection-set query crashed.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            i = 0
            n = len(lines)
            while i < n:
                marker = lines[i].strip()
                if marker == "*SELECTION_SET":
                    i = self._parse_one_selection_set(lines, i, accum["selection_sets"])
                elif marker == "*LOCAL_AXES":
                    i = self._parse_local_axes(lines, i + 1, accum["local_axes"])
                elif marker == "*SECTION_OFFSET":
                    i = self._parse_section_offset(lines, i + 1, accum["section_offsets"])
                elif marker == "*ELEMENT_INFO":
                    i = self._parse_element_info(lines, i + 1, accum["element_info"])
                else:
                    i += 1
        except Exception:
            logger.exception("CData parse error in %s", file_path)
            raise

    # --- per-section parsers ----------------------------------------- #

    @staticmethod
    def _parse_one_selection_set(lines: list[str], i: int, out: dict) -> int:
        """Parse one ``*SELECTION_SET`` block; return index after the block."""
        set_id = int(lines[i + 1].strip())

        raw_name = lines[i + 2].strip()
        name_length = int(raw_name.split()[0])
        offset = len(str(name_length)) + 1
        set_name = raw_name[offset : offset + name_length]

        n_nodes = int(lines[i + 3].strip())
        n_elems = int(lines[i + 4].strip())

        # Elements come right after the (possibly empty) nodes block.
        # Initialize unconditionally so NNODES=0 + NELEMENTS>0 doesn't
        # reference an unbound name.
        cursor = i + 5

        nodes: np.ndarray = np.empty(0, dtype=int)
        if n_nodes > 0:
            end = cursor + (n_nodes + 9) // 10
            nodes = np.fromstring(" ".join(lines[cursor:end]).strip(), sep=" ", dtype=int)
            cursor = end

        elements: np.ndarray = np.empty(0, dtype=int)
        if n_elems > 0:
            end = cursor + (n_elems + 9) // 10
            elements = np.fromstring(" ".join(lines[cursor:end]).strip(), sep=" ", dtype=int)
            cursor = end

        if set_id in out:
            out[set_id]["NODES"].update(int(x) for x in nodes)
            out[set_id]["ELEMENTS"].update(int(x) for x in elements)
        else:
            out[set_id] = {
                "SET_NAME": set_name,
                "NODES": set(int(x) for x in nodes),
                "ELEMENTS": set(int(x) for x in elements),
            }
        return cursor

    @staticmethod
    def _parse_local_axes(lines: list[str], i: int, out: dict) -> int:
        data, end = _read_section_lines(lines, i)
        for line in data:
            parts = line.split()
            elem_id = int(parts[0])
            out[elem_id] = np.array([float(x) for x in parts[1:5]], dtype=float)
        return end

    @staticmethod
    def _parse_section_offset(lines: list[str], i: int, out: dict) -> int:
        data, end = _read_section_lines(lines, i)
        for line in data:
            parts = line.split()
            elem_id = int(parts[0])
            out[elem_id] = np.array([float(x) for x in parts[1:3]], dtype=float)
        return end

    @staticmethod
    def _parse_element_info(lines: list[str], i: int, out: dict) -> int:
        data, end = _read_section_lines(lines, i)
        for line in data:
            tokens = line.split(" ", 2)
            elem_id = int(tokens[0])
            geom_id = int(tokens[1])
            rest = tokens[2]

            geom_name, rest = _read_length_prefixed(rest)

            sub_idx_str, sub_type, rest = rest.split(" ", 2)
            sub_geom_idx = int(sub_idx_str)

            pp_id_str, rest = rest.split(" ", 1)
            pp_id = int(pp_id_str)
            pp_name, rest = _read_length_prefixed(rest)

            ep_id_str, rest = rest.split(" ", 1)
            ep_id = int(ep_id_str)
            ep_name, _ = _read_length_prefixed(rest)

            out[elem_id] = ElementInfo(
                element_id=elem_id,
                geom_id=geom_id,
                geom_name=geom_name,
                sub_geom_idx=sub_geom_idx,
                sub_geom_type=sub_type,
                physical_property_id=pp_id,
                physical_property_name=pp_name,
                element_property_id=ep_id,
                element_property_name=ep_name,
            )
        return end


# The legacy ``CData`` alias lives on the ``STKO_to_python.model``
# package surface (quiet) and at the deep path
# ``STKO_to_python.model.cdata`` (DeprecationWarning via PEP 562
# ``__getattr__`` shim). It is intentionally not declared on this
# canonical module so that the library never trips its own warning.
