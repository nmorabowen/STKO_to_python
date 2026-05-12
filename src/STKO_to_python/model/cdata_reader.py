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
- ``*BEAM_PROFILE`` — 2D cross-section geometry (points, triangles,
  edges, sweeps) per profile id. Useful for plotting beam elements
  as extruded solids.
- ``*BEAM_PROFILE_ASSIGNMENT`` — element-id to profile-id mapping
  with weights, expressing variable cross-section along an element.

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

from .cdata_format import CDataFormatPolicy


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


@dataclass(frozen=True)
class BeamProfile:
    """2D cross-section geometry for a beam element profile.

    Mirrors one block inside a ``*BEAM_PROFILE`` section.

    - ``points`` are vertices in the section's local 2D frame.
    - ``triangles`` is a triangulation of the section (for filled
      rendering); each row is three 0-based point indices.
    - ``edges`` is the outline as a list of polyline edges; each
      edge is a 0-based ``ndarray`` of point indices, length varies
      per edge (STKO supports curved/multi-point edges).
    - ``sweeps`` indexes the points to sweep along the beam (for
      generating an extruded 3D mesh).
    """
    profile_id: int
    points: np.ndarray           # (npoints, 2) — x, y in section frame
    triangles: np.ndarray        # (ntriangles, 3) — 0-based point indices
    edges: list[np.ndarray]      # one (n_i,) array per edge, n_i varies
    sweeps: np.ndarray           # (nsweeps,) — 0-based point indices


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


def _consume_ids(
    lines: list[str], start: int, count: int
) -> tuple[np.ndarray, int]:
    """Consume exactly *count* integers starting at ``lines[start]``.

    Walks forward across as many lines as needed, regardless of how
    STKO chose to wrap the list. The current emitter uses 10 ids per
    line, but this helper is width-agnostic so a hand-edited file
    (or a future format variant) still parses correctly.

    Returns ``(ndarray, next_index)`` where ``next_index`` is the
    index of the first line *not* consumed.

    Raises ``ValueError`` if EOF or a section boundary (``*``)
    is reached before *count* ids have been gathered.
    """
    if count == 0:
        return np.empty(0, dtype=int), start

    collected: list[int] = []
    i = start
    n = len(lines)
    while len(collected) < count:
        if i >= n:
            raise ValueError(
                f"cdata: expected {count} ids, got only {len(collected)} "
                f"before EOF (started at line {start + 1})."
            )
        s = lines[i].strip()
        if not s or s.startswith("*"):
            raise ValueError(
                f"cdata: id list truncated at line {i + 1}; expected "
                f"{count} ids, got {len(collected)}."
            )
        collected.extend(int(x) for x in s.split())
        i += 1

    # If the last consumed line happened to have more ids than we
    # needed, trim. STKO doesn't currently emit this shape (its wrap
    # is exact), but the trim makes us safe against hand-edits.
    arr = np.array(collected[:count], dtype=int)
    return arr, i


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

    # Format policy is stateless and identical for every reader; share
    # one class-level instance rather than allocating per dataset.
    _format_policy: CDataFormatPolicy = CDataFormatPolicy()

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

    @cached_property
    def beam_profiles(self) -> dict[int, BeamProfile]:
        """``{profile_id -> BeamProfile}`` 2D cross-section definitions.

        Each entry holds the section's points, triangulation, edge
        outline, and sweep indices. Profile definitions are typically
        repeated identically across MP partitions; only the first
        occurrence is kept.
        """
        return self._parse_all_files()["beam_profiles"]

    @cached_property
    def beam_profile_assignments(self) -> dict[int, list[tuple[int, float]]]:
        """``{elem_id -> [(profile_id, weight), ...]}`` element → profile map.

        Each tuple is one profile assigned to the element with a
        weight in ``[0, 1]`` expressing relative span along the
        element. The ``beam_profiles`` accessor resolves ``profile_id``
        to the underlying cross-section geometry.
        """
        return self._parse_all_files()["beam_profile_assignments"]

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
            "selection_sets": {},              # int -> {"SET_NAME": str, "NODES": set, "ELEMENTS": set}
            "local_axes": {},                  # int -> ndarray(4,)
            "section_offsets": {},             # int -> ndarray(2,)
            "element_info": {},                # int -> ElementInfo
            "beam_profiles": {},               # int -> BeamProfile
            "beam_profile_assignments": {},    # int -> list[tuple[int, float]]
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

            policy = self._format_policy
            i = 0
            n = len(lines)
            while i < n:
                marker = lines[i].strip()
                if marker == policy.MARKER_SELECTION_SET:
                    i = self._parse_one_selection_set(lines, i, accum["selection_sets"])
                elif marker == policy.MARKER_LOCAL_AXES:
                    i = self._parse_local_axes(lines, i + 1, accum["local_axes"])
                elif marker == policy.MARKER_SECTION_OFFSET:
                    i = self._parse_section_offset(lines, i + 1, accum["section_offsets"])
                elif marker == policy.MARKER_BEAM_PROFILE:
                    i = self._parse_beam_profiles(lines, i + 1, accum["beam_profiles"])
                elif marker == policy.MARKER_BEAM_PROFILE_ASSIGNMENT:
                    i = self._parse_beam_profile_assignments(
                        lines, i + 1, accum["beam_profile_assignments"]
                    )
                elif marker == policy.MARKER_ELEMENT_INFO:
                    i = self._parse_element_info(lines, i + 1, accum["element_info"])
                else:
                    i += 1
        except Exception:
            logger.exception("CData parse error in %s", file_path)
            raise

    # --- per-section parsers ----------------------------------------- #

    @staticmethod
    def _parse_one_selection_set(lines: list[str], i: int, out: dict) -> int:
        """Parse one ``*SELECTION_SET`` block; return index after the block.

        Node and element id lists are read via ``_consume_ids`` which is
        width-agnostic — STKO currently wraps at 10 ids per line, but
        this parser also accepts 1-per-line, all-on-one-line, or any
        consistent variant.
        """
        set_id = int(lines[i + 1].strip())

        raw_name = lines[i + 2].strip()
        name_length = int(raw_name.split()[0])
        offset = len(str(name_length)) + 1
        set_name = raw_name[offset : offset + name_length]

        n_nodes = int(lines[i + 3].strip())
        n_elems = int(lines[i + 4].strip())

        # Both lists live right after the header; nodes first, elements
        # second. _consume_ids advances the cursor past whichever lines
        # the wrap width happened to use.
        cursor = i + 5
        nodes, cursor = _consume_ids(lines, cursor, n_nodes)
        elements, cursor = _consume_ids(lines, cursor, n_elems)

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
    def _parse_beam_profiles(lines: list[str], i: int, out: dict) -> int:
        """Parse one ``*BEAM_PROFILE`` section (may contain N profiles back-to-back).

        Per-profile layout:
            PROFILE_ID
            NPOINTS NTRIANGLES NEDGES NSWEEPS
            <NPOINTS rows of (x y)>
            <NTRIANGLES rows of (p1 p2 p3)>
            <NEDGES rows of (N p1 p2 ... pN)>     # variable length
            <NSWEEPS rows of (p1)>
        """
        data, end = _read_section_lines(lines, i)
        cursor = 0
        n = len(data)
        while cursor < n:
            profile_id = int(data[cursor].strip())
            cursor += 1
            counts = [int(x) for x in data[cursor].split()]
            n_pts, n_tris, n_edges, n_sweeps = counts[:4]
            cursor += 1

            points = np.array(
                [[float(v) for v in data[cursor + k].split()] for k in range(n_pts)],
                dtype=float,
            )
            cursor += n_pts

            triangles = np.array(
                [[int(v) for v in data[cursor + k].split()] for k in range(n_tris)],
                dtype=int,
            )
            cursor += n_tris

            edges: list[np.ndarray] = []
            for _ in range(n_edges):
                parts = [int(x) for x in data[cursor].split()]
                # parts[0] is N (count of points in this edge), parts[1:N+1] are the indices.
                edges.append(np.array(parts[1 : 1 + parts[0]], dtype=int))
                cursor += 1

            sweeps = np.array(
                [int(data[cursor + k].strip()) for k in range(n_sweeps)],
                dtype=int,
            )
            cursor += n_sweeps

            # Profile definitions are duplicated identically across MP
            # partitions; keep the first occurrence and drop later ones.
            if profile_id not in out:
                out[profile_id] = BeamProfile(
                    profile_id=profile_id,
                    points=points,
                    triangles=triangles,
                    edges=edges,
                    sweeps=sweeps,
                )
        return end

    @staticmethod
    def _parse_beam_profile_assignments(lines: list[str], i: int, out: dict) -> int:
        """Parse a ``*BEAM_PROFILE_ASSIGNMENT`` section.

        Each row is ``ELEM_ID N_PROFILES PID_1 WEIGHT_1 ... PID_N WEIGHT_N``.
        """
        data, end = _read_section_lines(lines, i)
        for line in data:
            parts = line.split()
            elem_id = int(parts[0])
            n_prof = int(parts[1])
            assignments: list[tuple[int, float]] = []
            for k in range(n_prof):
                pid = int(parts[2 + 2 * k])
                weight = float(parts[3 + 2 * k])
                assignments.append((pid, weight))
            out[elem_id] = assignments
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
