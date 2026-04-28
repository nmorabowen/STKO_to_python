"""MPCO bucket META parser.

A *bucket* is the HDF5 group at
``MODEL_STAGE[<n>]/RESULTS/ON_ELEMENTS/<result_name>/<class[<bracket>]>``.
Each bucket carries a ``META`` subgroup describing the on-disk column
layout and a ``DATA`` subgroup with one dataset per step. The library
historically ignored META and assigned generic ``val_1, val_2, ...``
column names. This module reads META and produces real component names
(``P``, ``Mz``, ``Px_1``, ...) plus a structured layout for downstream
consumers.

Format reference: ``docs/mpco_format_conventions.md`` §1–§3, §11–§12.

Two bucket shapes are supported:

* **Closed-form** (e.g. ``ElasticBeam3d`` ``force``/``localForce``):
  ``GAUSS_IDS == [[-1]]`` sentinel, single COMPONENTS segment with
  node-suffixed names (``Px_1, Py_1, ..., Mz_2``). No integration points.

* **Line-stations** (e.g. force-/disp-based beam ``section.force``):
  ``GAUSS_IDS == [[0], [1], ..., [n_IP-1]]``, one COMPONENTS segment per
  integration point separated by ``;``, each with section response codes
  (``P, Mz, My, T, Vy, Vz``).

The parser also validates the invariant
``NUM_COLUMNS == sum(MULTIPLICITY * NUM_COMPONENTS)`` and raises
``MpcoFormatError`` on any mismatch — fail-loud per convention §15.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


__all__ = [
    "MpcoFormatError",
    "BucketLayout",
    "parse_bucket_meta",
]


class MpcoFormatError(ValueError):
    """Raised when an MPCO bucket's META violates an expected invariant.

    Always carries the bucket path (when available) plus expected vs.
    actual values to make the failure self-explaining.
    """


# --------------------------------------------------------------------- #
# BucketLayout                                                          #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class BucketLayout:
    """Resolved per-bucket column layout.

    Attributes
    ----------
    closed_form : bool
        True for closed-form (no integration point) buckets, identified
        by the ``GAUSS_IDS == [[-1]]`` sentinel.
    n_ip : int
        Number of integration points. ``0`` for closed-form, otherwise
        ``len(gauss_ids)``.
    gauss_ids : tuple[int, ...]
        Flattened ``GAUSS_IDS``. For closed-form, ``(-1,)``.
    ip_components : tuple[tuple[str, ...], ...]
        Component names per IP segment. For closed-form, length 1 with
        node-suffixed names. For line-stations, one tuple per IP.
    flat_columns : tuple[str, ...]
        Flat column names aligned to the on-disk DATA slab. For
        line-stations, names are suffixed ``_ip<gauss_id>`` to keep the
        flat vector unambiguous (``P_ip0, Mz_ip0, ..., P_ip4, ...``).
        Length equals ``num_columns``.
    num_columns : int
        Width of the DATA slab (``NUM_COLUMNS`` attribute).
    """

    closed_form: bool
    n_ip: int
    gauss_ids: tuple
    ip_components: tuple
    flat_columns: tuple
    num_columns: int


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _decode_components_blob(raw) -> str:
    """Decode the COMPONENTS dataset (shape (1,) bytes string) to str."""
    arr = np.asarray(raw)
    if arr.shape == ():
        val = arr.item()
    elif arr.size >= 1:
        val = arr.reshape(-1)[0]
    else:
        raise MpcoFormatError("META/COMPONENTS is empty")
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


def _extract_segment_components(segment: str) -> List[str]:
    """Take the substring after the LAST '.' and split on ','.

    Handles paths of arbitrary depth (``0.``, ``0.1.2.``, ``0.1.2.3.``)
    per format docs §10.
    """
    body = segment.rsplit(".", 1)[-1]
    return [tok.strip() for tok in body.split(",") if tok.strip()]


# --------------------------------------------------------------------- #
# Public API                                                            #
# --------------------------------------------------------------------- #


def parse_bucket_meta(bucket_grp, *, bucket_path: str | None = None) -> BucketLayout:
    """Read META from an MPCO results bucket and resolve its column layout.

    Parameters
    ----------
    bucket_grp : h5py.Group
        The bucket group (e.g. ``f[".../section.force/64-DispBeamColumn3d[1000:1:0]"]``).
    bucket_path : str, optional
        Used only in error messages. If omitted, the parser falls back
        to ``getattr(bucket_grp, "name", "<unknown>")``.

    Returns
    -------
    BucketLayout

    Raises
    ------
    MpcoFormatError
        If META is missing, ``NUM_COLUMNS`` is missing, the invariant
        ``NUM_COLUMNS == sum(MULTIPLICITY * NUM_COMPONENTS)`` fails, or
        the resolved flat-column count disagrees with ``NUM_COLUMNS``.
    """
    path = bucket_path or getattr(bucket_grp, "name", "<unknown>")

    if "META" not in bucket_grp:
        raise MpcoFormatError(f"{path}: missing META subgroup")
    meta = bucket_grp["META"]

    for k in ("MULTIPLICITY", "GAUSS_IDS", "NUM_COMPONENTS", "COMPONENTS"):
        if k not in meta:
            raise MpcoFormatError(f"{path}: META/{k} not found")

    multiplicity = np.asarray(meta["MULTIPLICITY"][()]).flatten().astype(np.int64)
    gauss_ids = np.asarray(meta["GAUSS_IDS"][()]).flatten().astype(np.int64)
    num_components = np.asarray(meta["NUM_COMPONENTS"][()]).flatten().astype(np.int64)
    components_str = _decode_components_blob(meta["COMPONENTS"][()])

    num_columns_attr = bucket_grp.attrs.get("NUM_COLUMNS")
    if num_columns_attr is None:
        raise MpcoFormatError(f"{path}: NUM_COLUMNS attribute missing on bucket")
    num_columns = int(np.asarray(num_columns_attr).flatten()[0])

    # Closed-form sentinel (§3).
    closed_form = gauss_ids.size == 1 and int(gauss_ids[0]) == -1

    # Parse semicolon-separated COMPONENTS segments (§2). Layered-shell
    # fiber buckets (e.g. section.fiber.damage on ASDShellQ4) include
    # *empty* segments like ``"0.1.2.3.4."`` for layers that don't carry
    # the requested quantity — pair these with NUM_COMPONENTS=0. Don't
    # filter them out: the segment count must equal MULTIPLICITY.size
    # for the per-block consistency check below.
    raw_segments = components_str.split(";")
    # Drop a trailing empty segment that some writers emit after a
    # final ``;``. We detect this by NUM_COMPONENTS not having that
    # extra block; truncate to match.
    if len(raw_segments) == multiplicity.size + 1 and raw_segments[-1].strip() == "":
        raw_segments = raw_segments[:-1]
    if not raw_segments:
        raise MpcoFormatError(f"{path}: META/COMPONENTS parsed to zero segments")

    ip_components = tuple(
        tuple(_extract_segment_components(s)) for s in raw_segments
    )

    # NUM_COLUMNS invariant (§11).
    expected_total = int((multiplicity * num_components).sum())
    if expected_total != num_columns:
        raise MpcoFormatError(
            f"{path}: NUM_COLUMNS invariant violated. "
            f"NUM_COLUMNS={num_columns} but "
            f"sum(MULTIPLICITY * NUM_COMPONENTS) = {expected_total} "
            f"(MULTIPLICITY={multiplicity.tolist()}, "
            f"NUM_COMPONENTS={num_components.tolist()})"
        )

    # All known bucket shapes have one COMPONENTS segment per block
    # (i.e. per row of MULTIPLICITY/NUM_COMPONENTS/GAUSS_IDS). When
    # ``MULT[i] > 1`` (compressed META, §12 — fibers per IP), the
    # segment describes a *single* column-group's component names that
    # repeats ``MULT[i]`` times in the slab.
    n_blocks = multiplicity.size
    if len(ip_components) != n_blocks:
        raise MpcoFormatError(
            f"{path}: META segment count ({len(ip_components)}) "
            f"!= number of blocks ({n_blocks}) "
            f"[MULTIPLICITY shape={multiplicity.tolist()}]"
        )

    # Per-block sanity: each segment must declare exactly NUM_COMPONENTS[i] names.
    for i, (comps, expected_n) in enumerate(zip(ip_components, num_components)):
        if len(comps) != int(expected_n):
            raise MpcoFormatError(
                f"{path}: block {i} META segment has {len(comps)} names "
                f"but NUM_COMPONENTS[{i}]={int(expected_n)} "
                f"(segment={comps})"
            )

    # Closed-form: a single block with MULT=[1], the segment is already
    # the flat column list (Px_1, ..., Mz_2) — names not suffixed.
    if closed_form:
        if n_blocks != 1 or int(multiplicity[0]) != 1:
            raise MpcoFormatError(
                f"{path}: closed-form bucket (GAUSS_IDS=[[-1]]) expected "
                f"a single block with MULTIPLICITY=[[1]], got "
                f"MULTIPLICITY={multiplicity.tolist()}"
            )
        flat_columns = tuple(ip_components[0])
        if len(flat_columns) != num_columns:
            raise MpcoFormatError(
                f"{path}: closed-form flat columns ({len(flat_columns)}) "
                f"!= NUM_COLUMNS ({num_columns}). Names: {flat_columns}"
            )
        return BucketLayout(
            closed_form=True,
            n_ip=0,
            gauss_ids=tuple(int(g) for g in gauss_ids),
            ip_components=ip_components,
            flat_columns=flat_columns,
            num_columns=num_columns,
        )

    # Non-closed-form: blocks correspond to (gauss-point × layer)
    # pairs. The unique GAUSS_IDS values must form 0..n_unique-1, and
    # the array must be non-decreasing so we can group blocks by IP.
    # Layered shells repeat each gauss-id once per thickness layer
    # (e.g. ``[0,0,0,0,0, 1,1,1,1,1, ...]`` for 4 IPs × 5 layers); plain
    # gauss-level buckets have each gauss-id exactly once.
    if gauss_ids.size == 0:
        raise MpcoFormatError(f"{path}: GAUSS_IDS is empty")
    if not np.all(np.diff(gauss_ids) >= 0):
        raise MpcoFormatError(
            f"{path}: GAUSS_IDS must be non-decreasing for "
            f"non-closed-form buckets, got {gauss_ids.tolist()}"
        )
    unique_gids = np.unique(gauss_ids)
    expected_unique = np.arange(unique_gids.size, dtype=np.int64)
    if not np.array_equal(unique_gids, expected_unique):
        raise MpcoFormatError(
            f"{path}: unique GAUSS_IDS must be sequential 0..n-1 "
            f"for non-closed-form buckets, got {unique_gids.tolist()}"
        )

    # If any gauss-point repeats, this is a layered bucket. Compute
    # each block's layer index = its position among prior blocks with
    # the same gauss-id. n_layers_per_ip is the max repetition count.
    has_layers = bool((np.bincount(gauss_ids) > 1).any())
    layer_counter: Dict[int, int] = {}

    # Build the flat column vector. Suffix conventions:
    #   * plain gauss / line-stations (MULT=1, no layers):
    #         <comp>_ip<gid>
    #   * compressed fibers (MULT>1, no layers):
    #         <comp>_f<fiber>_ip<gid>
    #   * layered (gauss-id repeats, MULT=1):
    #         <comp>_l<layer>_ip<gid>
    #   * layered + fibers (gauss-id repeats, MULT>1):
    #         <comp>_f<fiber>_l<layer>_ip<gid>
    flat_list: List[str] = []
    for gid, mult_i, comps in zip(
        gauss_ids.tolist(), multiplicity.tolist(), ip_components
    ):
        gid = int(gid)
        mult_i = int(mult_i)
        layer_idx = layer_counter.get(gid, 0)
        layer_counter[gid] = layer_idx + 1
        # Empty NUM_COMPONENTS / segment → no columns from this block.
        if not comps:
            continue
        for fiber_j in range(mult_i):
            for c in comps:
                parts: List[str] = [c]
                if mult_i > 1:
                    parts.append(f"f{fiber_j}")
                if has_layers:
                    parts.append(f"l{layer_idx}")
                parts.append(f"ip{gid}")
                flat_list.append("_".join(parts))

    if len(flat_list) != num_columns:
        raise MpcoFormatError(
            f"{path}: line-station flat columns ({len(flat_list)}) "
            f"!= NUM_COLUMNS ({num_columns}). "
            f"ip_components={ip_components}"
        )

    return BucketLayout(
        closed_form=False,
        # ``n_ip`` is the number of *geometric* integration points
        # (unique gauss-ids), not the raw block count. Layered buckets
        # have one block per (gauss × layer) but n_ip stays at the
        # gauss-only count so it matches the catalog and the physical
        # element geometry.
        n_ip=int(unique_gids.size),
        gauss_ids=tuple(int(g) for g in gauss_ids),
        ip_components=ip_components,
        flat_columns=tuple(flat_list),
        num_columns=num_columns,
    )


def validate_data_shape(
    layout: BucketLayout,
    data_shape: Sequence[int],
    *,
    bucket_path: str | None = None,
) -> None:
    """Cross-check a DATA dataset's column count against the layout.

    Called once per bucket from the read path, before any data is
    consumed, to refuse silently mis-sized slabs.

    Parameters
    ----------
    layout : BucketLayout
        Result of :func:`parse_bucket_meta`.
    data_shape : sequence of int
        Shape of one step's DATA dataset (typically ``(n_elems, n_cols)``).
    bucket_path : str, optional
        Used in error messages.

    Raises
    ------
    MpcoFormatError
        If ``data_shape[1] != layout.num_columns``.
    """
    if len(data_shape) < 2:
        raise MpcoFormatError(
            f"{bucket_path or '<unknown>'}: DATA shape {tuple(data_shape)} "
            f"is not 2-D"
        )
    n_cols = int(data_shape[1])
    if n_cols != layout.num_columns:
        raise MpcoFormatError(
            f"{bucket_path or '<unknown>'}: DATA n_cols={n_cols} "
            f"disagrees with META NUM_COLUMNS={layout.num_columns}"
        )
