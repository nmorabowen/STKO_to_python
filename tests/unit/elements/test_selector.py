"""Unit tests for ``ElementSelector``.

The tests use a synthetic, self-contained ``ElementManager`` stand-in so
they don't depend on any ``.mpco`` fixture being on disk. The mock
exposes the two interfaces the selector touches:

* ``_ensure_elem_index_df()`` — pandas DataFrame matching
  :attr:`ElementManager._ELEM_DTYPE`'s columns.
* ``dataset._selection_resolver.resolve_elements(...)`` — the same
  resolver protocol used in production.

The geometry is a uniform 3 × 3 × 3 cubic grid of beam-column elements
plus a 3 × 3 × 3 grid of shell-quad elements (offset on the y-axis), so
that universe behavior under ``of_type`` is exercised across multiple
classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import pytest

from STKO_to_python.elements.selector import ElementSelector


# ---------------------------------------------------------------------- #
# Fixtures — synthetic element index                                     #
# ---------------------------------------------------------------------- #

BEAM_TYPE = "64-DispBeamColumn3d"
SHELL_TYPE = "203-ASDShellQ4"


def _build_index() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (elements_df, nodes_df).

    27 beam elements at integer grid positions (x, y=0, z) for x∈[0,2],
    z∈[0,2], with one beam per cell; plus 27 shell elements offset to
    y=10 so the two element classes are spatially well-separated.
    """
    # ----- nodes -----
    nodes: list[dict] = []
    nid = 1
    # Beam nodes: integer grid + (0..2) x (0) x (0..2)
    beam_node_ids: dict[tuple[int, int, int], int] = {}
    for ix in range(4):  # 0..3 → 4 layers per axis to support cells
        for iz in range(4):
            beam_node_ids[(ix, 0, iz)] = nid
            nodes.append(
                {"node_id": nid, "x": float(ix), "y": 0.0, "z": float(iz)}
            )
            nid += 1
    # Shell nodes: offset by y=10
    shell_node_ids: dict[tuple[int, int, int], int] = {}
    for ix in range(4):
        for iy in range(4):
            shell_node_ids[(ix, iy, 0)] = nid
            nodes.append(
                {
                    "node_id": nid,
                    "x": float(ix),
                    "y": 10.0 + float(iy),
                    "z": 0.0,
                }
            )
            nid += 1

    df_nodes = pd.DataFrame(nodes)

    # ----- beam elements (line, 2 nodes each) -----
    beam_rows: list[dict] = []
    eid = 1
    beam_decorated = f"{BEAM_TYPE}[1000:0]"
    for ix in range(3):  # x cell index 0..2
        for iz in range(3):  # z cell index 0..2
            n1 = beam_node_ids[(ix, 0, iz)]
            n2 = beam_node_ids[(ix + 1, 0, iz)]
            cx = (float(ix) + float(ix + 1)) / 2.0
            cz = float(iz)
            beam_rows.append(
                {
                    "element_id": eid,
                    "element_idx": eid - 1,
                    "file_id": 0,
                    "element_type": BEAM_TYPE,
                    "decorated_type": beam_decorated,
                    "node_list": (n1, n2),
                    "num_nodes": 2,
                    "centroid_x": cx,
                    "centroid_y": 0.0,
                    "centroid_z": cz,
                }
            )
            eid += 1

    # ----- shell elements (quad, 4 nodes each) -----
    shell_rows: list[dict] = []
    shell_decorated = f"{SHELL_TYPE}[200:0]"
    for ix in range(3):
        for iy in range(3):
            n1 = shell_node_ids[(ix, iy, 0)]
            n2 = shell_node_ids[(ix + 1, iy, 0)]
            n3 = shell_node_ids[(ix + 1, iy + 1, 0)]
            n4 = shell_node_ids[(ix, iy + 1, 0)]
            cx = float(ix) + 0.5
            cy = 10.0 + float(iy) + 0.5
            shell_rows.append(
                {
                    "element_id": eid,
                    "element_idx": eid - 28,
                    "file_id": 0,
                    "element_type": SHELL_TYPE,
                    "decorated_type": shell_decorated,
                    "node_list": (n1, n2, n3, n4),
                    "num_nodes": 4,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "centroid_z": 0.0,
                }
            )
            eid += 1

    df_elems = pd.DataFrame(beam_rows + shell_rows)
    return df_elems, df_nodes


# ---------------------------------------------------------------------- #
# Mock manager / dataset                                                  #
# ---------------------------------------------------------------------- #

@dataclass
class _MockSelResolver:
    element_sets: dict  # name -> np.ndarray; also accepts int ids in id_sets
    id_sets: dict  # int -> np.ndarray

    def resolve_elements(
        self,
        *,
        names: Optional[Sequence[str]] = None,
        ids: Optional[Sequence[int]] = None,
        explicit_ids: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        gathered: list[np.ndarray] = []
        if names:
            for n in names:
                key = str(n).strip().lower()
                if key not in self.element_sets:
                    raise ValueError(f"unknown set: {n}")
                gathered.append(self.element_sets[key])
        if ids:
            for sid in ids:
                if int(sid) not in self.id_sets:
                    raise ValueError(f"unknown set id: {sid}")
                gathered.append(self.id_sets[int(sid)])
        if explicit_ids is not None:
            gathered.append(np.asarray(list(explicit_ids), dtype=np.int64))
        if not gathered:
            raise ValueError("no inputs")
        return np.unique(np.concatenate(gathered))


class _MockDataset:
    def __init__(self, df_nodes: pd.DataFrame, resolver: _MockSelResolver):
        self.nodes_info = {"dataframe": df_nodes}
        self._selection_resolver = resolver


class _MockManager:
    """Stand-in for :class:`ElementManager` exposing only the surface
    that :class:`ElementSelector` touches."""

    def __init__(self, df_elems: pd.DataFrame, df_nodes: pd.DataFrame, resolver):
        self._df = df_elems
        self.dataset = _MockDataset(df_nodes, resolver)

    def _ensure_elem_index_df(self) -> pd.DataFrame:
        return self._df


@pytest.fixture
def mgr() -> _MockManager:
    df_elems, df_nodes = _build_index()
    # Two named sets:
    #   "FirstColumn"  — beams at x∈[0,1] with low z
    #   "Outer"        — shells on the perimeter (outermost ring)
    beam_first = (
        df_elems[
            (df_elems["element_type"] == BEAM_TYPE)
            & (df_elems["centroid_x"] <= 1.0)
        ]["element_id"]
        .to_numpy(np.int64)
    )
    shell_outer = (
        df_elems[
            (df_elems["element_type"] == SHELL_TYPE)
            & (
                (df_elems["centroid_x"] < 1.0)
                | (df_elems["centroid_x"] > 2.0)
                | (df_elems["centroid_y"] < 11.0)
                | (df_elems["centroid_y"] > 12.0)
            )
        ]["element_id"]
        .to_numpy(np.int64)
    )
    resolver = _MockSelResolver(
        element_sets={"firstcolumn": beam_first, "outer": shell_outer},
        id_sets={1: beam_first, 2: shell_outer},
    )
    return _MockManager(df_elems, df_nodes, resolver)


# ---------------------------------------------------------------------- #
# Tests — anchors                                                        #
# ---------------------------------------------------------------------- #

def test_select_returns_empty_selector(mgr: _MockManager):
    sel = ElementSelector(mgr)
    # No anchor → unanchored selector still resolves over the full index
    assert sel.df().shape[0] == 18  # 9 beams + 9 shells
    assert sel.count() == 18


def test_of_type_filters_to_class(mgr: _MockManager):
    sel = ElementSelector(mgr).of_type("DispBeamColumn3d")
    assert sel.count() == 9
    assert set(sel.ids().tolist()) == set(range(1, 10))


def test_of_type_strips_decorated_bracket(mgr: _MockManager):
    """``.of_type`` should accept either bare or decorated names."""
    bare = ElementSelector(mgr).of_type("DispBeamColumn3d")
    decorated = ElementSelector(mgr).of_type("64-DispBeamColumn3d[1000:0]")
    # Both anchors should resolve to the same 27 beams.
    assert set(bare.ids().tolist()) == set(decorated.ids().tolist())


def test_from_selection_by_name(mgr: _MockManager):
    sel = ElementSelector(mgr).from_selection("FirstColumn")
    assert sel.count() > 0
    # All resolved IDs are within the beam class.
    df = mgr._ensure_elem_index_df()
    beam_ids = set(
        df[df["element_type"] == BEAM_TYPE]["element_id"].tolist()
    )
    assert set(sel.ids().tolist()).issubset(beam_ids)


def test_from_selection_by_id(mgr: _MockManager):
    sel = ElementSelector(mgr).from_selection(2)  # "Outer" by id
    df = mgr._ensure_elem_index_df()
    shell_ids = set(
        df[df["element_type"] == SHELL_TYPE]["element_id"].tolist()
    )
    assert set(sel.ids().tolist()).issubset(shell_ids)


def test_with_ids_anchor(mgr: _MockManager):
    sel = ElementSelector(mgr).with_ids([1, 5, 9, 12])
    assert sorted(sel.ids().tolist()) == [1, 5, 9, 12]


# ---------------------------------------------------------------------- #
# Tests — spatial primitives                                             #
# ---------------------------------------------------------------------- #

def test_within_box_centroid(mgr: _MockManager):
    """Box covering the lower-left 2x2 of the beam grid (x∈[0,1.5])."""
    sel = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .within_box(min=(0.0, -0.5, 0.0), max=(1.5, 0.5, 1.5))
    )
    # Beams with centroid_x ∈ {0.5, 1.5} and centroid_z ∈ {0,1} → 2*2 = 4
    # but centroid_x=1.5 is on the boundary (inclusive) — yes.
    assert sel.count() == 4


def test_within_box_any_node(mgr: _MockManager):
    """``any_node`` mode picks up edge elements that the centroid mode misses."""
    sel_centroid = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .within_box(min=(0.0, -0.5, 0.0), max=(0.4, 0.5, 0.5), mode="centroid")
    )
    sel_any = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .within_box(min=(0.0, -0.5, 0.0), max=(0.4, 0.5, 0.5), mode="any_node")
    )
    # Centroid mode catches nothing (smallest beam centroid x=0.5).
    assert sel_centroid.count() == 0
    # any_node: only beam 1 has a node (0,0,0) within the small box;
    # beams 2 and 3 sit at z>=1 so are out of the box.
    assert sel_any.count() == 1
    assert sel_any.ids().tolist() == [1]


def test_within_distance(mgr: _MockManager):
    """Sphere centered at first-row beam midpoint."""
    sel = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .within_distance(point=(0.5, 0.0, 0.0), radius=0.1)
    )
    assert sel.count() == 1
    assert sel.ids().tolist() == [1]


def test_nearest_to_k(mgr: _MockManager):
    sel = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .nearest_to(point=(0.5, 0.0, 0.0), k=3)
    )
    assert sel.count() == 3
    # First should be element 1 (centroid (0.5, 0, 0)).
    assert sel.ids()[0] == 1


def test_nearest_to_zero_returns_empty(mgr: _MockManager):
    sel = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .nearest_to(point=(0.0, 0.0, 0.0), k=0)
    )
    assert sel.count() == 0


def test_on_plane_axis_aligned(mgr: _MockManager):
    """Plane z=1 cuts beams whose nodes straddle it. Beams are line
    elements with both nodes at the same z, so only beams *exactly on*
    z=1 are kept."""
    sel = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .on_plane(z=1.0)
    )
    # Beams in the z=1 row → 3.
    assert sel.count() == 3
    assert all(
        mgr._ensure_elem_index_df()
        .set_index("element_id")
        .loc[eid, "centroid_z"]
        == 1.0
        for eid in sel.ids()
    )


def test_on_plane_general_normal(mgr: _MockManager):
    """A plane normal (1,0,0) at x=1.0 — should pick up beams that span
    x=1 (beams whose two endpoints are at x=ix and x=ix+1)."""
    sel = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .on_plane(point=(1.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
    )
    # Beams with endpoints (0,1) → straddle x=1: yes, plus (1,2) just
    # touch x=1. Specifically, every beam with endpoint at x=1 OR
    # straddling x=1 → ix=0 (endpoints 0,1; touches plane) and ix=1
    # (endpoints 1,2; touches plane). 6 beams.
    assert sel.count() == 6


def test_near_line(mgr: _MockManager):
    """A line along x at y=0,z=0 — radius 0.1 catches the bottom row of beams."""
    sel = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .near_line(p0=(0.0, 0.0, 0.0), p1=(3.0, 0.0, 0.0), radius=0.1)
    )
    # Beams at z=0 (y always 0): 3 of them.
    assert sel.count() == 3


def test_centroid_in_axis(mgr: _MockManager):
    sel = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .centroid_in("z", lo=0.5, hi=1.5)
    )
    # Beams with centroid_z = 1 → 3.
    assert sel.count() == 3


def test_centroid_in_one_sided(mgr: _MockManager):
    sel = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .centroid_in("z", hi=0.5)
    )
    # centroid_z = 0 → 3 beams.
    assert sel.count() == 3


def test_where_predicate(mgr: _MockManager):
    """Predicate escape hatch — keep beams with even element_id."""
    sel = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .where(lambda df: (df["element_id"] % 2 == 0).to_numpy())
    )
    # 9 beams; even ids are 2,4,6,8 → 4.
    assert sel.count() == 4


def test_where_predicate_shape_mismatch_raises(mgr: _MockManager):
    sel = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .where(lambda df: np.array([True, False]))
    )
    with pytest.raises(ValueError, match="predicate returned shape"):
        sel.ids()


# ---------------------------------------------------------------------- #
# Tests — chaining (AND-narrowing)                                        #
# ---------------------------------------------------------------------- #

def test_chain_combines_via_and(mgr: _MockManager):
    """Chained primitives narrow conjunctively."""
    sel = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .within_box(min=(0.0, -0.5, 0.0), max=(1.5, 0.5, 1.5))
        .nearest_to(point=(0.5, 0.0, 0.0), k=2)
    )
    assert sel.count() == 2
    assert sel.ids()[0] == 1


# ---------------------------------------------------------------------- #
# Tests — boolean composition (& | ~)                                    #
# ---------------------------------------------------------------------- #

def test_intersection(mgr: _MockManager):
    a = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .within_box(min=(0.0, -0.5, 0.0), max=(2.5, 0.5, 0.5))
    )
    b = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .centroid_in("x", lo=0.0, hi=1.0)
    )
    out = (a & b).ids().tolist()
    # a: beams at z=0 → 3, b: beams with centroid_x ∈ [0,1] → 3 (x=0.5).
    # Intersection: beams at z=0 AND x≤1 → 1 (element 1 at (0.5, 0, 0)).
    assert sorted(out) == [1]


def test_union(mgr: _MockManager):
    a = ElementSelector(mgr).of_type("DispBeamColumn3d").with_ids([1, 2])
    b = ElementSelector(mgr).of_type("DispBeamColumn3d").with_ids([2, 3])
    assert sorted((a | b).ids().tolist()) == [1, 2, 3]


def test_negation_universe_is_of_type_anchor(mgr: _MockManager):
    """``~`` against ``of_type``-anchored selector returns the *other*
    elements of that same class — never crosses class boundaries."""
    a = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .within_box(min=(0.0, -0.5, 0.0), max=(1.5, 0.5, 1.5))
    )
    in_a = set(a.ids().tolist())
    not_a = set((~a).ids().tolist())
    all_beams = set(range(1, 10))
    assert in_a.union(not_a) == all_beams
    assert in_a.intersection(not_a) == set()
    # No shells leak into the negation.
    assert not_a.intersection(set(range(10, 19))) == set()


def test_negation_without_anchor_raises(mgr: _MockManager):
    sel = ElementSelector(mgr).within_box(
        min=(0.0, -0.5, 0.0), max=(1.5, 0.5, 1.5)
    )
    with pytest.raises(ValueError, match="without an of_type"):
        ~sel


def test_double_negation_returns_original(mgr: _MockManager):
    a = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .within_box(min=(0.0, -0.5, 0.0), max=(1.5, 0.5, 1.5))
    )
    assert sorted((~~a).ids().tolist()) == sorted(a.ids().tolist())


def test_combined_universe_for_or_is_union(mgr: _MockManager):
    """For ``a | b``, the negation universe is the union of universes."""
    a = ElementSelector(mgr).of_type("DispBeamColumn3d").within_box(
        min=(0.0, -0.5, 0.0), max=(0.6, 0.5, 0.6)
    )
    b = ElementSelector(mgr).of_type("ASDShellQ4").within_box(
        min=(0.0, 9.5, -0.1), max=(0.6, 10.6, 0.1)
    )
    union_uni = (a | b)._universe_ids()
    # Beams (universe of a) have ids 1..9, shells 10..18 → 1..18.
    assert sorted(union_uni.tolist()) == list(range(1, 19))


def test_combined_universe_for_and_is_intersection(mgr: _MockManager):
    """For ``a & b``, the negation universe is the intersection of universes."""
    a = ElementSelector(mgr).of_type("DispBeamColumn3d").with_ids([1, 2])
    b = ElementSelector(mgr).of_type("DispBeamColumn3d").with_ids([2, 3, 4])
    intersect_uni = (a & b)._universe_ids()
    # Both beams: universe is {1,2} ∩ {2,3,4} = {2}.
    assert intersect_uni.tolist() == [2]


def test_combinator_cannot_take_anchor_methods(mgr: _MockManager):
    a = ElementSelector(mgr).of_type("DispBeamColumn3d").with_ids([1])
    b = ElementSelector(mgr).of_type("DispBeamColumn3d").with_ids([2])
    combo = a & b
    with pytest.raises(TypeError, match="combined selector"):
        combo.of_type("ASDShellQ4")


def test_combinator_chains_via_boolean_again(mgr: _MockManager):
    a = ElementSelector(mgr).of_type("DispBeamColumn3d").with_ids([1, 2, 3])
    b = ElementSelector(mgr).of_type("DispBeamColumn3d").with_ids([2, 3, 4])
    c = ElementSelector(mgr).of_type("DispBeamColumn3d").with_ids([3, 4, 5])
    # (a & b) | c == {2,3} ∪ {3,4,5} = {2,3,4,5}
    assert sorted(((a & b) | c).ids().tolist()) == [2, 3, 4, 5]


# ---------------------------------------------------------------------- #
# Tests — outputs (.ids/.df/.mask/.count)                                 #
# ---------------------------------------------------------------------- #

def test_mask_indexed_by_universe(mgr: _MockManager):
    sel = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .with_ids([1, 5, 9])
    )
    m = sel.mask()
    assert m.sum() == 3
    # Mask universe is the of_type ∩ with_ids anchor → {1, 5, 9}
    assert sorted(m.index.tolist()) == [1, 5, 9]
    assert all(m.values)


def test_df_returns_index_rows(mgr: _MockManager):
    sel = ElementSelector(mgr).of_type("DispBeamColumn3d").with_ids([7])
    out = sel.df()
    assert len(out) == 1
    assert int(out["element_id"].iloc[0]) == 7
    assert out["element_type"].iloc[0] == BEAM_TYPE


def test_repr_includes_anchor_and_ops(mgr: _MockManager):
    sel = (
        ElementSelector(mgr)
        .of_type("DispBeamColumn3d")
        .within_box(min=(0, 0, 0), max=(1, 1, 1))
    )
    r = repr(sel)
    assert "DispBeamColumn3d" in r
    assert "WithinBoxOp" in r


# ---------------------------------------------------------------------- #
# Tests — immutability                                                   #
# ---------------------------------------------------------------------- #

def test_selectors_are_immutable(mgr: _MockManager):
    base = ElementSelector(mgr).of_type("DispBeamColumn3d")
    a = base.within_box(min=(0, -1, 0), max=(1, 1, 1))
    b = base.within_box(min=(2, -1, 0), max=(3, 1, 3))
    # base unchanged; a and b are independent views
    assert base.count() == 9
    # Strict check: re-resolving base still gives 9 (no mutation).
    assert base.count() == 9
    # a and b cover disjoint x-ranges so produce disjoint id sets.
    assert set(a.ids().tolist()).isdisjoint(set(b.ids().tolist()))
