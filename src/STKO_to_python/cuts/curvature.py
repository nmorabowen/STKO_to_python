"""In-plane wall curvature from section cuts.

Plane-sections companion to :func:`~.kernels.shell.compute_shell_cut`: where
the force cut integrates the traction along the cut, this module reads the
recorded generalized strains (``section.deformation``) on the same shell
intersections and extracts, per wall pier,

    kappa(t) = d(epsYY)/ds       (weighted least-squares along the cut)
    eps0(t)  = centroidal axial strain

— the deformation analogue of the in-plane moment being the first moment of
the axial traction. ``kappa`` is the natural carrier for ductility reporting
(normalize by a design yield curvature, e.g. Priestley's ~2*eps_y/l_w,
outside this module).

Piers are resolved from the chord geometry itself: chords are grouped by
direction family and wall-plane coordinate, then split on contiguity gaps
along the cut (openings). This mirrors the validated San Ramon pier
clustering (4 piers, symmetric twins; 2026-07-28).

Notes
-----
- Uses the Gauss-point *mean* of ``epsYY`` per element (the convention
  validated against the San Ramon campaign); chord-point sampling via the
  shape-function machinery is a possible refinement.
- ``epsYY`` is the element-local vertical membrane strain. For wall meshes
  with an assigned in-plane horizontal local X (the STKO ``-local``
  convention), local Y is vertical, which is what the plane-sections
  reading requires. A wall whose local axes are NOT of that form would
  need a rotation step — detected and rejected via the element rotation
  matrix check below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .plane import Plane
from .specs import SectionCutSpec, _load_pickle, _save_pickle
from .kernels.shell import ShellIntersection, find_shell_intersections

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet

#: tolerance for accepting local-Y as "vertical" (plane-sections reading)
_VERTICAL_TOL = 0.99


@dataclass(frozen=True)
class CutCurvatureResult:
    """Per-pier curvature histories for one cut.

    Attributes
    ----------
    df : pd.DataFrame
        Index = step; MultiIndex columns ``(pier, comp)`` with comp in
        {'kappa', 'eps0'}. ``kappa`` in 1/length units; positive means
        axial strain increasing toward +s (the pier's local along-wall
        coordinate).
    time : np.ndarray
    piers : pd.DataFrame
        One row per pier: family, plane coordinate, centroid s, length,
        and the element ids that built it.
    spec : SectionCutSpec
    model_stage : str
    """

    df: pd.DataFrame
    time: np.ndarray
    piers: pd.DataFrame
    spec: SectionCutSpec
    model_stage: str

    def series(self, pier: str, comp: str = "kappa") -> pd.Series:
        return self.df[(pier, comp)]

    def peaks(self) -> pd.DataFrame:
        """Peak |kappa| and |eps0| per pier."""
        return self.df.abs().max().unstack(level=1)

    def save_pickle(self, path, *, compress: bool | None = None):
        return _save_pickle(self, path, compress=compress)

    @classmethod
    def load_pickle(cls, path) -> "CutCurvatureResult":
        obj = _load_pickle(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a CutCurvatureResult.")
        return obj

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<CutCurvatureResult: {len(self.piers)} pier(s), "
            f"{len(self.df)} steps, stage {self.model_stage!r}>"
        )


def _cluster_chords(
    intersections: list[ShellIntersection],
    gap_factor: float,
) -> dict[str, list[ShellIntersection]]:
    """Group chords into piers: direction family -> wall plane -> contiguity."""
    recs = []
    for ix in intersections:
        mid = ix.chord_midpoint
        chord = ix.chord_endpoints_arr
        d = chord[1] - chord[0]
        d = d / np.linalg.norm(d)
        fam = "X" if abs(d[0]) >= abs(d[1]) else "Y"
        plane_c = mid[1] if fam == "X" else mid[0]
        s = mid[0] if fam == "X" else mid[1]
        recs.append((ix, fam, round(float(plane_c), 0), float(s),
                     float(ix.chord_length)))
    frame = pd.DataFrame(recs, columns=["ix", "fam", "plane", "s", "length"])

    piers: dict[str, list[ShellIntersection]] = {}
    for (fam, plane), grp in frame.groupby(["fam", "plane"]):
        g = grp.sort_values("s")
        brk = (g.s.diff() > gap_factor * g.length.median()).cumsum()
        for _, pg in g.groupby(brk):
            label = f"{fam}{plane:.0f}@{pg.s.mean():.0f}"
            piers[label] = list(pg.ix)
    return piers


def cut_curvature(
    dataset: "MPCODataSet",
    *,
    model_stage: str,
    plane: Plane | None = None,
    spec: SectionCutSpec | None = None,
    selection_set_name: str | None = None,
    selection_set_id: int | None = None,
    element_ids=None,
    gap_factor: float = 1.6,
) -> CutCurvatureResult:
    """Per-pier in-plane curvature histories on a section-cut plane.

    Calling forms mirror :meth:`MPCODataSet.section_cut`: pass a prebuilt
    ``spec``, or a ``plane`` plus element filters.
    """
    if spec is None:
        if plane is None:
            raise ValueError("Pass either spec or plane.")
        spec = SectionCutSpec(
            plane=plane,
            selection_set_name=selection_set_name,
            selection_set_id=selection_set_id,
            element_ids=tuple(element_ids) if element_ids is not None else None,
        )
    intersections = list(find_shell_intersections(dataset, spec))
    if not intersections:
        raise ValueError("No shells intersect the cut plane under this filter.")

    # sanity: local Y must be ~vertical for the plane-sections reading
    R0 = dataset.cdata.rotation_matrix(intersections[0].element_id)
    if abs(float(R0[2, 1])) < _VERTICAL_TOL:
        raise ValueError(
            "Element local Y is not vertical (|e2.z| = "
            f"{abs(float(R0[2, 1])):.3f} < {_VERTICAL_TOL}); the epsYY "
            "plane-sections reading does not apply. Rotate strains first."
        )

    piers = _cluster_chords(intersections, gap_factor)

    # batched reads per element type
    by_type: dict[str, list[int]] = {}
    for ix in intersections:
        by_type.setdefault(ix.element_type, []).append(ix.element_id)
    eyy: dict[int, np.ndarray] = {}
    time: np.ndarray | None = None
    for elem_type, eids in by_type.items():
        er = dataset.elements.get_element_results(
            results_name="section.deformation",
            element_type=elem_type,
            model_stage=model_stage,
            element_ids=eids,
        )
        n_ip = int(er.n_ip)
        cols = [f"epsYY_ip{k}" for k in range(n_ip)]
        for eid in eids:
            rows = er.df.xs(eid, level="element_id")
            missing = [c for c in cols if c not in rows.columns]
            if missing:
                raise KeyError(
                    f"section.deformation for element {eid} lacks columns "
                    f"{missing}; found {list(rows.columns)[:6]}..."
                )
            eyy[eid] = rows[cols].to_numpy(dtype=float).mean(axis=1)
        if time is None:
            time = np.asarray(er.time, dtype=float)

    out: dict[tuple, pd.Series] = {}
    pier_rows = []
    for label, ixs in piers.items():
        s = np.array([
            ix.chord_midpoint[0] if label.startswith("X") else ix.chord_midpoint[1]
            for ix in ixs
        ])
        w = np.array([float(ix.chord_length) for ix in ixs])
        s0 = float((s * w).sum() / w.sum())
        ds_ = s - s0
        denom = float((w * ds_**2).sum())
        E = np.column_stack([eyy[ix.element_id] for ix in ixs])
        out[(label, "kappa")] = pd.Series(E @ (w * ds_) / denom)
        out[(label, "eps0")] = pd.Series(E @ (w / w.sum()))
        pier_rows.append(dict(
            pier=label, fam=label[0], n_elements=len(ixs),
            length=float(w.sum()), s_centroid=s0,
            element_ids=tuple(ix.element_id for ix in ixs),
        ))

    df = pd.DataFrame(out)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["pier", "comp"])
    return CutCurvatureResult(
        df=df, time=time, piers=pd.DataFrame(pier_rows).set_index("pier"),
        spec=spec, model_stage=model_stage,
    )
