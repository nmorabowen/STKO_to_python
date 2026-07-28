"""Section-cut resultants for wall-type structures.

Turns Gauss-point section forces recorded by the MPCO recorder into member-level
resultant time histories via horizontal section cuts, independently of the
element technology beneath the cut (validated on ASDShellQ4 walls and
ForceBeamColumn3d columns).

Method
------
Element local frames are reconstructed from the *assigned* local axes: STKO
writes an explicit ``-local vx vy vz`` on every ASDShellQ4 it generates, and
this module parses those vectors from the generated ``elements.tcl`` rather
than re-deriving the element default. Per element:

    e3 = normalize(cross(P3 - P1, P4 - P2))     # shell normal (diagonals)
    e1 = normalize(project(local_vec, plane))    # assigned local X, in-plane
    e2 = cross(e3, e1)                           # in-plane, ~vertical for walls

with (e2, e3) flipped together when e2 points downward, so the cut normal is
always the +Z hemisphere. The traction per unit cut length on a horizontal cut
(outward normal e2) is assembled as a *vector*:

    t = Fxy * e1 + Fyy * e2 + Vyz * e3

which makes every sign automatic. Pier resultants are integrals of t (and its
first moment) over the elements of the cut band, grouped into piers by
clustering: direction family -> wall-plane coordinate -> contiguity along the
wall.

Verification (San Ramon campaign, 2026-07-28): grade-cut wall shear vs
story-inertia shear, corr = 0.995 on a rigid-diaphragm elastic model (1A);
walls+columns closure corr 0.989 / peak ratio 0.93 (X) on the full nonlinear
DRM model (4D).
"""

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cut_results import CutResultants

# ASDShellQ4 section.force layout: per Gauss point, 8 components in this order.
SHELL_FORCE_COMPONENTS = ("Fxx", "Fyy", "Fxy", "Mxx", "Myy", "Mxy", "Vxz", "Vyz")

_LOCAL_RE = re.compile(
    r"element\s+ASDShellQ4\s+(\d+)\s+.*-local\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)"
)
_TRANSF_RE = re.compile(
    r"geomTransf\s+\w+\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)"
)
_BEAM_RE = re.compile(r"element\s+forceBeamColumn\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")


@dataclass
class CutBand:
    """Elements of one horizontal cut, with their frames and pier labels."""

    name: str
    z_range: tuple
    frame: pd.DataFrame  # per element: pier, fam, plane, s, w, e1/e2/e3 vectors

    @property
    def element_ids(self):
        return self.frame.element_id.tolist()

    @property
    def piers(self):
        return sorted(self.frame.pier.unique())


class SectionCuts:
    """Horizontal section-cut resultants on an :class:`MPCODataSet`.

    Parameters
    ----------
    dataset : MPCODataSet
        An opened dataset whose element recorder carries ``section.force``
        (and optionally ``section.deformation``) for the wall shells.
    tcl_path : str, optional
        Path to the generated ``elements.tcl``. If omitted, the file is
        located inside the dataset's results directory.
    """

    def __init__(self, dataset, tcl_path: str | None = None):
        self.ds = dataset
        directory = getattr(dataset, "hd5f_directory", None) or getattr(
            dataset, "hdf5_directory", None
        )
        if tcl_path is None:
            candidates = glob.glob(os.path.join(str(directory), "elements.tcl"))
            if not candidates:
                raise FileNotFoundError(
                    f"elements.tcl not found in {directory}; pass tcl_path explicitly."
                )
            tcl_path = candidates[0]
        self.tcl_path = tcl_path
        self._local_vectors: dict[int, np.ndarray] | None = None
        self._transf: dict[int, np.ndarray] | None = None
        self._beam_nodes: dict[int, tuple] | None = None

    # ------------------------------------------------------------------ tcl
    def _parse_tcl(self) -> None:
        loc, transf, beams = {}, {}, {}
        with open(self.tcl_path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                m = _LOCAL_RE.match(line)
                if m:
                    loc[int(m.group(1))] = np.array(
                        [float(m.group(2)), float(m.group(3)), float(m.group(4))]
                    )
                    continue
                m = _TRANSF_RE.match(line)
                if m:
                    transf[int(m.group(1))] = np.array(
                        [float(m.group(2)), float(m.group(3)), float(m.group(4))]
                    )
                    continue
                m = _BEAM_RE.match(line)
                if m:
                    beams[int(m.group(1))] = (int(m.group(2)), int(m.group(3)),
                                              int(m.group(4)))
        self._local_vectors, self._transf, self._beam_nodes = loc, transf, beams

    @property
    def local_vectors(self) -> dict:
        """``{element_id: assigned -local vector}`` parsed from elements.tcl."""
        if self._local_vectors is None:
            self._parse_tcl()
        return self._local_vectors

    # ---------------------------------------------------------------- frames
    def _node_coords(self) -> pd.DataFrame:
        return self.ds.nodes_info["dataframe"].set_index("node_id")[["x", "y", "z"]]

    def horizontal_band(
        self,
        z_min: float,
        z_max: float,
        *,
        name: str | None = None,
        selection_set_id: int | None = None,
        selection_set_name: str | None = None,
        element_ids=None,
        gap_factor: float = 1.6,
    ) -> CutBand:
        """Build a cut band: wall shells with centroid z in (z_min, z_max).

        Elements are restricted by ONE of ``selection_set_id`` /
        ``selection_set_name`` / ``element_ids`` (or none for all shells with a
        parsed ``-local`` vector), and clustered into piers.
        """
        ei = self.ds.elements_info["dataframe"]
        keep = ei.element_id.isin(self.local_vectors.keys())
        if selection_set_name is not None:
            sel = {
                s["SET_NAME"].lower(): sid for sid, s in self.ds.selection_set.items()
            }
            selection_set_id = sel[selection_set_name.lower()]
        if selection_set_id is not None:
            ids = set(self.ds.selection_set[selection_set_id]["ELEMENTS"])
            keep &= ei.element_id.isin(ids)
        if element_ids is not None:
            keep &= ei.element_id.isin(set(element_ids))
        band = ei[keep & (ei.centroid_z > z_min) & (ei.centroid_z < z_max)]
        if band.empty:
            raise ValueError(f"No wall shells found in z=({z_min}, {z_max}).")

        coords = self._node_coords()
        rows = []
        for _, r in band.iterrows():
            P = coords.loc[list(r.node_list)].to_numpy()
            e3 = np.cross(P[2] - P[0], P[3] - P[1])
            e3 = e3 / np.linalg.norm(e3)
            lx = self.local_vectors[r.element_id]
            e1 = lx - (lx @ e3) * e3
            e1 = e1 / np.linalg.norm(e1)
            e2 = np.cross(e3, e1)
            if e2[2] < 0:  # enforce cut normal (v-hat) upward; flip pair
                e2, e3 = -e2, -e3
            w = float((P @ e1).max() - (P @ e1).min())
            fam = "X" if abs(e3[1]) > abs(e3[0]) else "Y"
            plane = r.centroid_y if fam == "X" else r.centroid_x
            s = r.centroid_x if fam == "X" else r.centroid_y
            rows.append(
                dict(element_id=r.element_id, fam=fam, plane=float(plane),
                     s=float(s), w=w, z=float(r.centroid_z),
                     e1=e1, e2=e2, e3=e3)
            )
        frame = pd.DataFrame(rows)
        frame = self._cluster_piers(frame, gap_factor=gap_factor)
        return CutBand(name=name or f"z{z_min:g}-{z_max:g}",
                       z_range=(z_min, z_max), frame=frame)

    @staticmethod
    def _cluster_piers(frame: pd.DataFrame, gap_factor: float = 1.6) -> pd.DataFrame:
        parts = []
        for (fam, plane), grp in frame.groupby(["fam", frame.plane.round(0)]):
            g = grp.sort_values("s")
            brk = (g.s.diff() > gap_factor * g.w.median()).cumsum()
            for _, pg in g.groupby(brk):
                parts.append(pg.assign(pier=f"{fam}{plane:.0f}@{pg.s.mean():.0f}"))
        return pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------ resultants
    def resultants(
        self,
        band: CutBand,
        model_stage: str,
        *,
        element_type: str = "203-ASDShellQ4",
        n_gauss: int = 4,
    ) -> CutResultants:
        """Per-pier resultant time histories for one cut band.

        Components per pier (wall frame, v-hat = up, s-hat = along the wall):
        ``N`` axial, ``V`` in-plane shear, ``Vn`` out-of-plane shear, and
        ``M`` in-plane moment about the pier centroid (first moment of the
        axial traction).
        """
        df = self.ds.elements.get_element_results(
            results_name="section.force", model_stage=model_stage,
            element_type=element_type, element_ids=band.element_ids,
        ).df

        def gauss_mean(comp: str) -> pd.Series:
            i = SHELL_FORCE_COMPONENTS.index(comp)
            cols = [f"val_{g * len(SHELL_FORCE_COMPONENTS) + i + 1}"
                    for g in range(n_gauss)]
            return df[cols].mean(axis=1)

        fyy, fxy, vyz = gauss_mean("Fyy"), gauss_mean("Fxy"), gauss_mean("Vyz")

        out: dict[tuple, pd.Series] = {}
        for pier, grp in band.frame.groupby("pier"):
            s0 = float((grp.s * grp.w).sum() / grp.w.sum())
            fam = grp.fam.iloc[0]
            s_hat = np.array([1.0, 0.0, 0.0]) if fam == "X" else np.array([0.0, 1.0, 0.0])
            n_hat = np.array([0.0, 1.0, 0.0]) if fam == "X" else np.array([1.0, 0.0, 0.0])
            N = V = Vn = M = 0.0
            for _, e in grp.iterrows():
                # traction vector per unit length on the horizontal cut
                tx = fxy.loc[e.element_id] * e.e1[0] + fyy.loc[e.element_id] * e.e2[0] \
                    + vyz.loc[e.element_id] * e.e3[0]
                ty = fxy.loc[e.element_id] * e.e1[1] + fyy.loc[e.element_id] * e.e2[1] \
                    + vyz.loc[e.element_id] * e.e3[1]
                tz = fxy.loc[e.element_id] * e.e1[2] + fyy.loc[e.element_id] * e.e2[2] \
                    + vyz.loc[e.element_id] * e.e3[2]
                N = N + tz * e.w
                V = V + (tx * s_hat[0] + ty * s_hat[1]) * e.w
                Vn = Vn + (tx * n_hat[0] + ty * n_hat[1]) * e.w
                M = M + tz * e.w * (e.s - s0)
            out[(band.name, pier, "N")] = N
            out[(band.name, pier, "V")] = V
            out[(band.name, pier, "Vn")] = Vn
            out[(band.name, pier, "M")] = M

        res = pd.DataFrame(out)
        res.columns = pd.MultiIndex.from_tuples(res.columns,
                                                names=["cut", "pier", "comp"])
        time = self.ds.time.loc[model_stage]["TIME"].to_numpy()
        info = band.frame.drop(columns=["e1", "e2", "e3"])
        return CutResultants(df=res, time=time, info=info,
                             name=getattr(self.ds, "name", "") or "")

    # ------------------------------------------------------------ beam shear
    def beam_shear(
        self,
        element_ids,
        model_stage: str,
        *,
        element_type: str = "74-ForceBeamColumn3d",
        n_sections: int = 5,
        n_comps: int = 4,
    ) -> pd.DataFrame:
        """Global shear vector histories for beam-columns from moment gradients.

        FBC ``section.force`` carries (P, Mz, My, T) only; the shear is the
        moment gradient between the end Lobatto sections (exact absent member
        loads). Mapping to global axes uses the geomTransf ``vecxz`` parsed
        from elements.tcl:

            x_loc = (node_j - node_i) / L
            y_loc = normalize(cross(vecxz, x_loc))
            z_loc = cross(x_loc, y_loc)
            V_vec = (dMy/dx) * z_loc - (dMz/dx) * y_loc

        The sign convention was validated by story-shear closure against
        inertia (San Ramon 4D, 2026-07-28); re-derive symbolically before
        reusing outside vertical members.

        Returns a DataFrame indexed by step with MultiIndex columns
        ``(element_id, 'Vx'|'Vy'|'Vz')``.
        """
        if self._beam_nodes is None:
            self._parse_tcl()
        coords = self._node_coords()
        df = self.ds.elements.get_element_results(
            results_name="section.force", model_stage=model_stage,
            element_type=element_type, element_ids=list(element_ids),
        ).df
        i_mz, i_my = 1, 2  # (P, Mz, My, T)
        first, last = 0, n_sections - 1
        out = {}
        for eid in element_ids:
            ni, nj, transf_tag = self._beam_nodes[eid]
            Pi, Pj = coords.loc[ni].to_numpy(), coords.loc[nj].to_numpy()
            L = float(np.linalg.norm(Pj - Pi))
            x_loc = (Pj - Pi) / L
            vecxz = self._transf[transf_tag]
            y_loc = np.cross(vecxz, x_loc)
            y_loc = y_loc / np.linalg.norm(y_loc)
            z_loc = np.cross(x_loc, y_loc)
            d = df.loc[eid]
            dMy = (d[f"val_{last * n_comps + i_my + 1}"]
                   - d[f"val_{first * n_comps + i_my + 1}"]) / L
            dMz = (d[f"val_{last * n_comps + i_mz + 1}"]
                   - d[f"val_{first * n_comps + i_mz + 1}"]) / L
            for k, ax in enumerate(("Vx", "Vy", "Vz")):
                out[(eid, ax)] = dMy * z_loc[k] - dMz * y_loc[k]
        res = pd.DataFrame(out)
        res.columns = pd.MultiIndex.from_tuples(res.columns,
                                                names=["element_id", "comp"])
        return res
