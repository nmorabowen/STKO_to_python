"""Section-cut subpackage.

Section cuts integrate internal forces over a plane sliced through a
discretized FE model and recover (F, M) resultants. The public surface
is built up in stages — this ``__init__`` re-exports the names that are
ready for users to consume.

v1 surface (incremental):
    Plane                – geometric primitive (this module)
    SectionCutSpec       – picklable cut specification (pending)
    DriftSpec            – picklable node-pair drift specification (pending)
    SectionCut           – dataset-bound cut with results + .plot (pending)
    SectionSweep         – grid of planes against one dataset (pending)
    MultiCutResult       – wrapper for MPCOResults.section_cut(spec) (pending)
"""
from __future__ import annotations

from .curvature import CutCurvatureResult, cut_curvature
from .multi_cut import MultiCutResult
from .plane import Plane
from .section_cut import SectionCut
from .specs import DriftSpec, SectionCutSpec
from .sweep import SectionSweep

__all__ = [
    "Plane",
    "SectionCut",
    "SectionCutSpec",
    "SectionSweep",
    "MultiCutResult",
    "DriftSpec",
    "cut_curvature",
    "CutCurvatureResult",
]
