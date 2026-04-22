"""Smoke test for ``examples/usage_tour.py``.

Executes the full tour against the checked-in elasticFrame fixture.
Guards against the scenario where the library's public API drifts and
silently makes the tour wrong — which would otherwise be caught only
by users running the examples manually.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOUR_PATH = REPO_ROOT / "examples" / "usage_tour.py"


def _load_tour_module():
    spec = importlib.util.spec_from_file_location("usage_tour", TOUR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["usage_tour"] = module
    spec.loader.exec_module(module)
    return module


def test_usage_tour_runs_end_to_end(elastic_frame_dir: Path, capsys):
    """Running the tour against elasticFrame should complete without
    raising, print the expected bookend messages, and leave matplotlib
    state clean."""
    tour = _load_tour_module()
    tour.run(elastic_frame_dir, "results")

    captured = capsys.readouterr()
    assert "Dataset introspection" in captured.out
    assert "ElementResults broker API" in captured.out
    assert "Tour complete." in captured.out


def test_usage_tour_script_path_exists():
    assert TOUR_PATH.is_file(), (
        f"examples/usage_tour.py should exist at {TOUR_PATH}"
    )
