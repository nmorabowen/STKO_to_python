"""Shared pytest fixtures for STKO_to_python tests.

Fixture-independent tests (unit tests mocking h5py, logger behavior,
import contracts, pickle round-trips against checked-in pickle strings)
live under ``tests/unit/``. Golden tests that exercise a real .mpco file
look for a fixture at ``tests/fixtures/golden.mpco`` and skip gracefully
when absent — see the ``mpco_fixture_path`` fixture below.
"""
from __future__ import annotations

from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"
GOLDEN_MPCO_NAME = "golden.mpco"

# Real-world examples checked in under stko_results_examples/. These are
# the canonical integration-test inputs; see memory/project_examples_folder.md.
REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "stko_results_examples"
ELASTIC_FRAME_DIR = EXAMPLES_DIR / "elasticFrame" / "results"
QUAD_FRAME_DIR = EXAMPLES_DIR / "elasticFrame" / "QuadFrame_results"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Absolute path to the tests/fixtures/ directory."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def mpco_fixture_path(fixtures_dir: Path) -> Path:
    """Path to the golden .mpco fixture; skip test if absent.

    The fixture is intentionally not checked in yet. Tests marked with
    @pytest.mark.fixture that depend on it should request this fixture;
    they will be auto-skipped until the file lands.
    """
    path = fixtures_dir / GOLDEN_MPCO_NAME
    if not path.exists():
        pytest.skip(f"Golden .mpco fixture not available at {path}")
    return path


@pytest.fixture(scope="session")
def elastic_frame_dir() -> Path:
    """Directory for the single-partition ``elasticFrame`` example.

    Skips the test if the directory is absent — keeps the test suite
    green on environments without the examples checked out.
    """
    if not (ELASTIC_FRAME_DIR / "results.mpco").exists():
        pytest.skip(f"elasticFrame example not available at {ELASTIC_FRAME_DIR}")
    return ELASTIC_FRAME_DIR


@pytest.fixture(scope="session")
def quad_frame_dir() -> Path:
    """Directory for the multi-partition ``QuadFrame`` MP example."""
    if not (QUAD_FRAME_DIR / "results.part-0.mpco").exists():
        pytest.skip(f"QuadFrame MP example not available at {QUAD_FRAME_DIR}")
    return QUAD_FRAME_DIR
