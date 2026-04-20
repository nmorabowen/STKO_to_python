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
