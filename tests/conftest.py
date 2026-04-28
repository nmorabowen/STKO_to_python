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


def _resolve_examples_dir(repo_root: Path) -> Path:
    """Return the ``stko_results_examples`` dir to read fixtures from.

    Worktrees spawned under ``<main>/.claude/worktrees/<name>/`` rarely
    carry the ~2 GB of example fixtures — only ``elasticFrame`` is small
    enough to be checked in. To keep tests in worktrees from silently
    skipping coverage, we fall back to the main checkout's copy when
    the worktree-local one is missing the heavier fixtures.

    Resolution order:

    1. ``<repo_root>/stko_results_examples`` if it has the heavier
       fixtures (``Test_NLShell`` or ``solid_partition_example``).
    2. The main checkout derived by stripping ``.claude/worktrees/<name>``
       from ``repo_root``, if that path has the examples.
    3. Whatever ``<repo_root>/stko_results_examples`` is — individual
       fixtures will skip per the existing ``.exists()`` checks.
    """
    local = repo_root / "stko_results_examples"
    has_heavy = (local / "Test_NLShell").exists() or (
        local / "solid_partition_example"
    ).exists()
    if has_heavy:
        return local

    parts = repo_root.parts
    try:
        idx = parts.index(".claude")
    except ValueError:
        return local
    if idx + 2 < len(parts) and parts[idx + 1] == "worktrees":
        main_root = Path(*parts[:idx])
        candidate = main_root / "stko_results_examples"
        if candidate.exists():
            return candidate
    return local


EXAMPLES_DIR = _resolve_examples_dir(REPO_ROOT)
ELASTIC_FRAME_DIR = EXAMPLES_DIR / "elasticFrame" / "results"
QUAD_FRAME_DIR = EXAMPLES_DIR / "elasticFrame" / "QuadFrame_results"
SOLID_PARTITION_DIR = EXAMPLES_DIR / "solid_partition_example"
NL_SHELL_DIR = EXAMPLES_DIR / "Test_NLShell"


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


@pytest.fixture(scope="session")
def solid_partition_dir() -> Path:
    """Two-partition fixture combining a Brick continuum and a
    DispBeamColumn3d (with ``section.fiber.stress`` compressed META).

    Exercises every META shape codified in
    ``docs/mpco_format_conventions.md`` — closed-form (beam + brick),
    line-stations (section.force/section.deformation), Gauss-level
    continuum (material.stress/material.strain), and MULTIPLICITY > 1
    fiber compression. Skips the test if the fixture is absent.
    """
    if not (SOLID_PARTITION_DIR / "Recorder.part-0.mpco").exists():
        pytest.skip(
            f"solid_partition_example not available at {SOLID_PARTITION_DIR}"
        )
    return SOLID_PARTITION_DIR


@pytest.fixture(scope="session")
def nl_shell_dir() -> Path:
    """Four-partition fixture covering the layered-shell META pattern.

    Exercises:

    * ``204-ASDShellT3`` (3-IP triangular shell) alongside
      ``203-ASDShellQ4`` (4-IP quad shell) in the same model.
    * Layered ``section.fiber.*`` buckets where a single bucket carries
      ``(gauss_point × thickness_layer)`` blocks — repeated GAUSS_IDS,
      mixed MULTIPLICITY across blocks, and empty COMPONENTS segments
      for layers that don't track the requested quantity.

    Skips the test if the fixture is absent (it's developer-local —
    ~2 GB of recorder output, gitignored).
    """
    if not (NL_SHELL_DIR / "Results.part-0.mpco").exists():
        pytest.skip(f"Test_NLShell not available at {NL_SHELL_DIR}")
    return NL_SHELL_DIR
