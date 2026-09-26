"""Phase 0 smoke tests for the viewer subpackage.

Phase 0 establishes the namespace and optional extras only — no
rendering, no Qt. These tests guard the contract that
``import STKO_to_python.viewer`` stays lightweight (does not pull
``pyvista`` / ``vtk`` / ``PySide6`` / ``trame`` at import time).

The two extras-gated tests (``test_viewer_3d_extra_resolves``,
``test_viewer_qt_extra_resolves``) only run when the optional deps are
installed — they verify that the resolved environment is consistent
with what the extras advertise. In the base CI matrix they skip; in
the dedicated ``viewer-extras`` CI job they run. They skip only when a
package is not installed: an installed package that fails to import
(for example ``PySide6.QtGui`` without ``libEGL.so.1``) fails the test.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


HEAVY_DEPS = frozenset(
    {
        "pyvista",
        "vtk",
        "PySide6",
        "pyvistaqt",
        "qtpy",
        "trame",
        "trame_vtk",
        "trame_vuetify",
        "imageio",
        "imageio_ffmpeg",
    }
)


def _has(module: str) -> bool:
    """Return False only if ``module`` (or a parent package) is not installed.

    Any other ``ImportError`` propagates. Catching all of them turned a
    broken install into a skip: on the CI runner ``import pyvistaqt``
    raised ``libEGL.so.1: cannot open shared object file`` and the Qt
    test skipped as "viewer extra not installed" for months.
    """
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing and (module == missing or module.startswith(missing + ".")):
            return False
        raise
    return True


def test_has_skips_only_when_the_module_is_missing(tmp_path, monkeypatch) -> None:
    """A missing module is a skip; a module that fails to import is an error."""
    (tmp_path / "stko_smoke_broken.py").write_text(
        'raise ImportError("libEGL.so.1: cannot open shared object file")\n',
        encoding="utf-8",
    )
    (tmp_path / "stko_smoke_needs_dep.py").write_text(
        "import stko_smoke_absent_dependency\n", encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    assert _has("stko_smoke_not_installed") is False
    assert _has("stko_smoke_not_installed.sub") is False
    with pytest.raises(ImportError, match="libEGL"):
        _has("stko_smoke_broken")
    with pytest.raises(ModuleNotFoundError, match="stko_smoke_absent_dependency"):
        _has("stko_smoke_needs_dep")


def test_viewer_imports_without_extras() -> None:
    """The viewer namespace must import cleanly on the base install."""
    import STKO_to_python.viewer as viewer

    assert viewer.__all__ == []


def test_viewer_import_is_light() -> None:
    """Importing the viewer must not pull heavy optional deps at import time.

    This is the core Phase 0 contract: ``pip install stko_to_python``
    (no extras) does not gain ``pyvista`` / ``vtk`` / ``PySide6`` /
    ``trame`` transitively, so importing the viewer namespace in that
    environment must not trip an ``ImportError`` — it must succeed and
    not leak any of those top-level packages into ``sys.modules``.
    """
    # Import in a fresh interpreter, so other test modules' imports can't
    # hide a leak. Purging ``sys.modules`` in this process instead left
    # PySide6 half-unloaded (shiboken6 stayed), and the next
    # ``import pyvistaqt`` re-ran qtpy's patching against already-patched
    # PySide6 classes and failed.
    import STKO_to_python

    src = str(Path(STKO_to_python.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (src, env.get("PYTHONPATH")) if p
    )
    code = (
        "import json, sys\n"
        "import STKO_to_python.viewer\n"
        "print(json.dumps({'file': STKO_to_python.__file__,\n"
        "                  'modules': sorted({n.split('.')[0] for n in sys.modules})}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, encoding="utf-8", env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert Path(out["file"]).resolve() == Path(STKO_to_python.__file__).resolve()

    leaked = set(out["modules"]) & HEAVY_DEPS
    assert not leaked, f"viewer import pulled heavy deps: {sorted(leaked)}"


def test_schema_versions_present() -> None:
    """Schema version constants exist even though no spec format does yet."""
    from STKO_to_python.viewer import _version

    assert isinstance(_version.SCENE_SPEC_SCHEMA, int)
    assert isinstance(_version.SESSION_SCHEMA, int)
    # Phase 0 values: ``0`` means "no on-disk format yet". When Phase 2
    # lands the ``SceneSpec`` format, ``SCENE_SPEC_SCHEMA`` moves to 1
    # and this assertion will need bumping.
    assert _version.SCENE_SPEC_SCHEMA == 0
    assert _version.SESSION_SCHEMA == 0


def _skip_extra(reason: str) -> None:
    """Skip, unless ``STKO_REQUIRE_VIEWER_EXTRAS`` is set: then fail.

    The CI job that installs the extras sets it, so an extra that stops
    resolving turns the job red instead of skipping quietly.
    """
    if os.environ.get("STKO_REQUIRE_VIEWER_EXTRAS"):
        pytest.fail(f"{reason}, but STKO_REQUIRE_VIEWER_EXTRAS is set")
    pytest.skip(reason)


# The skip checks run inside the tests, not in ``skipif``: an import error
# raised while collecting would abort the whole pytest session.
def test_viewer_3d_extra_resolves() -> None:
    """When ``[viewer-3d]`` is installed, pyvista + vtk import successfully."""
    if not (_has("pyvista") and _has("vtk")):
        _skip_extra("viewer-3d extra not installed")
    import pyvista  # noqa: F401
    import vtk  # noqa: F401


def test_viewer_qt_extra_resolves() -> None:
    """When ``[viewer]`` is installed, the Qt stack imports successfully."""
    if not (_has("PySide6") and _has("pyvistaqt") and _has("qtpy")):
        _skip_extra("viewer extra not installed")
    import PySide6  # noqa: F401
    import pyvistaqt  # noqa: F401
    import qtpy  # noqa: F401
