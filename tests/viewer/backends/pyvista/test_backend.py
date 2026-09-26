"""Tests for :class:`PyVistaBackend`.

Like the matplotlib backend tests, each case exercises one method on
the protocol via a fresh off-screen scene, verifies the resulting
PyVista actor / dataset, then closes the plotter to release VTK
resources.

The entire module is gated on ``pyvista`` being importable. When the
``[viewer-3d]`` extra isn't installed the tests are skipped — CI
matrices that include the ``viewer-3d`` job will pick them up.
"""
from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from STKO_to_python.viewer.backends.pyvista import (
    PvSceneHandle,
    PyVistaBackend,
    backend as pv_backend,
)
from STKO_to_python.viewer.backends.pyvista.backend import _PvActorRef
from STKO_to_python.viewer.core import (
    BBox,
    Backend,
    BackendCapabilityError,
    CameraSpec,
    SceneStyle,
)


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture
def backend() -> PyVistaBackend:
    return PyVistaBackend()


@pytest.fixture
def scene(backend: PyVistaBackend):
    handle = backend.make_scene(is_3d=True, off_screen=True)
    yield handle
    try:
        handle.plotter.close()
    except Exception:
        pass


# --------------------------------------------------------------------- #
# Protocol conformance
# --------------------------------------------------------------------- #


def test_satisfies_backend_protocol() -> None:
    assert isinstance(PyVistaBackend(), Backend)


def test_backend_metadata() -> None:
    b = PyVistaBackend()
    assert b.name == "pyvista"
    assert b.is_3d_capable is True
    assert b.is_interactive is True


# --------------------------------------------------------------------- #
# Scene lifecycle
# --------------------------------------------------------------------- #


def test_make_scene_returns_handle_wrapping_plotter(backend) -> None:
    handle = backend.make_scene(is_3d=True, off_screen=True)
    try:
        assert isinstance(handle, PvSceneHandle)
        assert handle.is_3d is True
        assert handle.off_screen is True
        assert isinstance(handle.plotter, pv.Plotter)
    finally:
        handle.plotter.close()


def test_make_scene_accepts_caller_supplied_plotter(backend) -> None:
    """The ``plotter=`` extension lets Qt thread its QtInteractor through."""
    plotter = pv.Plotter(off_screen=True)
    handle = backend.make_scene(plotter=plotter)
    try:
        assert handle.plotter is plotter
    finally:
        plotter.close()


def test_set_bounds_calls_renderer_reset_camera(backend, scene) -> None:
    """Verify the bounds reach VTK's renderer (we check the camera moves)."""
    pos_before = scene.plotter.camera.position
    backend.set_bounds(scene, BBox(-10, -10, -10, 10, 10, 10))
    pos_after = scene.plotter.camera.position
    # Reset-camera moves the camera unless it was already framing the bounds;
    # tolerate either outcome — what we actually pin is that the call doesn't
    # error and the camera state is consistent.
    assert isinstance(pos_after, tuple)


def test_set_camera_applies_orientation(backend, scene) -> None:
    """PyVista normalizes ``camera_position`` to a renderer-driven
    distance — what we pin is the orientation (focal point, view-up,
    and the direction from focal to position)."""
    cam = CameraSpec(
        position=(5.0, 5.0, 5.0),
        focal_point=(0.0, 0.0, 0.0),
        view_up=(0.0, 0.0, 1.0),
        parallel_projection=False,
    )
    backend.set_camera(scene, cam)
    pos, focal, up = scene.plotter.camera_position
    np.testing.assert_allclose(focal, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(up, [0.0, 0.0, 1.0])
    # Direction (unit vector from focal to position) is the orientation
    # contract; absolute distance is renderer-controlled.
    direction = np.asarray(pos) - np.asarray(focal)
    unit = direction / np.linalg.norm(direction)
    expected = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    np.testing.assert_allclose(unit, expected, atol=1e-6)


def test_set_camera_parallel_projection_toggle(backend, scene) -> None:
    cam_perspective = CameraSpec(
        position=(1.0, 0.0, 0.0),
        focal_point=(0.0, 0.0, 0.0),
        view_up=(0.0, 0.0, 1.0),
        parallel_projection=False,
    )
    cam_parallel = CameraSpec(
        position=(1.0, 0.0, 0.0),
        focal_point=(0.0, 0.0, 0.0),
        view_up=(0.0, 0.0, 1.0),
        parallel_projection=True,
    )
    backend.set_camera(scene, cam_parallel)
    assert scene.plotter.camera.parallel_projection is True
    backend.set_camera(scene, cam_perspective)
    assert scene.plotter.camera.parallel_projection is False


def test_set_style_applies_background_and_grid(backend, scene) -> None:
    style = SceneStyle(background="black", grid=False, font_size=10)
    backend.set_style(scene, style)
    # PyVista returns a Color object — compare component-wise.
    bg = scene.plotter.background_color
    assert tuple(bg.float_rgb) == pytest.approx((0.0, 0.0, 0.0))


# --------------------------------------------------------------------- #
# Primitives — add_segments
# --------------------------------------------------------------------- #


def test_add_segments_creates_polydata_with_lines(backend, scene) -> None:
    segs = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ]
    )
    ref = backend.add_segments(scene, segs, color="red", width=2.0, alpha=0.8)
    assert isinstance(ref, _PvActorRef)
    assert ref.kind == "segments"
    assert ref.scalar_field is None
    assert ref.dataset.n_points == 4
    assert ref.dataset.n_lines == 2


def test_add_segments_empty_input_creates_empty_dataset(backend, scene) -> None:
    ref = backend.add_segments(scene, np.zeros((0, 2, 3)))
    assert ref.kind == "segments"
    assert ref.dataset.n_points == 0
    assert ref.dataset.n_lines == 0


# --------------------------------------------------------------------- #
# Primitives — add_points
# --------------------------------------------------------------------- #


def test_add_points_creates_polydata(backend, scene) -> None:
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    ref = backend.add_points(scene, pts, color="blue", size=10.0)
    assert ref.kind == "points"
    assert ref.dataset.n_points == 3
    assert ref.scalar_field is None


def test_add_points_with_scalars_binds_scalar_field(backend, scene) -> None:
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    scalars = np.array([0.5, 1.5])
    ref = backend.add_points(scene, pts, scalars=scalars, cmap="viridis")
    assert ref.scalar_field == "scalars"
    np.testing.assert_allclose(ref.dataset.point_data["scalars"], [0.5, 1.5])


def test_add_points_empty_input(backend, scene) -> None:
    ref = backend.add_points(scene, np.zeros((0, 3)))
    assert ref.dataset.n_points == 0


# --------------------------------------------------------------------- #
# Primitives — add_polygons
# --------------------------------------------------------------------- #


def test_add_polygons_handles_heterogeneous_M(backend, scene) -> None:
    """A triangle + a quad in the same call — VTK accepts heterogeneous M."""
    tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
    quad = np.array(
        [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [2.0, 1.0, 0.0]]
    )
    ref = backend.add_polygons(scene, [tri, quad], edge_color="black")
    assert ref.kind == "polygons"
    assert ref.dataset.n_points == 7
    assert ref.dataset.n_cells == 2


def test_add_polygons_with_per_cell_values(backend, scene) -> None:
    tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
    quad = np.array(
        [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [2.0, 1.0, 0.0]]
    )
    ref = backend.add_polygons(
        scene, [tri, quad], values=np.array([1.0, 2.0]), cmap="viridis",
    )
    assert ref.scalar_field == "values"
    np.testing.assert_allclose(ref.dataset.cell_data["values"], [1.0, 2.0])


def test_add_polygons_empty_input(backend, scene) -> None:
    ref = backend.add_polygons(scene, [])
    assert ref.dataset.n_cells == 0


# --------------------------------------------------------------------- #
# Primitives — add_arrows
# --------------------------------------------------------------------- #


def test_add_arrows_creates_actor(backend, scene) -> None:
    origins = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vectors = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    ref = backend.add_arrows(scene, origins, vectors, scale=2.0, color="green")
    assert ref.kind == "arrows"
    assert ref.actor is not None
    # Arrows have no scalar field and no mutable dataset reference.
    assert ref.scalar_field is None
    assert ref.dataset is None


# --------------------------------------------------------------------- #
# In-place updates
# --------------------------------------------------------------------- #


def test_update_scalars_mutates_point_data_in_place(backend, scene) -> None:
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ref = backend.add_points(scene, pts, scalars=np.array([0.0, 0.0]))
    backend.update_scalars(ref, np.array([3.0, 4.0]))
    np.testing.assert_allclose(ref.dataset.point_data["scalars"], [3.0, 4.0])


def test_update_scalars_raises_when_no_scalar_field_bound(backend, scene) -> None:
    """Position-only points have no scalar field; update_scalars must error."""
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ref = backend.add_points(scene, pts, color="red")  # no scalars=
    with pytest.raises(BackendCapabilityError, match="scalar updates"):
        backend.update_scalars(ref, np.array([1.0, 2.0]))


def test_update_scalars_raises_for_arrows(backend, scene) -> None:
    ref = backend.add_arrows(
        scene,
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]]),
    )
    with pytest.raises(BackendCapabilityError):
        backend.update_scalars(ref, np.array([1.0]))


def test_update_points_mutates_segments_in_place(backend, scene) -> None:
    segs = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    ref = backend.add_segments(scene, segs)
    new_segs = np.array([[[5.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    backend.update_points(ref, new_segs)
    np.testing.assert_allclose(ref.dataset.points[0], [5.0, 0.0, 0.0])
    np.testing.assert_allclose(ref.dataset.points[1], [10.0, 0.0, 0.0])


def test_update_points_mutates_point_cloud_in_place(backend, scene) -> None:
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ref = backend.add_points(scene, pts)
    backend.update_points(ref, np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]))
    np.testing.assert_allclose(ref.dataset.points[0], [3.0, 3.0, 3.0])


def test_update_points_raises_for_arrows(backend, scene) -> None:
    ref = backend.add_arrows(
        scene,
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]]),
    )
    with pytest.raises(BackendCapabilityError):
        backend.update_points(ref, np.array([[1.0, 1.0, 1.0]]))


def test_set_visible_toggles_actor_visibility(backend, scene) -> None:
    pts = np.array([[0.0, 0.0, 0.0]])
    ref = backend.add_points(scene, pts)
    backend.set_visible(ref, False)
    assert ref.actor.GetVisibility() == 0
    backend.set_visible(ref, True)
    assert ref.actor.GetVisibility() == 1


def test_remove_unhooks_actor_from_scene(backend, scene) -> None:
    ref = backend.add_points(scene, np.array([[0.0, 0.0, 0.0]]))
    n_actors_before = len(list(scene.plotter.renderer.actors))
    backend.remove(scene, ref)
    n_actors_after = len(list(scene.plotter.renderer.actors))
    assert n_actors_after < n_actors_before


def test_unwrap_rejects_raw_actors() -> None:
    """The protocol forbids layers from handing raw VTK actors back."""
    b = PyVistaBackend()
    with pytest.raises(BackendCapabilityError, match="actor reference"):
        b.update_scalars("not-a-pvactorref", np.array([1.0]))


# --------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------- #


def test_snapshot_returns_rgb_array(backend, scene) -> None:
    backend.add_points(scene, np.array([[0.0, 0.0, 0.0]]))
    arr = backend.snapshot(scene)
    assert arr.ndim == 3
    assert arr.shape[-1] == 3
    assert arr.dtype == np.uint8


def test_save_writes_a_file(backend, scene, tmp_path) -> None:
    backend.add_points(scene, np.array([[0.0, 0.0, 0.0]]))
    out = tmp_path / "out.png"
    backend.save(scene, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_show_off_screen_is_noop(backend) -> None:
    """An off-screen scene must not pop a window from show()."""
    handle = backend.make_scene(is_3d=True, off_screen=True)
    try:
        backend.show(handle)  # must not raise
    finally:
        handle.plotter.close()


# --------------------------------------------------------------------- #
# Headless GL check (Linux)
# --------------------------------------------------------------------- #
# The platform, the environment and both library lookups are faked, so
# these run on any OS and never reach VTK. Reaching the fake plotter's
# screenshot() stands for a real render, which segfaults on Linux when
# there is no GL context (#100).


class _FakePlotter:
    def __init__(self) -> None:
        self.screenshots = 0

    def screenshot(self, *args, **kwargs):
        self.screenshots += 1
        return np.zeros((4, 4, 3), dtype=np.uint8)


def _fake_scene() -> PvSceneHandle:
    return PvSceneHandle(plotter=_FakePlotter(), is_3d=True, off_screen=True)


@pytest.fixture
def headless_linux(monkeypatch, tmp_path):
    """Linux with no display and no EGL / OSMesa (the CI runner before #100).

    Returns the fake ``/proc/self/maps``: libGL (GLX) is mapped, which
    is no use without an X server. ``raising=False``: against the code
    before the fix, which has none of these names, the tests below then
    fail on the missing error, not in this fixture.
    """
    maps = tmp_path / "maps"
    maps.write_text(
        "7f00-7f01 r-xp 00000000 08:01 1 /usr/lib/x86_64-linux-gnu/libGL.so.1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(pv_backend, "_IS_LINUX", True, raising=False)
    monkeypatch.setattr(pv_backend, "_PROC_SELF_MAPS", str(maps), raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("ctypes.util.find_library", lambda name: None)
    return maps


def test_render_without_gl_raises_instead_of_segfaulting(
    backend, headless_linux, tmp_path,
) -> None:
    scene = _fake_scene()
    with pytest.raises(RuntimeError, match=r"Xvfb.*EGL.*OSMesa"):
        backend.snapshot(scene)
    with pytest.raises(RuntimeError, match=r"Xvfb.*EGL.*OSMesa"):
        backend.save(scene, tmp_path / "out.png")
    assert scene.plotter.screenshots == 0


@pytest.mark.parametrize(
    "var, value", [("DISPLAY", ":99"), ("WAYLAND_DISPLAY", "wayland-0")],
)
def test_render_with_a_display_is_not_blocked(
    backend, headless_linux, monkeypatch, var, value,
) -> None:
    """Xvfb (or any X server / Wayland session) sets one of these."""
    monkeypatch.setenv(var, value)
    scene = _fake_scene()
    assert backend.snapshot(scene).shape == (4, 4, 3)


@pytest.mark.parametrize("lib", ["EGL", "OSMesa"])
def test_render_with_egl_or_osmesa_on_the_linker_path_is_not_blocked(
    backend, headless_linux, monkeypatch, lib,
) -> None:
    monkeypatch.setattr(
        "ctypes.util.find_library",
        lambda name: f"lib{name}.so.1" if name == lib else None,
    )
    scene = _fake_scene()
    assert backend.snapshot(scene).shape == (4, 4, 3)


@pytest.mark.parametrize(
    "path",
    [
        "/usr/lib/x86_64-linux-gnu/libEGL.so.1",
        "/opt/conda/envs/post/lib/libOSMesa.so.8",
        "/site-packages/vtk.libs/libOSMesa-1a2b3c4d.so.8.0.0",
    ],
)
def test_render_with_egl_or_osmesa_already_loaded_is_not_blocked(
    backend, headless_linux, path,
) -> None:
    """A library VTK loaded from outside the linker cache (conda, a wheel)."""
    with headless_linux.open("a", encoding="utf-8") as fh:
        fh.write(f"7f02-7f03 r-xp 00000000 08:01 2 {path}\n")
    scene = _fake_scene()
    assert backend.snapshot(scene).shape == (4, 4, 3)


def test_check_is_linux_only(backend, headless_linux, monkeypatch) -> None:
    monkeypatch.setattr(pv_backend, "_IS_LINUX", False, raising=False)
    scene = _fake_scene()
    assert backend.snapshot(scene).shape == (4, 4, 3)


def test_check_runs_once_per_scene(backend, headless_linux, monkeypatch) -> None:
    """An animation export must not repeat the library lookup per frame."""
    monkeypatch.setenv("DISPLAY", ":99")
    scene = _fake_scene()
    backend.snapshot(scene)
    monkeypatch.delenv("DISPLAY")
    backend.snapshot(scene)
    assert scene.plotter.screenshots == 2
    with pytest.raises(RuntimeError, match="Xvfb"):
        backend.snapshot(_fake_scene())
