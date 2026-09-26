"""PyVista :class:`Backend` implementation.

Maps every primitive on the
:class:`STKO_to_python.viewer.core.Backend` protocol onto PyVista /
VTK's 3-D drawing surface. The scene handle wraps a
:class:`pyvista.Plotter` (windowed or off-screen) and the per-actor
state is a small :class:`_PvActorRef` record carrying the actor +
the underlying ``pyvista.DataSet`` + the bound scalar field name.

The dataset is held on the actor ref so :meth:`update_scalars` and
:meth:`update_points` mutate the same arrays in place — the
apeGmsh perf contract that keeps animation playback fast. Each
``Modified()`` invalidation flags only the changed actor; the rest
of the scene re-renders unchanged.

The Phase 3.0 release matches the
:class:`STKO_to_python.viewer.backends.mpl.MplBackend` API surface
1-to-1; concrete 3-D layer types
(``ContourLayer``, ``GaussLayer``, ``VolumeLayer``, …) ride on top
in Phase 3.1+ and gain capabilities (``add_clipped``, ``add_slice``,
``add_iso``) that PyVista handles natively but matplotlib cannot.

This module is **only** imported when the optional ``[viewer-3d]``
extra is installed. Layer / scene code reaches it indirectly through
``Scene.backend`` so the import is lazy from the consumer side too.
"""
from __future__ import annotations

import ctypes.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

try:
    import pyvista as pv
except ImportError as exc:  # pragma: no cover - exercised by the smoke test
    raise ImportError(
        "PyVistaBackend requires the [viewer-3d] extra. Install with "
        "'pip install \"stko_to_python[viewer-3d]\"' or "
        "'pip install pyvista>=0.43 vtk>=9.2'."
    ) from exc

from ...core.errors import BackendCapabilityError
from ...core.types import ActorRef, BBox, CameraSpec

if TYPE_CHECKING:
    from ...core.style import SceneStyle


# ---------------------------------------------------------------------- #
# Handles & actor records
# ---------------------------------------------------------------------- #


@dataclass
class PvSceneHandle:
    """Wraps a single :class:`pyvista.Plotter`.

    The plotter is allocated by :meth:`PyVistaBackend.make_scene`;
    one scene = one plotter. The ``is_3d`` flag is informational —
    PyVista is always 3-D, but the layer code may use the flag to
    short-circuit 2-D-only branches when the scene is intentionally
    flat.

    Attributes:
        plotter: The :class:`pyvista.Plotter` (or
            :class:`pyvistaqt.QtInteractor` in Phase 4).
        is_3d: Always ``True`` for PyVista today; kept on the handle
            so layers can introspect the projection the way they do
            on :class:`~STKO_to_python.viewer.backends.mpl.MplSceneHandle`.
        off_screen: When ``True``, :meth:`PyVistaBackend.show` is a
            no-op — used by the headless CLI and CI snapshots.
    """

    plotter: Any
    is_3d: bool
    off_screen: bool
    # Set once the headless-GL check has passed for this plotter, so an
    # animation export doesn't repeat the library lookup on every frame.
    _gl_checked: bool = field(default=False, init=False, repr=False, compare=False)


@dataclass
class _PvActorRef:
    """Opaque actor record returned to layers.

    Layers treat the ref as a cookie; the backend unwraps it on
    :meth:`update_*` / :meth:`remove` / :meth:`set_visible` to reach
    the actor and its underlying dataset. The dataset is held here
    so ``point_data`` / ``cell_data`` mutation can happen on the
    pre-allocated arrays.

    ``scalar_field`` records the name of the scalar array bound at
    actor-creation time (when the primitive received a ``scalars=``
    argument). :meth:`PyVistaBackend.update_scalars` writes into the
    same name.
    """

    actor: Any
    dataset: Any  # pyvista.DataSet | None
    scalar_field: str | None
    kind: str  # "segments" | "points" | "polygons" | "arrows"


# ---------------------------------------------------------------------- #
# Headless GL check (Linux)
# ---------------------------------------------------------------------- #

_IS_LINUX = sys.platform.startswith("linux")
_PROC_SELF_MAPS = "/proc/self/maps"
# What VTK falls back to on Linux when there is no X display.
_HEADLESS_GL_LIBS = ("EGL", "OSMesa")


def _headless_gl_library_present() -> bool:
    """Return True if an EGL or OSMesa library is available to VTK.

    Two cheap lookups; neither creates a GL context:

    - ``ctypes.util.find_library`` (the dynamic linker cache,
      ``LD_LIBRARY_PATH``);
    - the libraries already mapped into this process
      (``/proc/self/maps``). By the time a scene renders, VTK has
      already created its render window and loaded whichever library it
      found, including one outside the linker cache (a conda
      environment, a copy bundled in a wheel).
    """
    if any(ctypes.util.find_library(name) for name in _HEADLESS_GL_LIBS):
        return True
    try:
        with open(_PROC_SELF_MAPS, encoding="utf-8", errors="replace") as fh:
            maps = fh.read()
    except OSError:
        return False
    return any(f"/lib{name}" in maps for name in _HEADLESS_GL_LIBS)


def _missing_gl_context() -> bool:
    """Return True when VTK has no way to get an OpenGL context here.

    Linux only; always False on Windows and macOS. True when
    ``DISPLAY`` and ``WAYLAND_DISPLAY`` are both unset or empty AND no
    EGL or OSMesa library is present (:func:`_headless_gl_library_present`).
    In that setup VTK 9.7 falls through to an OSMesa window with no
    library behind it and segfaults on the first render (#100).

    It cannot detect a ``DISPLAY`` that points at no running X server,
    or an EGL / OSMesa library that is present but unusable (for example
    libglvnd's ``libEGL.so.1`` with no vendor driver): those still reach
    VTK. It would wrongly return True only if EGL or OSMesa were linked
    statically into VTK, with no separate library file.
    """
    if not _IS_LINUX:
        return False
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return False
    return not _headless_gl_library_present()


# ---------------------------------------------------------------------- #
# Backend implementation
# ---------------------------------------------------------------------- #


class PyVistaBackend:
    """Backend protocol satisfied via PyVista's 3-D primitives.

    Construction is free — the backend holds no scene state. Each
    :meth:`make_scene` call allocates a fresh :class:`pyvista.Plotter`
    wrapped in a :class:`PvSceneHandle`. Layers receive
    :class:`_PvActorRef` records as ``ActorRef`` cookies; subsequent
    ``update_*`` / ``set_visible`` / ``remove`` calls unwrap those
    to reach the actor and its underlying dataset.

    Off-screen rendering uses VTK's native off-screen path (no Qt, no
    window). On Linux, VTK still needs an OpenGL context from an X
    server (``xvfb-run`` provides one), EGL or OSMesa — see
    ``docs/viewer/03-deployment-targets.md`` §7. When none of them is
    available, :meth:`snapshot` and :meth:`save` raise ``RuntimeError``
    instead of letting VTK segfault.
    """

    name: str = "pyvista"
    is_3d_capable: bool = True
    is_interactive: bool = True

    # ------------------------------------------------------------------ #
    # Scene lifecycle
    # ------------------------------------------------------------------ #
    def make_scene(
        self,
        *,
        is_3d: bool = True,
        off_screen: bool = False,
        plotter: Any = None,
    ) -> PvSceneHandle:
        """Allocate (or borrow) a :class:`pyvista.Plotter` for the scene.

        The optional ``plotter`` argument is a backend-specific
        extension beyond the :class:`Backend` protocol — it lets the
        Phase 4 Qt UI thread its
        :class:`pyvistaqt.QtInteractor` instance through the scene
        machinery without the backend creating its own window.
        """
        if plotter is None:
            plotter = pv.Plotter(off_screen=off_screen)
        # Layers can legitimately add empty actors (e.g. a selection that
        # filters out every element of a class, or a layer whose result
        # field happens to be zero-length at attach). PyVista's default
        # theme raises on empty input; relax that locally so the
        # ``add_*`` primitives stay total over the input domain.
        try:
            plotter.theme.allow_empty_mesh = True
        except AttributeError:  # pragma: no cover - older pyvista
            pass
        return PvSceneHandle(plotter=plotter, is_3d=is_3d, off_screen=off_screen)

    def set_bounds(self, scene: PvSceneHandle, bbox: BBox) -> None:
        """Reframe the camera to span ``bbox``.

        Delegates to ``vtkRenderer.ResetCamera(bounds[6])`` — the
        same path PyVista's own ``reset_camera`` uses, but with
        caller-supplied bounds rather than the scene's auto-computed
        ones. Useful when a filter / selection layer hides part of
        the model but the camera should still frame the whole
        geometry.
        """
        bounds = (
            bbox.x_min, bbox.x_max,
            bbox.y_min, bbox.y_max,
            bbox.z_min, bbox.z_max,
        )
        scene.plotter.renderer.ResetCamera(bounds)

    def set_camera(self, scene: PvSceneHandle, cam: CameraSpec) -> None:
        """Apply a saved camera state."""
        scene.plotter.camera_position = (
            tuple(cam.position),
            tuple(cam.focal_point),
            tuple(cam.view_up),
        )
        if cam.parallel_projection:
            scene.plotter.enable_parallel_projection()
        else:
            scene.plotter.disable_parallel_projection()

    def set_style(self, scene: PvSceneHandle, style: "SceneStyle") -> None:
        """Apply scene-wide styling (background, grid).

        Font size is intentionally not propagated to a global —
        matplotlib's ``plt.rcParams["font.size"]`` is a process-wide
        knob, but PyVista takes font size per ``add_text`` /
        ``show_grid`` call. Layers that add text labels in Phase 3+
        can read ``style.font_size`` directly when they need it.
        """
        scene.plotter.background_color = style.background
        if style.grid:
            scene.plotter.show_grid()

    # ------------------------------------------------------------------ #
    # Primitives
    # ------------------------------------------------------------------ #
    def add_segments(
        self,
        scene: PvSceneHandle,
        segments: np.ndarray,
        *,
        color: Any = None,
        width: float | None = None,
        alpha: float | None = None,
        label: str | None = None,
    ) -> ActorRef:
        """Add a line-segment batch as a :class:`pyvista.PolyData`."""
        segs = np.asarray(segments, dtype=np.float64)
        if segs.size == 0:
            poly = pv.PolyData()
        else:
            n = int(segs.shape[0])
            points = segs.reshape(-1, 3).astype(np.float32)
            # VTK line topology: each segment is encoded as [2, i0, i1].
            lines = np.empty(n * 3, dtype=np.int64)
            lines[0::3] = 2
            lines[1::3] = np.arange(0, 2 * n, 2)
            lines[2::3] = np.arange(1, 2 * n, 2)
            poly = pv.PolyData(points, lines=lines)

        kwargs: dict[str, Any] = {}
        if color is not None:
            kwargs["color"] = color
        if width is not None:
            kwargs["line_width"] = float(width)
        if alpha is not None:
            kwargs["opacity"] = float(alpha)
        if label is not None:
            kwargs["label"] = label
        actor = scene.plotter.add_mesh(poly, **kwargs)
        return _PvActorRef(
            actor=actor, dataset=poly, scalar_field=None, kind="segments",
        )

    def add_points(
        self,
        scene: PvSceneHandle,
        points: np.ndarray,
        *,
        color: Any = None,
        size: float | None = None,
        scalars: np.ndarray | None = None,
        cmap: str | None = None,
    ) -> ActorRef:
        """Add a point cloud as a :class:`pyvista.PolyData`.

        ``scalars`` are stored in ``point_data["scalars"]`` so
        :meth:`update_scalars` has a known name to write into.
        ``render_points_as_spheres=True`` matches the apeGmsh
        recommendation for Gauss-marker clouds (avoids z-fighting
        at glancing angles).
        """
        pts = np.asarray(points, dtype=np.float64).astype(np.float32)
        if pts.size == 0:
            poly = pv.PolyData()
        else:
            poly = pv.PolyData(pts)

        kwargs: dict[str, Any] = {
            "render_points_as_spheres": True,
            "style": "points",
        }
        scalar_field: str | None = None
        if scalars is not None:
            scalar_field = "scalars"
            poly.point_data[scalar_field] = np.asarray(scalars, dtype=np.float64)
            kwargs["scalars"] = scalar_field
            if cmap is not None:
                kwargs["cmap"] = cmap
        elif color is not None:
            kwargs["color"] = color
        if size is not None:
            kwargs["point_size"] = float(size)
        actor = scene.plotter.add_mesh(poly, **kwargs)
        return _PvActorRef(
            actor=actor, dataset=poly,
            scalar_field=scalar_field, kind="points",
        )

    def add_polygons(
        self,
        scene: PvSceneHandle,
        polygons: Any,
        *,
        values: np.ndarray | None = None,
        point_values: np.ndarray | None = None,
        cmap: str | None = None,
        edge_color: Any = None,
    ) -> ActorRef:
        """Add filled polygons as a :class:`pyvista.PolyData`.

        ``polygons`` is a sequence of ``(M_i, 3)`` vertex arrays —
        one M-sided polygon per entry. Heterogeneous ``M_i`` is
        supported via VTK's variable-length face encoding.

        ``values`` (per-cell) is stored in ``cell_data["values"]``;
        ``point_values`` (per-vertex, in the same flattened order as
        the concatenated polygon vertex stream) is stored in
        ``point_data["point_values"]``. Passing both is a
        ``ValueError``. :meth:`update_scalars` writes into whichever
        field is bound — caller doesn't need to remember which.
        """
        if values is not None and point_values is not None:
            raise ValueError(
                "add_polygons accepts either values= or point_values=, "
                "not both."
            )
        polys_list = list(polygons) if not isinstance(polygons, list) else polygons
        if not polys_list:
            poly = pv.PolyData()
            kwargs: dict[str, Any] = {}
            if edge_color is not None:
                kwargs["edge_color"] = edge_color
                kwargs["show_edges"] = True
            actor = scene.plotter.add_mesh(poly, **kwargs)
            return _PvActorRef(
                actor=actor, dataset=poly,
                scalar_field=None, kind="polygons",
            )

        all_pts: list[np.ndarray] = []
        faces: list[int] = []
        offset = 0
        for p in polys_list:
            arr = np.asarray(p, dtype=np.float64)
            if arr.size == 0:
                continue
            n_v = int(arr.shape[0])
            all_pts.append(arr)
            faces.append(n_v)
            faces.extend(range(offset, offset + n_v))
            offset += n_v
        if not all_pts:
            poly = pv.PolyData()
        else:
            points = np.vstack(all_pts).astype(np.float32)
            poly = pv.PolyData(points, faces=np.asarray(faces, dtype=np.int64))

        kwargs = {}
        if edge_color is not None:
            kwargs["edge_color"] = edge_color
            kwargs["show_edges"] = True
        scalar_field: str | None = None
        if values is not None and poly.n_cells > 0:
            scalar_field = "values"
            poly.cell_data[scalar_field] = np.asarray(values, dtype=np.float64)
            kwargs["scalars"] = scalar_field
            kwargs["preference"] = "cell"
            if cmap is not None:
                kwargs["cmap"] = cmap
        elif point_values is not None and poly.n_points > 0:
            scalar_field = "point_values"
            poly.point_data[scalar_field] = np.asarray(
                point_values, dtype=np.float64,
            )
            kwargs["scalars"] = scalar_field
            kwargs["preference"] = "point"
            if cmap is not None:
                kwargs["cmap"] = cmap
        actor = scene.plotter.add_mesh(poly, **kwargs)
        return _PvActorRef(
            actor=actor, dataset=poly,
            scalar_field=scalar_field, kind="polygons",
        )

    def add_arrows(
        self,
        scene: PvSceneHandle,
        origins: np.ndarray,
        vectors: np.ndarray,
        *,
        scale: float = 1.0,
        color: Any = None,
        cmap: str | None = None,
    ) -> ActorRef:
        """Add an arrow-glyph batch via :meth:`pyvista.Plotter.add_arrows`.

        PyVista's ``add_arrows`` internally builds a glyph filter
        from origins + directions; the resulting actor is **not**
        cheaply mutable, so :meth:`update_points` raises
        :class:`BackendCapabilityError` for arrows. Animation-driven
        vector updates go through ``remove`` + a fresh
        :meth:`add_arrows`. The
        :class:`~STKO_to_python.viewer.layers.VectorLayer` already
        carries that perf gap noted from Phase 2.6; landing an
        in-place ``update_arrows`` primitive is a Phase 3.X
        follow-up.
        """
        o = np.asarray(origins, dtype=np.float64).astype(np.float32)
        v = (np.asarray(vectors, dtype=np.float64) * float(scale)).astype(np.float32)
        actor = scene.plotter.add_arrows(o, v, mag=1.0)
        if color is not None:
            # PyVista's add_arrows ignores color in some versions —
            # apply via actor property as a fallback.
            try:
                actor.prop.color = color
            except Exception:  # pragma: no cover - defensive
                pass
        return _PvActorRef(
            actor=actor, dataset=None, scalar_field=None, kind="arrows",
        )

    # ------------------------------------------------------------------ #
    # In-place actor updates
    # ------------------------------------------------------------------ #
    def update_scalars(self, actor: ActorRef, scalars: np.ndarray) -> None:
        """Mutate an actor's scalar field in place.

        Requires the actor was created with a ``scalars=`` /
        ``values=`` / ``point_values=`` argument (which bound the
        field name at creation time). Arrows have no scalar field —
        they raise :class:`BackendCapabilityError`. The write is
        dispatched to ``point_data`` or ``cell_data`` explicitly
        based on where the field was originally bound, so length-
        equal point and cell arrays never collide.
        """
        ref = self._unwrap(actor)
        if ref.dataset is None or ref.scalar_field is None:
            raise BackendCapabilityError(
                f"Actor (kind={ref.kind!r}) does not support scalar updates "
                "— no scalar field was bound at creation."
            )
        arr = np.asarray(scalars, dtype=np.float64)
        field = ref.scalar_field
        if field in ref.dataset.point_data:
            ref.dataset.point_data[field] = arr
        elif field in ref.dataset.cell_data:
            ref.dataset.cell_data[field] = arr
        else:
            # First-time binding via the dataset's __setitem__ dispatch.
            # Used for the ``points`` kind, whose scalars live on
            # ``point_data`` from creation.
            ref.dataset[field] = arr
        ref.dataset.Modified()

    def update_points(self, actor: ActorRef, points: np.ndarray) -> None:
        """Mutate an actor's vertex positions in place.

        Handles segments (``(N, 2, 3)`` → flatten), points
        (``(N, 3)``), and polygons (``(N, 3)`` — face topology
        unchanged). Arrows raise :class:`BackendCapabilityError`
        because PyVista's arrow glyph is built by a non-trivially
        invertible filter; callers go through ``remove`` + a fresh
        :meth:`add_arrows`.
        """
        ref = self._unwrap(actor)
        if ref.kind == "arrows" or ref.dataset is None:
            raise BackendCapabilityError(
                f"Actor (kind={ref.kind!r}) does not support point updates."
            )
        pts = np.asarray(points, dtype=np.float64)
        if ref.kind == "segments" and pts.ndim == 3:
            pts = pts.reshape(-1, 3)
        ref.dataset.points = pts.astype(np.float32)
        ref.dataset.Modified()

    def set_visible(self, actor: ActorRef, visible: bool) -> None:
        ref = self._unwrap(actor)
        ref.actor.SetVisibility(bool(visible))

    def remove(self, scene: PvSceneHandle, actor: ActorRef) -> None:
        """Detach ``actor`` from the scene. Idempotent if already removed."""
        ref = self._unwrap(actor)
        try:
            scene.plotter.remove_actor(ref.actor)
        except Exception:  # pragma: no cover - defensive (plotter closed, etc.)
            pass

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def show(self, scene: PvSceneHandle) -> None:
        """Hand off to :meth:`pyvista.Plotter.show` unless off-screen."""
        if scene.off_screen:
            return
        scene.plotter.show()

    def save(self, scene: PvSceneHandle, path: Path, *, dpi: int = 300) -> None:
        """Persist the current frame to disk.

        ``dpi`` is honoured as best as PyVista allows — the plotter
        has no direct DPI knob, so the window size is interpreted
        relative to a notional 96-DPI canvas and ``dpi`` scales it.
        Callers that need precise pixel counts should set
        ``scene.plotter.window_size`` before calling.
        """
        self._require_gl_context(scene)
        if dpi != 300:
            # Best-effort scaling from default DPI.
            scale = max(1, int(round(dpi / 96.0)))
            # transparent_background defaults to None which avoids forcing
            # alpha; explicit False matches the matplotlib backend.
            scene.plotter.screenshot(
                str(path), transparent_background=False, scale=scale,
            )
        else:
            scene.plotter.screenshot(str(path), transparent_background=False)

    def snapshot(self, scene: PvSceneHandle) -> np.ndarray:
        """Return the current frame as an ``(H, W, 3)`` ``uint8`` RGB array."""
        self._require_gl_context(scene)
        img = scene.plotter.screenshot(
            return_img=True, transparent_background=False,
        )
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr.astype(np.uint8, copy=False)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _require_gl_context(scene: PvSceneHandle) -> None:
        """Raise instead of rendering when VTK has no OpenGL context.

        Called before every render. See :func:`_missing_gl_context` for
        what the check can and cannot detect.
        """
        if scene._gl_checked:
            return
        if _missing_gl_context():
            raise RuntimeError(
                "PyVistaBackend cannot render: on Linux VTK needs an OpenGL "
                "context, and this process has none. DISPLAY and "
                "WAYLAND_DISPLAY are unset, and no EGL or OSMesa library was "
                "found; VTK would segfault on the first render. Provide one "
                "of: an X server (run under Xvfb, e.g. "
                "'xvfb-run -a -s \"-screen 0 1024x768x24\" python ...'), an "
                "EGL library (libEGL, from Mesa or the GPU driver), or OSMesa "
                "(libOSMesa). See docs/viewer/03-deployment-targets.md §7."
            )
        scene._gl_checked = True

    @staticmethod
    def _unwrap(actor: ActorRef) -> _PvActorRef:
        if not isinstance(actor, _PvActorRef):
            raise BackendCapabilityError(
                f"Expected a PyVistaBackend actor reference, got "
                f"{type(actor).__name__}. Layers must not pass raw VTK "
                "actors back through the backend API."
            )
        return actor


__all__ = ["PvSceneHandle", "PyVistaBackend"]
