---
name: stko-viewer
description: >
  Checklist and visual-verification procedure for CHANGING the STKO_to_python viewer
  (src/STKO_to_python/viewer/: Scene, Layer, Backend, the MPCO DataSource adapter,
  MplBackend, PyVistaBackend, Mesh/DeformedMesh/Node/Vector/Contour layers, viewer.math)
  or a ds.plot.* entry point that routes through it (ds.plot.mesh, ds.plot.deformed_shape).
  Use before editing viewer code or tests/viewer/, and before claiming that a plot change
  is "byte-identical" or "visually unchanged". Not for making plots with the library --
  that is the end-user stko-to-python skill.
---

# Changing the viewer — checklist and visual verification

Read this before changing anything under `src/STKO_to_python/viewer/` or `tests/viewer/`,
or a `ds.plot.*` path that builds a `Scene`. **A green test suite is not visual
verification.** The rewire tests from #89 and #90 assert counts and segment data, not
pixels, even though `docs/viewer/00-roadmap.md` "Definition of done" asks for an image diff.

Out of scope: making plots with the library (the end-user `stko-to-python` skill).

## Design rules (read the source doc; don't re-derive them)

- [ ] Coupling: layers never import a backend, core never imports backends, and cuts
      never import the viewer. `docs/viewer/01-architecture.md` "Coupling boundaries
      (don't cross these)". Only review enforces these.
- [ ] `update_*` mutates in place, with no actor recreation per step. A backend that
      can't do something raises `BackendCapabilityError`; never fall back silently.
      Same doc, "The `Backend` protocol" (its "Hard rules").
- [ ] `import STKO_to_python.viewer` stays light: no pyvista, vtk or Qt at import time
      (`tests/viewer/test_smoke.py`).
- [ ] Ports from apeGmsh: `docs/viewer/02-porting-from-apegmsh.md` "Process for each port".

## Environment — say which one ran the tests

- [ ] Without the `[viewer-3d]` extra, `pytest.importorskip` skips every PyVista test,
      and `-ra` lists them as "could not import 'pyvista'". A backend or 3-D change
      needs a run in a venv with `pip install -e ".[viewer-3d,test]"`. Name that env and
      its pyvista/vtk versions in the PR.
- [ ] CI's "Viewer extras resolve" job has been red since #92: `PyVistaBackend.snapshot`
      segfaults (exit 139) on the headless Ubuntu runner, so CI validates no PyVista
      code, and every merge after #92 landed on red. `AGENTS.md`: "Never merge on a
      red check".

## Visual verification — reference vs candidate

1. The reference is the base commit, rendered from its own tree:
   `git worktree add <tmp>/ref origin/main` (or `git archive origin/main src | tar -x -C <tmp>/ref`).
2. Render the SAME calls from both trees, off-screen, on the same fixture and at the same dpi:
   ```python
   # PYTHONPATH=<tree>/src MPLBACKEND=Agg python render.py
   ds = MPCODataSet(fixture_dir, "results", verbose=False)
   calls = {"mesh": lambda: ds.plot.mesh(),
            "def_s0": lambda: ds.plot.deformed_shape(model_stage="MODEL_STAGE[1]", step=0, scale=10.0),
            "def_s5": lambda: ds.plot.deformed_shape(model_stage="MODEL_STAGE[1]", step=5, scale=10.0)}
   for name, call in calls.items():
       ax, meta = call(); ax.figure.savefig(out / f"{name}.png", dpi=100); plt.close(ax.figure)
   ```
   For a `Scene` you build yourself, pass `off_screen=True`, then call `scene.snapshot()`
   (an H×W×3 uint8 array) or `scene.save(path)`.
3. Compare each pair with `np.array_equal(plt.imread(ref), plt.imread(cand))`. A
   "byte-identical" claim requires IDENTICAL. For anything else, report the differing
   pixel count and the reason for it.
4. **Open and look at every candidate image.** The diff tells you *that* something
   changed, not whether the result is right. Check for:
   - clipping: axes limits or camera framing that cut off the model (`Scene.fit_bounds`,
     `_autoscale_axes` in `plotting/deformed_shape.py`);
   - overlap and z-order (matplotlib): `MeshLayer` draws at 1.0, `ContourLayer` at 1.5,
     and the deformed shape and scatter at 2.0; look for an overlay hidden behind
     another, or a colorbar on top of the model;
   - stale state: frames at two steps must differ when the data differs (step 0 vs
     step 5 on `elasticFrame/results` differ by about 84 px). A layer whose
     `update_to_step` fails to mutate its actor keeps showing the attach step.
     `ContourLayer` freezes `clim` at the attach step on purpose: the colorbar stays
     fixed while the colors change;
   - empty selections: filtering out a whole element class must neither raise nor
     leave old geometry on screen.
5. Fixtures: `elasticFrame/results` (beams), `elasticFrame/QuadFrame_results` (MP
   shells), `elasticFrame/elasticFrame_mesh_displacementBased_results`. Solids need the
   gitignored `solid_partition_example`.
6. Input: there is no Qt UI yet (roadmap Phase 4). Drive interaction from code: call
   `scene.set_step(k)`, toggle `layer.visible`, and call `viewer.math.picking` with
   known coordinates, then assert on the result.

## Clean up what you launched

- [ ] Never call `show()`, `plt.show()` or `plotter.show()` on a scene that isn't
      off-screen: in an agent session it blocks. Close every figure (`plt.close(fig)`)
      and every plotter (`plotter.close()`) you open, as the tests do.
- [ ] If you start a process (a Qt window, a notebook kernel, a render subprocess),
      record its PID at launch and kill that PID when you are done. Never kill by name.

## Before the PR

- [ ] `PYTHONPATH=src MPLBACKEND=Agg python -m pytest tests/viewer tests/integration/test_mesh_plot.py tests/integration/test_deformed_shape.py -q`,
      plus the pyvista-env run for 3-D changes.
- [ ] In the PR body: the env that ran the pyvista tests, the image-diff table
      (file: IDENTICAL or N px), and the list of images you opened.

Found a new trap? Add it to `AGENTS.md` "Lessons from incidents", then add a line here.
