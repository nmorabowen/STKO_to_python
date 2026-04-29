# Stress contour on a shear wall

> Render the membrane stress σ_xx across a shell wall at the step
> where it peaks, overlaid on the model wireframe.

The engineering question is: *at the most-loaded instant of the
analysis, where in the wall is the membrane carrying the largest
in-plane stress?* In OpenSees, shells like `203-ASDShellQ4` write a
`section.force` recorder bucket with one set of resultants per Gauss
point. The *peak step* is whichever step makes that resultant largest
in absolute value, and the *contour* is just an `(x, z)` scatter
colored by the value at every IP.

The fixture used here is
`stko_results_examples/elasticFrame/QuadFrame_results` — a 5×5 m wall
panel meshed with `203-ASDShellQ4` shells (4 Gauss points each) and
edge beams. Two partition files (`results.part-0.mpco`,
`results.part-1.mpco`) are merged transparently; you never write
partition-aware code.

```python
from pathlib import Path
import matplotlib.pyplot as plt

from STKO_to_python import MPCODataSet

REPO_ROOT = Path("path/to/STKO_to_python")
DATASET = REPO_ROOT / "stko_results_examples" / "elasticFrame" / "QuadFrame_results"
PUSHOVER_STAGE = "MODEL_STAGE[2]"

ds = MPCODataSet(str(DATASET), "results", verbose=False)
print(ds.unique_element_types)
# ['203-ASDShellQ4[...]', '5-ElasticBeam3d[...]']
```

## 1. Pull `section.force` for every shell in the wall

```python
edf = ds.elements_info["dataframe"]
shell_ids = (
    edf.query("element_type == '203-ASDShellQ4'")["element_id"]
    .astype(int)
    .tolist()
)

er = ds.elements.get_element_results(
    results_name="section.force",
    element_type="203-ASDShellQ4",
    model_stage=PUSHOVER_STAGE,
    element_ids=shell_ids,
)
print(er.list_canonicals())
# ('bending_moment_xx', 'bending_moment_xy', 'bending_moment_yy',
#  'membrane_xx', 'membrane_xy', 'membrane_yy',
#  'transverse_shear_xz', 'transverse_shear_yz')
print(er.n_ip, er.gp_natural.shape)   # 4, (4, 2)
```

## 2. Identify "the peak step" via `time_of_peak`

`time_of_peak` returns, per element, the step index at which the
named column is largest in absolute value. To pick a single global
peak step for the whole wall, take the mode (or any other reduction)
of those per-element values:

```python
# membrane_xx has 4 columns (Fxx_ip0 ... Fxx_ip3) — peak across them
fxx_cols = er.canonical_columns("membrane_xx")
peak_per_ip_per_elem = er.df[list(fxx_cols)].abs().groupby(
    level="element_id"
).max()
# step at which each element peaks
peak_step_per_elem = (
    er.df[list(fxx_cols)].abs().max(axis=1)
    .groupby(level="element_id").idxmax().str[1]
)
peak_step = int(peak_step_per_elem.mode().iloc[0])
print(f"Global peak step: {peak_step}  (t = {er.time[peak_step]:.3f})")
```

For the simpler "peak of one fixed column" idiom, the built-in helper
works too:

```python
# step where Fxx at IP 0 peaks (per element)
er.time_of_peak("Fxx_ip0").head()
# element_id
# 101    9
# 102    9
# ...
```

## 3. Render the contour

`ds.plot.mesh_with_contour` is the one-shot helper: it draws the
element wireframe and overlays a per-IP physical-space scatter
colored by the canonical value. Internally it calls
`ds.plot.mesh()` and then `er.plot.scatter(..., ax=ax)` — see
`er.plot.scatter` if you need finer control.

```python
ax, meta = ds.plot.mesh_with_contour(
    er,
    "membrane_xx",
    step=peak_step,
    element_type="203-ASDShellQ4",
    axes=("x", "z"),                # elevation view, not top-down
    cmap="RdBu_r",
    s=24,
)
# meta["scatter"] is the inner scatter meta — its "scatter" key is the
# matplotlib handle that the colorbar wants.
ax.figure.colorbar(
    meta["scatter"]["scatter"], ax=ax, label="σ_xx (membrane)",
)
ax.set_title(f"Membrane σ_xx at step {peak_step} (t = {er.time[peak_step]:.2f})")
plt.show()
```

`meta` is `{"mesh": <plot_mesh meta>, "scatter": <ElementResultsPlotter.scatter meta>}`.
The inner scatter meta carries the raw `x`, `y`, `values`, and the
matplotlib `scatter` handle — useful if you want a colorbar or want to
compose with other axes.

## 4. Hand-built version (without the helper)

If you want finer control — different mesh styling, or scatter
without the wireframe — call the underlying methods directly. Note
that `ds.plot.mesh()` returns a 3-D axes when the model has a
non-trivial `Z` extent; let it create the axes (don't pre-make a 2-D
one with `plt.subplots()`):

```python
ax, _ = ds.plot.mesh(
    element_type="203-ASDShellQ4",
    edge_color="lightgray",
    linewidth=0.4,
)
er.plot.scatter("membrane_xx", step=peak_step, ax=ax,
                axes=("x", "z"), cmap="RdBu_r", s=24)
```

For pure 2-D scatter without a backdrop, drop the mesh call entirely —
`er.plot.scatter` will create its own 2-D axes.

`er.plot.scatter` requires `physical_coords()` to resolve — i.e. the
element class must be in the shape-function catalog
(`utilities/shape_functions.py`) and node coords must have been
captured at fetch time. Both hold for `203-ASDShellQ4`.

## 5. Verify with `peak_abs`

`peak_abs` collapses the time axis to the per-element max-absolute
value across whatever columns you ask for — useful for sanity-checking
that the chosen step is actually loaded:

```python
peaks = er.peak_abs(component="Fxx_ip0").sort_values(
    "Fxx_ip0_peak_abs", ascending=False
)
print(peaks.head())
```

## Variations

- **Different stress component**: swap `"membrane_xx"` for
  `"bending_moment_xx"`, `"membrane_yy"`, or `"transverse_shear_xz"`.
  Names are listed in `er.list_canonicals()`.
- **Top-down view** (e.g. for a slab): pass `axes=("x", "y")`.
- **Top-of-section stress on a layered shell**: fetch
  `"material.stress"` instead of `"section.force"` — that bucket
  carries σ_11 etc. per layer per Gauss point. The contour idiom is
  unchanged; the canonical name becomes `"stress_11"`.
- **Live-loaded model** (this fixture has trivial loading and the
  contour is largely flat): re-run against your own model with a
  proper lateral load case and the contrast lights up.
- **Single-step plot from a saved `ElementResults`**: pickle the result
  with `er.save_pickle("wall.pkl.gz")` and reload — `physical_coords()`
  still resolves, so the contour pipeline works without re-opening the
  HDF5 files.

For a wider tour of the multi-partition shell API see
[Quad-frame shells](../examples/quad_frame_shell.md).
