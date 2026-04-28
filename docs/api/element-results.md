# ElementResults

Self-contained container for element-level results extracted from an
MPCO HDF5 file. Once created via `ds.elements.get_element_results()`,
it carries its data, time array, Gauss-point geometry, and node
coordinates entirely within the object — independent of the original
dataset. It can be pickled and reloaded without re-opening the HDF5
files.

---

## Construction

```python
from STKO_to_python import MPCODataSet

ds = MPCODataSet(r"C:\path\to\results", "results")

er = ds.elements.get_element_results(
    results_name="section.force",      # HDF5 group name
    element_type="203-ASDShellQ4",     # base element class
    model_stage="MODEL_STAGE[1]",      # omit → first stage
    element_ids=[101, 102, 103],       # explicit IDs, or…
    # selection_set_name="ShearWall",  # …named selection set
    # selection_set_id=7,              # …selection set by index
)
```

All three selection modes may be combined (their results are unioned).

---

## Key attributes

| Attribute | Type | Description |
|---|---|---|
| `er.df` | `pd.DataFrame` | MultiIndex `(element_id, step)` × component columns |
| `er.time` | `np.ndarray` | Time value for each step |
| `er.element_ids` | `tuple[int, ...]` | Element IDs in this result set |
| `er.element_type` | `str` | Base element class (e.g. `"203-ASDShellQ4"`) |
| `er.results_name` | `str` | HDF5 result group name (e.g. `"section.force"`) |
| `er.model_stage` | `str` | Model stage name |
| `er.n_elements` | `int` | Number of elements |
| `er.n_steps` | `int` | Number of recorded steps |
| `er.n_components` | `int` | Number of columns in `df` |
| `er.n_ip` | `int` | Integration-point count (0 for closed-form) |
| `er.gp_xi` | `np.ndarray \| None` | 1-D natural coordinates ξ ∈ [-1,+1] (line elements with `GP_X` attribute only) |
| `er.gp_natural` | `np.ndarray \| None` | Multi-D natural coordinates from static catalog, shape `(n_ip, gp_dim)` |
| `er.gp_weights` | `np.ndarray \| None` | Quadrature weights, shape `(n_ip,)` (catalog-driven elements only) |
| `er.gp_dim` | `int` | Spatial dimension of the parent domain (1/2/3) |
| `er.element_node_coords` | `np.ndarray \| None` | Node coordinates per element, shape `(n_e, n_nodes, 3)` |
| `er.plot` | `ElementResultsPlotter` | Bound plotting helper — see below |

---

## Introspection

```python
er.list_components()      # tuple of all column names
er.list_canonicals()      # tuple of engineering canonical names
er.n_elements             # int
er.n_steps                # int
er.empty                  # bool — True if df is empty
```

---

## Fetching data

```python
# All components, all elements
df = er.fetch()

# Single component
s = er.fetch("Mz_1")
s = er.fetch("P_ip2", element_ids=[100, 101])

# Attribute-style access
s = er.Mz_1.series        # full column
s = er.Mz_1[[100, 101]]  # filtered
s = er.Mz_1[100]          # single element
```

---

## Time snapshots

```python
df = er.at_step(100)        # all elements at step 100
df = er.at_time(5.0)        # nearest step to t=5.0
```

---

## Envelope

```python
env = er.envelope()              # all components → {col}_min, {col}_max per element
env = er.envelope("Mz_1")       # single component
```

---

## Integration-point slicing

```python
sub = er.at_ip(0)     # columns ending in _ip0 only
sub = er.at_ip(3)     # columns ending in _ip3 only
```

Raises `ValueError` for closed-form buckets (`n_ip == 0`) or
out-of-range indices.

---

## Canonical engineering names

```python
er.list_canonicals()
# e.g. ('membrane_xx', 'membrane_yy', 'bending_moment_xx', ...)

er.canonical_columns("membrane_xx")
# ('Fxx_ip0', 'Fxx_ip1', 'Fxx_ip2', 'Fxx_ip3')

df = er.canonical("bending_moment_z")
```

See [Canonical names](canonical-names.md) for the full map.

---

## Physical-space integration

```python
phys = er.physical_coords()   # (n_e, n_ip, 3) — physical (x,y,z) per IP
dets = er.jacobian_dets()     # (n_e, n_ip) — |J| per IP

# Volume/area/length integral of a canonical quantity per element
s = er.integrate_canonical("stress_11")  # Series indexed (element_id, step)
```

`physical_coords()` and `jacobian_dets()` return `None` for closed-form
buckets, when element class is not in the shape-function catalog, or
when node coordinates weren't populated at fetch time.
`integrate_canonical()` raises `ValueError` in those cases with a
pointer to manual integration.

---

## Time-series statistics

Per-element scalar summaries over the full step history:

```python
peaks = er.peak_abs()                    # all components — abs peak per element
peaks = er.peak_abs(component="Mz_1")   # single component

idx = er.time_of_peak("Mz_1")           # step index of abs peak (default)
idx = er.time_of_peak("Mz_1", abs=False) # step index of signed peak

ce = er.cumulative_envelope("N_1")
# MultiIndex (element_id, step): N_1_running_min, N_1_running_max

df_summary = er.summary()
# One row per element: max, min, peak_abs, residual, mean
```

---

## DataFrame export

```python
df_flat = er.to_dataframe(include_time=True)
# Flat DataFrame with 'time' column appended
```

---

## Pickle serialization

```python
er.save_pickle("wall_forces.pkl")
er.save_pickle("wall_forces.pkl.gz")   # compressed

from STKO_to_python.elements.element_results import ElementResults
er = ElementResults.load_pickle("wall_forces.pkl")
er = ElementResults.load_pickle("wall_forces.pkl.gz")
```

Compression is auto-detected from the `.gz` extension. All geometry
arrays (`element_node_coords`, `gp_natural`, `gp_weights`, `gp_xi`)
survive the round-trip, so `physical_coords()`, `jacobian_dets()`, and
`integrate_canonical()` work on reloaded objects without re-opening the
HDF5 files.

---

## API reference

::: STKO_to_python.elements.element_results.ElementResults

---

# ElementResultsPlotter

Bound to an `ElementResults` instance as `er.plot`. Three engineering
plot families that mirror the structure of
[NodalResultsPlotter](plotting.md#nodalresultsplotter).

All methods accept an optional `ax=` to compose into existing figures
and return `(ax, meta)` for programmatic inspection.

---

## `plot.history` — time history

```python
ax, meta = er.plot.history("Mz_1", element_ids=[1, 2, 3])
ax, meta = er.plot.history("P_ip2", x_axis="step")
```

| Parameter | Default | Description |
|---|---|---|
| `component` | required | Column name (e.g. `"Mz_1"`, `"P_ip2"`, `"sigma11_l0_ip0"`) |
| `element_ids` | `None` (all) | Restrict to specific elements |
| `ax` | `None` | Existing axes to draw on |
| `x_axis` | `"time"` | `"time"` or `"step"` |
| `**plot_kwargs` | | Forwarded to `ax.plot` |

Returns `(ax, {"x": ..., "y_per_element": {eid: array}})`.

---

## `plot.diagram` — beam force/moment/strain diagram

For line elements only (`gp_dim == 1`). Plots a canonical quantity as
a function of physical position along the beam at a single step — the
classic moment/shear/axial diagram.

```python
ax, meta = er.plot.diagram("bending_moment_z", element_id=1, step=100)
ax, meta = er.plot.diagram("axial_force", element_id=1, step=100,
                            x_in_natural=True)  # xi in [-1, +1]
```

| Parameter | Default | Description |
|---|---|---|
| `component_canonical` | required | Canonical name — must resolve to `n_ip` columns |
| `element_id` | required | Single element to plot |
| `step` | required | Step index |
| `ax` | `None` | Existing axes |
| `x_in_natural` | `False` | Plot in ξ ∈ [-1,+1]; `False` → physical position along element |
| `**plot_kwargs` | | Forwarded to `ax.plot` |

Returns `(ax, {"x": ..., "y": ..., "columns": [...]})`.

Raises `ValueError` for non-line elements (use `scatter()` instead),
missing node coords when `x_in_natural=False`, or when the canonical
doesn't resolve to exactly `n_ip` columns.

---

## `plot.scatter` — spatial scatter for shells/solids

Scatter integration-point physical positions colored by component value
at a single step. Lightweight stand-in for contour plots — no mesh
renderer needed.

```python
ax, meta = er_shell.plot.scatter("membrane_xx", step=100)
ax, meta = er_shell.plot.scatter("membrane_xx", step=100, axes=("x", "z"))
ax, meta = er_brick.plot.scatter("stress_11", step=100)

# Add a colorbar
fig = ax.figure
fig.colorbar(meta["scatter"], ax=ax)
```

| Parameter | Default | Description |
|---|---|---|
| `component_canonical` | required | Canonical name — must resolve to `n_ip` columns |
| `step` | required | Step index |
| `ax` | `None` | Existing axes |
| `axes` | `("x", "y")` | Two of `"x"`, `"y"`, `"z"` for the 2-D projection |
| `**scatter_kwargs` | | Forwarded to `ax.scatter` (e.g. `cmap`, `s`) |

Returns `(ax, {"x": ..., "y": ..., "values": ..., "scatter": PathCollection})`.

Raises `ValueError` when `physical_coords()` returns `None` (closed-form bucket,
unknown element class, or missing node coords).

---

## API reference

::: STKO_to_python.elements.element_results_plotting.ElementResultsPlotter
