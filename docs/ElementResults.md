# ElementResults — Element Results Container

`ElementResults` is a self-contained container for element-level results extracted from MPCO HDF5 files. Once created (via `ds.elements.get_element_results()`), it carries the data, time array, and metadata needed for post-processing — independent of the original dataset. It can be pickled and reloaded later without needing the HDF5 files.

## Creating an ElementResults Object

```python
from STKO_to_python import MPCODataSet

ds = MPCODataSet(r"C:\path\to\results", "mpco")

# Extract element results
er = ds.elements.get_element_results(
    results_name="globalForces",
    element_type="203-ASDShellQ4",
    selection_set_name="ShearWall",
)

# From this point on, 'er' is fully independent of 'ds'
```

### Selection Options

You must provide **at least one** of `element_ids`, `selection_set_id`, or `selection_set_name`. If multiple are given, they are unioned. The `element_type` parameter is always required. Element selection supports three modes:

```python
# By selection set name (case-insensitive)
er = ds.elements.get_element_results(
    results_name="globalForces",
    element_type="203-ASDShellQ4",
    selection_set_name="ShearWall",
)

# By selection set ID
er = ds.elements.get_element_results(
    results_name="globalForces",
    element_type="203-ASDShellQ4",
    selection_set_id=7,
)

# By explicit element IDs
er = ds.elements.get_element_results(
    results_name="section_deformation",
    element_type="2-truss",
    element_ids=[100, 101, 102, 103],
)
```

### Specifying Model Stage

By default the first model stage is used. To specify:

```python
er = ds.elements.get_element_results(
    results_name="globalForces",
    element_type="203-ASDShellQ4",
    selection_set_name="ShearWall",
    model_stage="MODEL_STAGE[2]",
)
```

## Internal Structure

The core data is in `er.df` — a pandas DataFrame with:

- **Index**: MultiIndex `(element_id, step)`
- **Columns**: real component names parsed from each bucket's
  `META/COMPONENTS`. The shape depends on the bucket topology — see
  [Column naming](#column-naming) below. Falls back to
  `val_1, val_2, ..., val_N` only when META is absent.

Additional attributes:

```python
er.time            # np.ndarray — time values per step
er.name            # str — model name
er.element_ids     # tuple[int, ...] — element IDs in this result set
er.element_type    # str — base element type (e.g. "203-ASDShellQ4")
er.results_name    # str — HDF5 result group name (e.g. "globalForces")
er.model_stage     # str — model stage this data comes from
```

## Column naming

Names come straight from each bucket's `META/COMPONENTS`. The four
patterns the parser produces:

| Bucket shape | Example | Names |
|---|---|---|
| Closed-form beam (`force`, `localForce`) | `5-ElasticBeam3d` | `Px_1, Py_1, ..., Mz_2` (global) or `N_1, Vy_1, ..., Mz_2` (local) |
| Closed-form continuum (`force` on solid) | `56-Brick`, 8 nodes × 3 DOFs | `P1_1, P2_1, P3_1, ..., P3_8` |
| Line-stations (`section.force`, `section.deformation`) | force-/disp-based beam, 5 IPs × 4 section comps | `P_ip0, Mz_ip0, My_ip0, T_ip0, ..., T_ip4` |
| Gauss-level continuum (`material.stress`, `material.strain`) | `56-Brick`, 8 IPs × 6 stress comps | `sigma11_ip0, sigma22_ip0, ..., sigma13_ip7` |
| Compressed fibers (`section.fiber.stress`) | `MULTIPLICITY > 1`, 6 fibers × 2 IPs | `sigma11_f0_ip0, sigma11_f1_ip0, ..., sigma11_f5_ip1` |
| Layered shell (`section.fiber.damage` on ASDShellQ4) | 4 IPs × layered concrete section (only some layers carry damage) | `d+_l0_ip0, d-_l0_ip0, d+_l2_ip0, ..., d-_l4_ip3` |
| Layered + fibers | rare; both layers and fibers per layer | `<comp>_f<fiber>_l<layer>_ip<gauss>` |

See [`docs/mpco_format_conventions.md`](mpco_format_conventions.md) for
the underlying META layout, and `STKO_to_python/io/meta_parser.py` for
the parser that produces these names.

## Introspection

```python
er.list_components()
# ('Px_1', 'Py_1', 'Pz_1', 'Mx_1', 'My_1', 'Mz_1',
#  'Px_2', 'Py_2', 'Pz_2', 'Mx_2', 'My_2', 'Mz_2')

er.n_components    # 12
er.n_elements      # 150
er.n_steps         # 2000
er.empty           # False
```

## Fetching Data

### Basic fetch

```python
# All components, all elements
df = er.fetch()

# Single component (e.g., axial force at IP 2)
s = er.fetch("P_ip2")

# Filter by element IDs
s = er.fetch("Mz_1", element_ids=[100, 101])

# All components, filtered elements
df = er.fetch(element_ids=[100, 101])
```

### Attribute-style access (ResultView)

Each column is available as an attribute via `_ElementResultView`:

```python
# Equivalent to er.fetch("Mz_1")
s = er.Mz_1.series

# Filter by element IDs
s = er.Mz_1[100]           # single element
s = er.Mz_1[[100, 101]]    # multiple elements
s = er.Mz_1[:]             # all elements
```

## Envelope

Compute min/max over all time steps for each element:

```python
# All components
env = er.envelope()
# Returns DataFrame indexed by element_id:
#   Px_1_min, Px_1_max, Py_1_min, Py_1_max, ...

# Single component
env = er.envelope(component="Mz_1")
# Returns DataFrame indexed by element_id:
#   Mz_1_min, Mz_1_max
```

## Canonical (engineering-friendly) names

The on-disk MPCO shortnames vary by element family — `P_ip0` for axial
force at the first IP of a line-station beam, `N_1` for axial force at
node 1 of a closed-form `localForce` bucket, `Fxx_ip0` for shell
membrane Fxx, `sigma11_ip7` for the 11-stress at the 8th Gauss point
of a brick. The canonical-name layer maps engineering quantities
(`axial_force`, `bending_moment_z`, `stress_11`, `membrane_xx`, …) to
whichever shortnames are present in the result:

```python
er = ds.elements.get_element_results("section.force", "64-DispBeamColumn3d", element_ids=[1])
er.list_canonicals()
# ('axial_force', 'bending_moment_y', 'bending_moment_z', 'torsion')

er.canonical_columns("axial_force")
# ('P_ip0', 'P_ip1', 'P_ip2', 'P_ip3', 'P_ip4')

er.canonical("bending_moment_z")    # DataFrame view
# columns: Mz_ip0, Mz_ip1, Mz_ip2, Mz_ip3, Mz_ip4
```

Asking for a quantity that doesn't apply to the result raises:

```python
er_shell = ds.elements.get_element_results("section.force", "203-ASDShellQ4", element_ids=[100])
er_shell.canonical("axial_force")
# ValueError: No columns matching canonical name 'axial_force'.
# Present canonicals: ('membrane_xx', 'membrane_yy', ..., 'transverse_shear_yz')
```

The full canonical → MPCO-shortname map and the regex used to strip
`_<int>` / `_ip<int>` / `_f<int>_ip<int>` suffixes live in
[`elements/canonical.py`](api/index.md). Note: `My`/`Mz` and similar
local↔global axis collisions (per
[`mpco_format_conventions.md` §9](mpco_format_conventions.md)) are
**disambiguated by which results bucket you fetched**, not by the
canonical name. Fetch `localForce` for element-local moments, fetch
`force` (globalForce) and access columns directly by shortname for
global-axis moments.

## Integration-point coordinates

Line-station and gauss-level buckets that have a `GP_X` attribute on
their connectivity dataset (force/disp-based beam-columns) expose the
integration-point positions as `er.gp_xi` — a 1-D numpy array in
**natural coordinates** ξ ∈ [-1, +1].

```python
er = ds.elements.get_element_results("section.force", "64-DispBeamColumn3d", element_ids=[1])
er.n_ip       # 5  (number of integration points)
er.gp_xi      # array([-1., -0.65, 0., 0.65, 1.])  — Lobatto 5-pt
```

To convert to physical coordinates [0, L] for plotting:

```python
x = er.physical_x(beam_length=2.0)
# array([0., 0.345, 1., 1.655, 2.])
```

Per-IP slicing:

```python
sub = er.at_ip(2)                  # only columns ending '_ip2'
# columns: ['P_ip2', 'Mz_ip2', 'My_ip2', 'T_ip2']
moments_at_midspan = sub["Mz_ip2"]
```

Closed-form buckets (no integration points) have `er.gp_xi is None`,
`er.n_ip == 0`, and calling `er.at_ip(...)` or `er.physical_x(...)`
raises `ValueError`.

### Multi-dimensional integration points (shells & solids)

For shells, plane elements, and 3-D solids the integration-point
positions are *fixed by the element class* and not written to disk.
The library carries a static catalog at
[`utilities/gauss_points.py`](api/index.md) and exposes the resolved
layout as a multi-dimensional array on `ElementResults`:

```python
# Brick continuum, 8 Gauss points
er = ds.elements.get_element_results("material.stress", "56-Brick", element_ids=[100])
er.n_ip            # 8
er.gp_dim          # 3
er.gp_natural      # shape (8, 3) — (ξ, η, ζ) at ±1/√3 per axis
er.gp_weights      # shape (8,)   — all 1.0 for 2×2×2 Gauss-Legendre

# Shell, 4 Gauss points
er_shell = ds.elements.get_element_results("section.force", "203-ASDShellQ4", element_ids=[7])
er_shell.gp_natural    # shape (4, 2)
er_shell.gp_weights    # shape (4,)
```

`gp_xi` stays line-element-only — it's the 1-D ξ from connectivity
``GP_X`` for force/disp-based beams. `gp_natural` is the multi-D
generalization, also populated for line elements as a shape `(n_ip, 1)`
view of `gp_xi`. `gp_weights` is catalog-driven only (line-element
Lobatto / Legendre weights aren't written to MPCO, so it's `None`
there).

`at_ip(idx)` works on any bucket with integration points — line,
plane, or solid. For numerical integration:

```python
# Volume integral of stress_11 over the parent cube (per element):
# ∫ sigma11 dV ≈ Σ sigma11_ip * weight_ip * |J|
# (|J| from element geometry — supplied by the user.)
sigma_at_ips = er.canonical("stress_11").to_numpy()   # (n_steps*n_elem, 8)
weighted = sigma_at_ips * er.gp_weights[None, :]
```

**Ordering convention.** Tensor-product schemes enumerate IPs with **ξ
varying fastest, then η, then ζ**. If your model uses a non-default
integration scheme or the OpenSees ordering happens to differ, override
the catalog entry in `utilities/gauss_points.py`.

**Unknown classes.** Element classes not in the catalog yield
`gp_natural=None`. Adding an entry is a one-line change. The library
ships with shape-function and Gauss-rule primitives for the most
common parent domains:

| Primitive | Domain | Use for |
|---|---|---|
| `gauss_legendre_1d` / `_line2_N` (internal) | bi-unit interval | beam line elements |
| `tensor_product_2d` / shell shape functions | bi-unit square | quad shells / plane elements |
| `tensor_product_3d` / brick shape functions | bi-unit cube | hex solids |
| `gauss_triangle` / `tri3_N`, `tri3_dN` | unit triangle | triangular shells / plane (e.g. `ASDShellT3`) |
| `gauss_tetrahedron` / `tet4_N`, `tet4_dN` | unit tetrahedron | linear tets (e.g. `FourNodeTetrahedron`) |

To register a new element class observed in your fixtures:

```python
from STKO_to_python.utilities.gauss_points import (
    ELEMENT_IP_CATALOG, gauss_triangle,
)
from STKO_to_python.utilities.shape_functions import (
    SHAPE_FUNCTIONS, tri3_N, tri3_dN,
)

# Suppose your MPCO file has connectivity dataset
# "204-ASDShellT3[<rule>:<cust>]" with 3 IPs per element.
ELEMENT_IP_CATALOG["204-ASDShellT3"] = {3: gauss_triangle(3)}
SHAPE_FUNCTIONS["204-ASDShellT3"] = (tri3_N, tri3_dN, "shell")
```

After this, `gp_natural`, `gp_weights`, `physical_coords()`,
`jacobian_dets()`, and `integrate_canonical()` all work for that
class without further changes.

### Physical coordinates and Jacobians

For numerical integration over the *physical* element (volume / area /
length, not just the parent domain) the library exposes shape-function-
based mapping for each catalogued class. Two methods on
`ElementResults`:

```python
phys = er.physical_coords()   # (n_elements, n_ip, 3) — physical (x, y, z) per IP
dets = er.jacobian_dets()     # (n_elements, n_ip)    — |J| per IP
```

For solids the determinant is `|det(∂x/∂ξ)|` (volume measure); for
shells it's `||∂x/∂ξ × ∂x/∂η||` (surface measure); for line elements
it's `||∂x/∂ξ||` (line measure).

**Numerical integration.** The convenience method
`integrate_canonical(name)` does the multiply-and-sum dance over IPs
and returns a Series indexed by `(element_id, step)` — same shape as
the rest of the data:

```python
# Volume integral of σ_11 over each brick element, per step
s = er.integrate_canonical("stress_11")
s.unstack("element_id").head()             # tidy step × element matrix

# Same idiom on a shell — integrates over physical surface
moments = er_shell.integrate_canonical("bending_moment_xx")
```

Internally the helper applies `value × gp_weights × |J|` over the
integration points and asserts the canonical resolves to exactly
`n_ip` columns. Closed-form buckets, missing node coords, unknown
element classes, and compressed-fiber buckets raise with a pointer
to manual integration. For full control:

```python
cols = er.canonical_columns("stress_11")
sigma_step = er.df.xs(100, level="step")[list(cols)].to_numpy()  # (n_e, n_ip)
volume_int_per_elem = (sigma_step * er.gp_weights[None, :] * dets).sum(axis=1)
```

**Inputs.** `physical_coords()` and `jacobian_dets()` rely on
``element_node_coords`` (populated automatically from the dataset's
node table at fetch time, aligned with `er.element_ids`). Both return
``None`` when any of these is missing:

- The bucket is closed-form (`gp_natural is None`), or
- Node coordinates aren't available on the dataset, or
- The element class isn't in the shape-function catalog
  (``utilities/shape_functions.py``).

The ``element_node_coords`` and ``element_node_ids`` arrays survive
pickle round-trips, so a saved `ElementResults` carries everything
needed to recompute physical coords without re-opening the MPCO file.

For the underlying convention details, see
[mpco_format_conventions.md §1, §7](mpco_format_conventions.md). The
natural↔physical conversion utilities live at
[`utilities/coords.py`](api/index.md).

## Time Snapshots

### By step index

```python
df_step = er.at_step(100)
# Returns DataFrame indexed by element_id with the named columns.
```

### By time value

```python
df_time = er.at_time(5.0)
# Finds step closest to t=5.0, returns same as at_step()
```

## Exporting to DataFrame

```python
# Flat DataFrame with time column attached
df_flat = er.to_dataframe(include_time=True)
# Adds a 'time' column mapped from step index to time value
```

## Per-element time-series statistics

For pushover and dynamic post-processing:

```python
# Per-element absolute peak across the full step history
peaks = er.peak_abs()                       # all components
peaks = er.peak_abs(component="Mz_1")       # one component

# Step index where a component peaks (per element)
idx_abs    = er.time_of_peak("Mz_1")              # by |value| (default)
idx_signed = er.time_of_peak("Mz_1", abs=False)   # by signed value

# Running min/max envelope at every step (monotonic-load workflows)
ce = er.cumulative_envelope("N_1")
# MultiIndex (element_id, step) with columns N_1_running_min/max

# One-row-per-element summary: max, min, peak_abs, residual, mean
s = er.summary()
```

## Plotting

`ElementResults.plot` is a small wrapper around matplotlib for the
three engineering views that come up most:

```python
# Time history of a component for one or more elements
ax, meta = er.plot.history("Mz_1", element_ids=[1, 2, 3])
ax, meta = er.plot.history("P_ip2", x_axis="step")  # step instead of time

# Force / moment / strain diagram along a beam (line elements only)
ax, meta = er.plot.diagram("axial_force", element_id=1, step=100)
ax, meta = er.plot.diagram("bending_moment_z", element_id=1, step=100,
                            x_in_natural=True)  # ξ ∈ [-1, +1]

# Spatial scatter for shells / planes / solids (color = component value)
ax, meta = er_shell.plot.scatter("membrane_xx", step=100)
ax, meta = er_shell.plot.scatter("membrane_xx", step=100, axes=("x", "z"))
ax, meta = er_brick.plot.scatter("stress_11", step=100)
```

Each method returns ``(ax, meta)`` — pass an existing ``ax=`` to compose
plots, or use ``meta["x"]`` / ``meta["values"]`` directly if you want to
build your own visualization. ``diagram()`` requires the result to be a
line element (``gp_dim == 1``); ``scatter()`` requires
``physical_coords()`` to resolve (so closed-form buckets and unknown
classes raise loudly).

## Pickle Serialization

Save and reload without needing the original HDF5 files:

```python
# Save
er.save_pickle("shell_forces.pkl")

# Save compressed
er.save_pickle("shell_forces.pkl.gz")

# Reload
from STKO_to_python import ElementResults
er = ElementResults.load_pickle("shell_forces.pkl")
er = ElementResults.load_pickle("shell_forces.pkl.gz")
```

Compression is auto-detected from the `.gz` extension, or forced with `compress=True/False`.

## Spatial Queries (via ds.elements)

Before extracting results, the Elements class provides spatial filtering that can narrow down which elements you query:

```python
# All elements at certain Z-levels
df = ds.elements.get_elements_at_z_levels(
    list_z=[0.0, 3.0, 6.0],
    element_type="203-ASDShellQ4",
)

# Elements from a selection set at Z-levels
df = ds.elements.get_elements_in_selection_at_z_levels(
    list_z=[0.0, 3.0],
    selection_set_name="ShearWall",
    element_type="203-ASDShellQ4",
)

# Combined: selection + Z filter, then fetch, grouped by decorated type
results_by_type = ds.elements.get_element_results_by_selection_and_z(
    results_name="globalForces",
    list_z=[0.0, 3.0],
    selection_set_name="ShearWall",
)
# Returns: {'203-ASDShellQ4[...details...]': ElementResults, ...}
```

### Discovering Available Results

```python
avail = ds.elements.get_available_element_results(
    element_type="203-ASDShellQ4",
)
# Returns: {partition_id: {result_name: [decorated_type_names]}}
```

## Full Example: Shell Wall Force Envelopes

```python
from STKO_to_python import MPCODataSet, ElementResults

# --- Load ---
ds = MPCODataSet(r"C:\results\building", "mpco", name="RC_Building")

# --- Explore available element results ---
avail = ds.elements.get_available_element_results(
    element_type="203-ASDShellQ4",
)
print(avail)

# --- Extract shell forces ---
er = ds.elements.get_element_results(
    results_name="globalForces",
    element_type="203-ASDShellQ4",
    selection_set_name="ShearWall",
    model_stage="MODEL_STAGE[2]",
)

# --- Save for reuse ---
er.save_pickle("wall_forces.pkl.gz")

# --- Later: reload ---
er = ElementResults.load_pickle("wall_forces.pkl.gz")

# --- Inspect ---
print(er)
# ElementResults: globalForces on 203-ASDShellQ4
#   Elements: 150, Steps: 2000, Components: 6

print(er.list_components())
# ('Px_1', 'Py_1', 'Pz_1', 'Mx_1', 'My_1', 'Mz_1',
#  'Px_2', 'Py_2', 'Pz_2', 'Mx_2', 'My_2', 'Mz_2')

# --- Force envelope ---
env = er.envelope()
print(env.head())
#             Px_1_min  Px_1_max  Py_1_min  Py_1_max  ...

# --- Snapshot at t=5s ---
snap = er.at_time(5.0)
print(snap.head())

# --- Time history for specific element ---
s = er.Mz_1[42]
print(s)

# --- Export with time column ---
df = er.to_dataframe(include_time=True)
df.to_csv("wall_forces_flat.csv")
```

## Full Example: Z-Level Filtering

```python
from STKO_to_python import MPCODataSet

ds = MPCODataSet(r"C:\results\building", "mpco")

# Get shell elements at base and first floor
results_by_type = ds.elements.get_element_results_by_selection_and_z(
    results_name="globalForces",
    list_z=[0.0, 3.5],
    selection_set_name="ShearWall",
)

for etype, er in results_by_type.items():
    print(f"\n{etype}: {er.n_elements} elements")
    # Pick any component name from list_components(). Example for shell
    # globalForce: er.list_components() yields ('P1','P2',...,'P24'),
    # one per node-DOF.
    comp = er.list_components()[0]
    env = er.envelope(component=comp)
    print(f"  Max {comp}: {env[f'{comp}_max'].max():.2f}")
    print(f"  Min {comp}: {env[f'{comp}_min'].min():.2f}")

    # Save each type separately
    er.save_pickle(f"forces_{etype.replace('/', '_')}.pkl.gz")
```

## Notes

- Column names come from each bucket's `META/COMPONENTS`. They reflect the actual physical components recorded by OpenSees — `Px_1` (global X-force at element node 1), `Mz_ip3` (bending moment about local z at the 4th integration point), `sigma11_f0_ip0` (axial stress in the first fiber at the first IP), etc. See [Column naming](#column-naming) for the patterns and [`docs/mpco_format_conventions.md`](mpco_format_conventions.md) for the underlying format details.
- Files written with older recorder versions that lack META fall back to `val_1, val_2, ..., val_N`. The parser raises `MpcoFormatError` if META is present but malformed (NUM_COLUMNS mismatch, non-sequential GAUSS_IDS, etc.) — see `STKO_to_python/io/meta_parser.py`.
- The `element_type` parameter uses the base type (before the bracket), e.g. `"203-ASDShellQ4"` not `"203-ASDShellQ4[some_detail]"`.
- `at_time()` uses the closest step to the requested time. If the exact time is not found, no error is raised — it picks the nearest available step.
