# Working with Elements — Comprehensive Guide

This guide walks through the full element-results workflow in
`STKO_to_python`, from discovering what's in your `.mpco` file to
slicing per-integration-point results and plotting them. It covers
both closed-form buckets (one row per element) and gauss-level /
line-station / fiber buckets (one row per element with multiple
integration-point columns).

For the on-disk format that drives all of this, see
[mpco_format_conventions.md](mpco_format_conventions.md).

---

## 1. The two layers

| Layer | Object | What it does |
|------|--------|------|
| Domain manager | `ds.elements` (`ElementManager`) | Lives on the dataset. Indexes elements, reads HDF5, applies selection sets, returns `ElementResults`. |
| Result container | `ElementResults` | Self-contained, picklable view over a result bucket. Knows its DataFrame, time array, integration-point metadata, and component names. |

```python
from STKO_to_python import MPCODataSet

ds = MPCODataSet(r"C:\path\to\results", "Recorder")  # recorder base name
er = ds.elements.get_element_results(
    results_name="material.stress",
    element_type="56-Brick",
    selection_set_name="my_solid_block",
)
```

`er` is detached — pickle it, ship it, reload it without the original
HDF5 files.

---

## 2. Discovery: what's in this dataset?

Before fetching anything, find out which element types and which
result buckets exist.

### 2.1 Element types

```python
ds.element_types          # dict — full breakdown per result group
ds.unique_element_types   # list of decorated names, e.g.
                          # ['56-Brick[1000:1]', '64-DispBeamColumn3d[2000:1]']
```

Decorated names look like `"<typeId>-<typeName>[<ruleId>:<streamIdx>]"`.
The `[…]` is the *connectivity bracket*; everything before the `[` is
the **base type** you pass to `get_element_results(element_type=…)`.

### 2.2 Element index

The full table of all elements (element_id, file_id, type, node list,
centroid) is precomputed at dataset construction:

```python
df_elem = ds.elements_info["dataframe"]
df_elem.columns
# ['element_id', 'element_idx', 'file_id', 'element_type',
#  'decorated_type', 'node_list', 'num_nodes',
#  'centroid_x', 'centroid_y', 'centroid_z']
```

This is your go-to for filtering elements by type, location, or
connectivity *without* hitting HDF5 again.

### 2.3 Available result buckets

```python
avail = ds.elements.get_available_element_results()
# {partition_id: {result_name: [decorated_types_with_that_result]}}
```

Use this to learn which `results_name` strings are valid for the
element types in your model. Common ones:

| `results_name` | Element family | Bucket shape |
|----|----|----|
| `globalForces`, `force` | beams (closed-form) | one row per element, one set of components per node |
| `localForce` | beams (closed-form) | element-local axes |
| `section.force`, `section.deformation` | force/disp-beam (line-station) | per-IP columns (`P_ip0..P_ipN`) |
| `section.fiber.stress`, `section.fiber.strain` | fiber sections | compressed multiplicity (`sigma11_f0_ip0..`) |
| `material.stress`, `material.strain` | continuum (Brick, Quad, Tet) | per-Gauss-point columns (`sigma11_ip0..ip7`) |
| `globalForces` | shells | per-node forces/moments |

---

## 3. Selecting elements

`get_element_results()` accepts three orthogonal selection inputs and
takes their **union**:

```python
er = ds.elements.get_element_results(
    results_name="globalForces",
    element_type="203-ASDShellQ4",
    element_ids=[1, 2, 3],            # explicit IDs
    selection_set_id=7,               # ID from .cdata
    selection_set_name="ShearWall",   # name from .cdata (case-insensitive)
)
```

You must pass **at least one** of these three. The resolver lives in
`STKO_to_python.selection.resolver` and can also be invoked directly:

```python
ids = ds.elements._resolve_element_ids(
    selection_set_name=["WestPier", "EastPier"],
)
```

`element_type` is always required and is the **base** type
(`"56-Brick"`, not `"56-Brick[1000:1]"`).

### 3.1 Z-level filtering

For story-level slicing (drift profiles, story shears) you can
intersect a selection with one or more horizontal planes:

```python
df_at_z = ds.elements.get_elements_in_selection_at_z_levels(
    list_z=[0.0, 3.0, 6.0, 9.0],
    selection_set_name="Columns",
    element_type="64-DispBeamColumn3d",
)

# Or fetch results in one call, grouped by decorated type:
results_by_type = ds.elements.get_element_results_by_selection_and_z(
    results_name="localForce",
    list_z=[0.0, 3.0, 6.0, 9.0],
    selection_set_name="Columns",
)
# {decorated_type: ElementResults}
```

---

## 4. The `ElementResults` container

```python
er = ds.elements.get_element_results(
    results_name="section.force",
    element_type="64-DispBeamColumn3d",
    model_stage="MODEL_STAGE[1]",
    element_ids=[1, 2, 3],
)

repr(er)
# ElementResults(results_name='section.force', element_type='64-DispBeamColumn3d',
#                n_elements=3, n_steps=120, n_components=20, n_ip=5)
```

### 4.1 Shape and metadata

| Attribute | Meaning |
|---|---|
| `er.df` | `MultiIndex(['element_id', 'step'])` × component columns |
| `er.time` | 1-D `ndarray` of times for each step |
| `er.element_ids` | sorted tuple of resolved IDs |
| `er.element_type`, `er.results_name`, `er.model_stage` | echo of inputs |
| `er.n_elements`, `er.n_steps`, `er.n_components`, `er.n_ip` | scalars |
| `er.gp_xi` | natural ξ ∈ [-1, +1] per IP, **or `None`** (see §5) |

### 4.2 Three ways to pull a column

```python
# (a) explicit fetch — full API
sub = er.fetch(component="Mz_ip2", element_ids=[1, 2])  # Series

# (b) attribute view (uses META-derived names)
view = er.Mz_ip2          # _ElementResultView
sub  = er.Mz_ip2[[1, 2]]  # Series for elements 1, 2
all_ = er.Mz_ip2[:]       # all elements

# (c) plain DataFrame slicing
sub = er.df.loc[(slice(None), slice(None)), "Mz_ip2"]
```

### 4.3 Introspection

```python
er.list_components()
# ('P_ip0', 'Mz_ip0', 'My_ip0', 'T_ip0', 'P_ip1', ...)

er.list_canonicals()
# ('axial_force', 'bending_moment_z', 'bending_moment_y', 'torsion')
```

Canonical names are engineering-friendly aliases that work across
element families; see §6.

---

## 5. Integration-point access

Per-IP data lives in suffixed columns (`_ip<k>`, `_f<f>_ip<k>` for
fibers). Whether you can convert IP indices to natural coordinates
depends on the bucket.

### 5.1 When `gp_xi` is populated (custom-rule beams)

Force-based and displacement-based beam-columns with a custom
integration rule write a `GP_X` attribute on their connectivity
dataset. The library reads it into `er.gp_xi`:

```python
er = ds.elements.get_element_results(
    results_name="section.force",
    element_type="64-DispBeamColumn3d",
    element_ids=beam_ids,
)

er.n_ip          # 5
er.gp_xi         # array([-1.0, -0.5, 0.0, 0.5, 1.0])  (5-pt Lobatto)

# Slice columns belonging to one IP
ip0_df = er.at_ip(0)
# columns: ['P_ip0', 'Mz_ip0', 'My_ip0', 'T_ip0']

# Convert natural coords to physical positions along an L-meter element
x_phys = er.physical_x(length=4.0)
```

### 5.2 When `gp_xi` is `None` (closed-form & continuum)

| Bucket | `gp_xi` | `at_ip()` | Why |
|---|---|---|---|
| Closed-form `force`, `globalForces`, `localForce` | `None` | raises `ValueError` | no IPs |
| Continuum `material.stress` on `56-Brick` etc. | `None` | raises `ValueError` | no `GP_X` attribute on continuum connectivity |
| Custom-rule beam `section.force` / `section.fiber.stress` | array | works | `GP_X` is written |

For continuum (Brick / Quad / Tet) the columns are still **per-IP and
correctly named** — you just can't get the natural ξ-coordinates from
the file. Slice by suffix manually:

```python
er = ds.elements.get_element_results(
    results_name="material.stress",
    element_type="56-Brick",
    selection_set_name="solid_block",
)

ip0_cols = er.df.filter(regex=r"_ip0$")           # all stresses at IP 0
sigma11  = er.df.filter(regex=r"^sigma11_ip\d+$") # σ₁₁ at every IP
```

### 5.3 Fiber sections

Fiber buckets (`section.fiber.stress` on a `64-DispBeamColumn3d`)
double-suffix — `<short>_f<fiberIdx>_ip<ipIdx>`. `er.at_ip(k)` returns
**all fibers** at that IP:

```python
er = ds.elements.get_element_results(
    results_name="section.fiber.stress",
    element_type="64-DispBeamColumn3d",
    element_ids=beam_ids,
)
# columns like: 'sigma11_f0_ip0', 'sigma11_f1_ip0', ...

ip0 = er.at_ip(0)
# columns: ['sigma11_f0_ip0', 'sigma11_f1_ip0', ..., 'sigma11_fN_ip0']
```

---

## 6. Canonical (engineering) names

`ElementResults` exposes a small map from engineering names to the
MPCO shortnames that vary by element family:

```python
er.list_canonicals()
# ('axial_force', 'bending_moment_z', ...)

cols = er.canonical_columns("axial_force")
# Force/disp beam line-stations:  ('P_ip0', 'P_ip1', ..., 'P_ip4')
# Beam localForce (closed-form):  ('N_1', 'N_2')

axial = er.canonical("axial_force")     # DataFrame of those columns
```

The full map lives in [`STKO_to_python/elements/canonical.py`](../src/STKO_to_python/elements/canonical.py)
and covers axial force, bending moments, torsion, shears, beam section
deformations, shell resultants, continuum stresses/strains, damage,
plasticity, and global-axis nodal forces.

Canonical lookup is a thin filter — for full control use
`er.df` and the column-suffix conventions directly.

---

## 7. Time-domain operations

```python
er.at_step(42)                # DataFrame indexed by element_id at step 42
er.at_time(2.35)              # closest step to t=2.35 s
er.envelope()                 # min/max per element across steps
er.envelope("Mz_ip2")         # envelope for one component
er.to_dataframe(include_time=True)  # flat, with 'time' column attached
```

Envelope output:

```
                Mz_ip2_min   Mz_ip2_max
element_id
1               -1.34e+05    +1.21e+05
2               -2.05e+05    +1.95e+05
...
```

---

## 8. Plotting

`ElementResults` has a `.plot` facade mirroring the nodal one:

```python
ax, meta = er.plot.xy(
    y_component="Mz_ip2",
    y_operation="Max",          # Max / Min / AbsMax / Sum / Mean / per-element
    x_results_name="TIME",      # or another component for X
)
```

Or skip the container and let the dataset facade pick the columns:

```python
ax, meta = ds.plot.xy(
    model_stage="MODEL_STAGE[1]",
    results_name="section.force",
    element_type="64-DispBeamColumn3d",
    element_ids=[1, 2, 3],
    y_component="Mz_ip2",
    y_operation="Max",
    x_results_name="TIME",
)
```

See [`docs/api/plotting.md`](api/plotting.md) for the full plot API.

---

## 9. Aggregation across many results

For multi-case studies (e.g. 11 ground motions, same model), wrap
fetched containers in `MPCOResults` to vectorize across cases:

```python
from STKO_to_python.MPCOList.MPCOResults import MPCOResults

cases = MPCOResults({
    "GM01": ds01.elements.get_element_results(...),
    "GM02": ds02.elements.get_element_results(...),
    ...
})

cases.envelope("Mz_ip2")           # max|.| per element across all cases
cases.df_long(component="Mz_ip2")  # tidy DataFrame for seaborn / plotnine
```

---

## 10. Pickling

```python
er.save_pickle("frame_force.pkl")
er.save_pickle("frame_force.pkl.gz")  # gzip auto-detected by suffix

from STKO_to_python.elements.element_results import ElementResults
er2 = ElementResults.load_pickle("frame_force.pkl")
```

`gp_xi` survives the round-trip; the attribute-view proxies are
rebuilt on load.

---

## 11. Multi-partition (MP) datasets

When OpenSees-MP writes one `.mpco` per process
(`results.part-0.mpco`, `results.part-1.mpco`, …), the dataset
auto-discovers all parts. Element IDs are global; fetching is
transparent across partitions. The `file_id` column in
`ds.elements_info["dataframe"]` tells you which partition owns each
element.

The QuadFrame example under
`stko_results_examples/QuadFrame_results/` exercises this path.

---

## 12. Heterogeneous integration rules

If a single element class in your model uses **different** integration
rules in different element instances (e.g., some Lobatto-5 columns and
some Lobatto-3 columns), they live in separate decorated buckets
(`64-DispBeamColumn3d[1000:1]` vs `64-DispBeamColumn3d[1001:1]`).

`get_element_results()` will refuse to silently merge them and raise
`MpcoFormatError`. Query each decorated bucket explicitly:

```python
er_a = ds.elements.get_element_results(
    results_name="section.force",
    element_type="64-DispBeamColumn3d[1000:1]",  # decorated
    element_ids=ids_a,
)
er_b = ds.elements.get_element_results(
    results_name="section.force",
    element_type="64-DispBeamColumn3d[1001:1]",
    element_ids=ids_b,
)
```

See [mpco_format_conventions.md §4](mpco_format_conventions.md) for
the underlying axis.

---

## 13. Reference: API cheat sheet

### `ds.elements` (ElementManager)

| Method | Purpose |
|---|---|
| `get_element_results(results_name, element_type, *, element_ids=None, selection_set_id=None, selection_set_name=None, model_stage=None, verbose=False)` | Fetch results for one bucket |
| `get_available_element_results(element_type=None)` | List buckets per partition |
| `get_elements_at_z_levels(list_z, element_type=None)` | Filter by Z planes |
| `get_elements_in_selection_at_z_levels(list_z, *, selection_set_id=None, selection_set_name=None, element_ids=None, element_type=None)` | Selection ∩ Z |
| `get_element_results_by_selection_and_z(results_name, list_z, *, …)` | Filter + fetch in one call |

### `ElementResults`

| Member | Returns |
|---|---|
| `df` | results DataFrame, `MultiIndex(['element_id', 'step'])` |
| `time` | `ndarray` of step times |
| `element_ids`, `element_type`, `results_name`, `model_stage` | metadata |
| `gp_xi`, `n_ip` | natural ξ array & count (or `None` / 0) |
| `list_components()`, `list_canonicals()`, `canonical_columns(name)`, `canonical(name)` | introspection |
| `fetch(component=None, *, element_ids=None)` | flexible getter |
| `at_step(step)`, `at_time(t)` | snapshots |
| `at_ip(ip_idx)` | per-IP slice (raises if `gp_xi is None`) |
| `physical_x(length)` | natural ξ → physical x |
| `envelope(component=None)` | min/max across steps |
| `to_dataframe(include_time=True)` | flat output |
| `plot` | XY plot facade |
| `save_pickle(path)`, `load_pickle(path)` | persistence |

---

## 14. Worked end-to-end example

```python
from STKO_to_python import MPCODataSet

ds = MPCODataSet(r"./stko_results_examples/elasticFrame/results", "results")

# 1. Discover
print(ds.unique_element_types)
print(ds.elements.get_available_element_results())

# 2. Pick a bucket and a selection
er = ds.elements.get_element_results(
    results_name="section.force",
    element_type="64-DispBeamColumn3d",
    selection_set_name="ground_floor_columns",
    model_stage="MODEL_STAGE[1]",
)

# 3. Inspect
print(er)                     # ElementResults(...)
print(er.list_components())   # ('P_ip0', 'Mz_ip0', ...)
print(er.gp_xi)               # array([-1, -.5, 0, .5, 1])

# 4. Engineering view
moments = er.canonical("bending_moment_z")  # all Mz_ip<k>
peak = moments.abs().groupby("element_id").max()

# 5. Per-IP at a target time
mid = er.at_time(2.0)
mid_at_base = er.at_ip(0).xs(slice(None), level="step", drop_level=False)

# 6. Plot peak base moment vs time, max across columns
ax, meta = er.plot.xy(
    y_component="Mz_ip0",
    y_operation="AbsMax",
    x_results_name="TIME",
)

# 7. Persist
er.save_pickle("base_section_force.pkl.gz")
```

---

## See also

- [ElementResults.md](ElementResults.md) — focused container reference
- [mpco_format_conventions.md](mpco_format_conventions.md) — on-disk format
- [api/plotting.md](api/plotting.md) — plot facade details
- [api/mpco-results.md](api/mpco-results.md) — multi-case aggregation
- [`examples/usage_tour.py`](../examples/usage_tour.py) — runnable end-to-end tour
