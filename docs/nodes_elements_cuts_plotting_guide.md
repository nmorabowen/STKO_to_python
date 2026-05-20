# Nodes, Elements, Section Cuts & Plotting — Usage Guide

A practical, copy-pasteable guide to the four workhorses of `STKO_to_python`:

1. **Nodes** — nodal result extraction (`NodalResults`)
2. **Elements** — element result extraction (`ElementResults`)
3. **Section cuts** — integrate internal forces over a plane (`SectionCut`, `SectionSweep`)
4. **Plotting** — every `.plot` surface in the library

Every signature below is taken from the current source. Snippets assume:

```python
from STKO_to_python import MPCODataSet

ds = MPCODataSet(
    hdf5_directory="path/to/results",   # folder with results.part-*.mpco
    recorder_name="results",            # base name (no .part-N.mpco)
    name="MyModel",                     # optional; defaults to folder name
    verbose=False,                      # True → INFO logging + print_summary
)
```

`MPCODataSet` is also a context manager — use `with MPCODataSet(...) as ds:`
for scripts that open many datasets in sequence (closes pooled HDF5 handles
and drops query caches at scope exit).

Discover what's in the file first:

```python
ds.model_stages              # ['MODEL_STAGE[1]', ...]
ds.node_results_names        # ['DISPLACEMENT', 'REACTION_FORCE', ...]
ds.element_results_names     # ['force', 'section.force', ...]
ds.unique_element_types      # ['5-ElasticBeam3d', '203-ASDShellQ4', ...]
ds.number_of_steps           # {stage: int}
ds.time                      # DataFrame, MultiIndex (MODEL_STAGE, STEP)
ds.nodes_info["dataframe"]   # node_id, file_id, index, x, y, z
ds.elements_info["dataframe"]# element_id, element_type, node_list, centroid_*
ds.selection_set             # {set_id: {SET_NAME, NODES, ELEMENTS}} (lazy)

# Logging-based summaries (need verbose=True or logging.basicConfig(INFO)):
ds.print_summary()
ds.print_selection_set_info()
ds.elements.get_available_element_results()   # {part: {result: [types]}}
```

---

## 1. Nodes

### 1.1 Fetching nodal results

```python
nr = ds.nodes.get_nodal_results(
    results_name="DISPLACEMENT",     # str, list[str], or None = ALL results
    model_stage="MODEL_STAGE[1]",    # str, list[str], or None = ALL stages
    node_ids=[1, 2, 3, 4],           # int / list / nested list / ndarray
    selection_set_id=None,           # int or list[int]
    selection_set_name=None,         # str or list[str] (case-insensitive)
    selector=None,                   # a NodeSelector (see 1.4)
)
```

All arguments are keyword-only. Node sources are **unioned** (explicit
`node_ids` + selection set + selector all merge). Omitting every filter
returns **all nodes**.

> **Stage default differs from elements.** For nodes, `model_stage=None`
> concatenates **all** stages onto one contiguous global step axis. For
> elements, `model_stage=None` is the **first stage only**. Be explicit if
> it matters.

Pass a list of result names to read several results in one HDF5 pass:

```python
nr = ds.nodes.get_nodal_results(
    results_name=["DISPLACEMENT", "ACCELERATION"],
    model_stage="MODEL_STAGE[1]",
)
```

### 1.2 The `NodalResults` container

Index is `(node_id, step)`; columns are a MultiIndex `(result, component)`.
Components are **1-indexed** (OpenSees convention: 1=X, 2=Y, 3=Z).

```python
# Dynamic attribute views — the idiomatic accessor:
nr.DISPLACEMENT[1]              # comp 1, all nodes        -> Series
nr.DISPLACEMENT[1, [14, 25]]    # comp 1, nodes 14 & 25    -> Series/DataFrame
nr.DISPLACEMENT[:]              # all comps, all nodes     -> DataFrame
nr.DISPLACEMENT[:, [14, 25]]    # all comps, nodes 14 & 25 -> DataFrame

# Explicit fetch (same engine, more options):
nr.fetch(result_name="DISPLACEMENT", component=1, node_ids=[14, 25])
nr.fetch(result_name="DISPLACEMENT", component=1,
         selection_set_name="Roof")
nr.fetch_nearest(points=[(0.5, 0.0, 30.0)], result_name="DISPLACEMENT",
                 component=1)

# Introspection:
nr.list_results()                      # ('ACCELERATION', 'DISPLACEMENT')
nr.list_components("DISPLACEMENT")     # ('1', '2', '3')
nr.df                                  # the raw DataFrame
nr.time                                # 1-D ndarray, contiguous across stages

# .info sub-object:
nr.info.nodes_ids
nr.info.model_stages
nr.info.stage_step_ranges              # {stage: (start, end)} global steps
nr.info.nearest_node_id([(0.5, 0.0, 2.0)])

# Pickle (gzip auto-detected from .gz suffix):
nr.save_pickle("disp.pkl.gz")
from STKO_to_python import NodalResults
nr = NodalResults.load_pickle("disp.pkl.gz")
```

### 1.3 Engineering aggregations on `NodalResults`

These forward to a shared `AggregationEngine`. **All keyword-only.**

```python
# Relative drift between two nodes (Series indexed by step):
nr.drift(top=4, bottom=1, component=1)
nr.drift(top=4, bottom=1, component=1, reduce="abs_max")    # -> float
nr.delta_u(top=4, bottom=1, component=1)                     # raw Δ (no /h)

# Residual (end-of-record) drift, averaged over the last `tail` steps:
nr.residual_drift(top=4, bottom=1, component=1, tail=3, agg="mean")

# Story-by-story envelopes (auto-groups nodes by Z within dz_tol):
nr.interstory_drift_envelope(component=1, selection_set_name="Frame",
                             dz_tol=1e-3)
nr.story_pga_envelope(component=1, result_name="ACCELERATION",
                      to_g=True, g_value=9810)

# Plan/torsion/rocking:
nr.roof_torsion(node_a_id=101, node_b_id=140)
nr.base_rocking(node_coords_xy=[(0,0),(10,0),(0,10)], z_coord=0.0)
nr.asce_torsional_irregularity(component=1,
    side_a_top=(0,0,30), side_a_bottom=(0,0,0),
    side_b_top=(20,0,30), side_b_bottom=(20,0,0))

# Orbit (two-component displacement trajectory):
sx, sy = nr.orbit(node_ids=1, x_component=1, y_component=2)
```

### 1.4 Chainable node selector

`ds.nodes.select()` returns a lazy `NodeSelector`. Chain spatial primitives
(each one AND-narrows), combine with `& | ~`, resolve with `.ids()` /
`.mask()` / `.df()` / `.count()`:

```python
ids = (ds.nodes.select()
       .from_selection("Roof")                       # anchor (req. for ~)
       .within_box(min=(0, 0, 28), max=(50, 50, 32))
       .nearest_to((25, 25, 30), k=8)
       .ids())

# Other primitives: .with_ids(...), .on_plane(...), .near_line(...),
# .within_distance(point, radius), .coord_in(...), .at_level(z),
# .attached_to(element_ids), .where(lambda df: ...)

# Feed straight into a fetch (unions with node_ids if both given):
nr = ds.nodes.get_nodal_results(
    results_name="DISPLACEMENT", model_stage="MODEL_STAGE[1]",
    selector=ds.nodes.select().at_level(30.0),
)
```

### 1.5 Threshold / time-window masks

```python
mask = (nr.where(time=(0.0, 10.0))
          .component("DISPLACEMENT", 1).abs_peak().gt(0.05))
hot_nodes = nr[mask]      # a trimmed NodalResults
ids       = mask.ids()    # int64 ndarray

# magnitude across components:
nr.where().magnitude("DISPLACEMENT").peak().gt(0.05)
```

---

## 2. Elements

### 2.1 Fetching element results

```python
er = ds.elements.get_element_results(
    "force",                        # results_name (positional)
    element_type="5-ElasticBeam3d", # REQUIRED (base type, not decorated)
    element_ids=[1, 2, 3],          # or selection_set_id / _name / selector
    selection_set_id=None,
    selection_set_name=None,
    selector=None,
    model_stage="MODEL_STAGE[1]",   # None = FIRST stage only (see note in 1.1)
    verbose=False,
)
```

`element_type` is mandatory unless you pass a `selector` with `.of_type(...)`.
Use the **base** type (`"5-ElasticBeam3d"`), not the decorated bracket form.
Discover valid names:

```python
ds.elements.get_available_element_results()
ds.elements.get_available_element_results(element_type="5-ElasticBeam3d")
```

### 2.2 The `ElementResults` container

Index is `(element_id, step)`; columns are real engineering names parsed
from the bucket's `META` (e.g. `Pz_1`, `Mz_ip3`, `sigma11_f0_ip0`) — only
falling back to `val_1, val_2, ...` when no `META` is present.

```python
# Dynamic attribute views:
er.Pz_1               # all elements -> _ElementResultView
er.Pz_1[[1, 2]]       # elements 1 & 2 -> Series
er.Pz_1[:]            # all elements -> Series

# Explicit fetch:
er.fetch(component="Pz_1", element_ids=[1, 2])

# Introspection:
er.list_components()          # ('Px_1', 'Py_1', ..., 'Mz_2')
er.n_elements, er.n_steps, er.n_components
er.df                         # raw DataFrame

# Per-element time-series stats:
er.envelope(component="Pz_1")     # min/max over steps, per element
er.peak_abs("Mz_1")               # max(|.|) per element
er.time_of_peak("Mz_1", abs=True) # argmax step per element
er.cumulative_envelope("Mz_1")    # running min/max per (element, step)
er.summary()                      # max/min/peak_abs/residual/mean per element

# Snapshots:
er.at_step(5)                     # DataFrame indexed by element_id
er.at_time(0.5)                   # nearest recorded step
er.to_dataframe(include_time=True)

# Pickle:
er.save_pickle("forces.pkl.gz")
from STKO_to_python import ElementResults
er = ElementResults.load_pickle("forces.pkl.gz")
```

### 2.3 Canonical engineering names

Map element-class-specific column names to portable engineering names:

```python
er.list_canonicals()                      # ('axial_force', 'bending_moment_z', ...)
er.canonical_columns("bending_moment_z")  # ('Mz_1', 'Mz_2') in on-disk order
er.canonical("bending_moment_z")          # DataFrame subset (raises if no match)
```

### 2.4 Integration points & physical coordinates

For line-station / Gauss-level buckets (`section.force`, fiber stress, …):

```python
er.gp_xi          # natural ξ ∈ [-1,1], 1-D (line elements) — None if closed-form
er.gp_natural     # (n_ip, dim): dim 1=line, 2=shell, 3=solid
er.n_ip, er.gp_dim
er.at_ip(0)                    # columns for IP 0 -> DataFrame
er.physical_x(length=3.0)      # ξ -> physical position along a beam of length L

# Shells / solids (catalog-driven, needs node coords):
er.physical_coords()           # (n_elements, n_ip, 3) or None
er.jacobian_dets()             # (n_elements, n_ip) or None
er.integrate_canonical("stress_11")   # ∫ value dΩ per (element, step) -> Series
```

Closed-form buckets (e.g. `force` on `ElasticBeam3d`) have `gp_xi is None`,
`n_ip == 0`; `at_ip()` / `physical_x()` / `integrate_canonical()` raise with
a clear message.

### 2.5 Z-level filtering & selectors

```python
# Elements crossing horizontal planes:
ds.elements.get_elements_at_z_levels([0.0, 3.5, 7.0])
ds.elements.get_elements_in_selection_at_z_levels(
    [0.0, 3.5], selection_set_name="Columns")

# Results filtered by selection + Z (grouped by decorated type):
ds.elements.get_element_results_by_selection_and_z(
    "force", [0.0], selection_set_id=2)

# Chainable element selector:
ids = (ds.elements.select()
       .of_type("DispBeamColumn3d")
       .within_box(min=(0,0,0), max=(10,10,30))
       .nearest_to((5,5,15), k=20)
       .ids())
er = ds.elements.get_element_results(
    "force", selector=ds.elements.select().of_type("5-ElasticBeam3d"))
```

Element masks mirror the node side:

```python
mask = er.where(time=(0.0, 10.0)).component("Mz_ip0").abs_peak().gt(50.0)
hot  = er[mask]
ids  = mask.ids()
```

---

## 3. Section cuts

A section cut slices the model with a plane and integrates internal
tractions over the elements it crosses, recovering a `(F, M)` resultant
time series. Beams, shells, and solids are all composed automatically.

```python
from STKO_to_python import Plane, SectionCut, SectionCutSpec, SectionSweep, DriftSpec
```

### 3.1 Defining the plane

```python
Plane.horizontal(z=2500.0)                   # ⟂ global Z at elevation z
Plane.vertical(axis="x", at=10.0)            # ⟂ global X at x = 10
Plane.from_three_points(p1, p2, p3, normal_hint=(0,0,1))
Plane.horizontal_grid([0.0, 3.0, 6.0])       # list of Planes (for sweeps)
Plane(point=(0,0,5), normal=(0,0,1))         # general; normal auto-normalized
```

### 3.2 Computing a cut

The inline form (one-shot) needs `plane`, `model_stage`, and **one element
filter** (`element_ids`, `selection_set_id`, or `selection_set_name`):

```python
cut = ds.section_cut(
    plane=Plane.horizontal(z=2500.0),
    element_ids=shell_ids,            # or selection_set_name="Wall"
    model_stage="MODEL_STAGE[1]",     # REQUIRED keyword
    side="positive",                  # "positive" | "negative" (sign convention)
    label="Wall @ z=2500",            # optional, used by plotters
    bounding_polygon=None,            # optional convex clip (see 3.5)
)
```

The reusable form takes a picklable `SectionCutSpec`:

```python
spec = SectionCutSpec(plane=Plane.horizontal(z=2500.0),
                       selection_set_name="Wall", label="Wall")
cut  = ds.section_cut(spec=spec, model_stage="MODEL_STAGE[1]")
spec.save_pickle("wall_cut.spec.pkl")
```

### 3.3 Reading the result

```python
cut.F            # (n_steps, 3) force the kept side exerts on the discarded side
cut.M            # (n_steps, 3) moment about cut.centroid
cut.time         # (n_steps,)
cut.centroid     # (3,) reference point for M
cut.n_steps

cut.at_step(10)             # 6-element Series Fx,Fy,Fz,Mx,My,Mz
cut.at_time(0.5)            # nearest step
cut.envelope()              # per-component max/min/peak_abs + when
cut.to_dataframe()          # rows=time, cols=Fx..Mz
cut.resultant()             # (F.copy(), M.copy())
cut.moment_about((0,0,0))   # transfer M to another reference point

# Which elements contributed:
cut.intersections           # beams
cut.shell_intersections     # shells
cut.solid_intersections     # solids
cut.per_beam_F, cut.per_shell_F, cut.per_solid_F   # {element_id: (n,3)}
cut.contributing_element_ids

cut.save_pickle("wall_cut.pkl.gz")          # no live dataset reference
```

### 3.4 Validators (no support reactions required)

```python
ok, residual = cut.consistency_check(ds)        # Newton 3rd law: side flip ≈ 0
ok, residual = cut.compare_to(other_cut)        # two parallel cuts agree
```

`consistency_check` is free and works for DRM / PML / explicit dynamics —
it recomputes the cut with the opposite side and verifies the sum is ~0.

### 3.5 Bounding polygon (sub-region cuts)

When selection sets don't pre-filter to the region of interest, restrict
the cut to a **convex** polygon lying on the plane:

```python
right_half = ((mid_x, -big, z), (big, -big, z), (big, big, z), (mid_x, big, z))
cut = ds.section_cut(plane=Plane.horizontal(z=z), element_ids=all_ids,
                      model_stage="MODEL_STAGE[1]", bounding_polygon=right_half)
```

Non-convex / off-plane / degenerate polygons raise at construction.

### 3.6 Layered-shell decomposition

For `LayeredShell` sections (needs a `sections.tcl` next to the data,
exposed as `ds.layered_sections`):

```python
n_layers = len(ds.layered_sections[section_id])

# Per-layer contribution (shell-only; beams/solids dropped):
layer0 = cut.per_layer_force(0, ds)
# sum over all layers recovers the shell portion of the standard cut.

# Inline shortcuts:
top = ds.section_cut(plane=..., element_ids=..., model_stage=...,
                      per_layer=n_layers - 1)
fib = ds.section_cut(plane=..., element_ids=..., model_stage=...,
                      per_layer=2, per_fiber=0)   # per_fiber requires per_layer
```

### 3.7 Section sweep (many parallel planes)

The "story shear vs elevation" / depth-profile pattern:

```python
sweep = ds.section_sweep(
    planes=Plane.horizontal_grid([0, 3, 6, 9, 12]),
    selection_set_name="Frame",
    model_stage="MODEL_STAGE[1]",
)

sweep.envelope()                 # one row per plane, 18 cols (6 comp × 3 stat)
sweep.to_dataframe("Fx")         # rows=time, cols=plane index (heatmap source)
sweep.plane_locators("z")        # elevation of each plane (axis auto-inferred)
sweep[0]                         # the SectionCut at plane 0 (indexable/iterable)

# Per-plane filters → build specs and use from_specs:
SectionSweep.from_specs([spec_lvl1, spec_lvl2], ds, model_stage="MODEL_STAGE[1]")
```

---

## 4. Plotting

Every plot method returns `(ax_or_fig, meta)` where `meta` is a dict of the
parameters that drove the plot. Pass `ax=` to compose onto an existing axes.

### 4.1 Nodal X–Y plots — `nr.plot.xy` / `ds.plot.xy`

> The current API is `.plot.xy(...)`. There is no `plot_roof_drift` /
> `plot_story_drifts` / `dataset.plot.nodes.*` — use `xy` + the engineering
> aggregations from §1.3, or `plot_TH` for raw time histories.

```python
nr = ds.nodes.get_nodal_results(results_name="DISPLACEMENT",
                                 model_stage="MODEL_STAGE[1]")

# Per-result: displacement (comp 1, max over nodes) vs time
ax, meta = nr.plot.xy(
    y_results_name="DISPLACEMENT", y_direction=1, y_operation="Max",
    x_results_name="TIME",          # "TIME" | "STEP" | another result name
)

# Force–displacement (reaction sum vs roof displacement) in one shot:
ax, meta = ds.plot.xy(
    model_stage="MODEL_STAGE[1]",
    results_name="REACTION_FORCE",
    selection_set_name="Base",
    y_direction=1, y_operation="Sum",
    x_results_name="DISPLACEMENT",  # triggers a second fetch
    x_direction=1, x_operation="Mean",
)
```

`y_operation` / `x_operation` accept `Aggregator` ops — `"Sum"`, `"Mean"`,
`"Max"`, `"Min"`, `"Std"`, `"Percentile"` (pass
`operation_kwargs={"percentile": 95}`), `"Envelope"`, `"Cumulative"`,
`"SignedCumulative"`, `"RunningEnvelope"` — **or** `"All"` / `"Raw"` to draw
one curve per node (no aggregation; x must be `"TIME"` or `"STEP"`).
For multi-stage results, stage boundaries are drawn as dashed lines and
returned in `meta["stage_boundaries"]`.

Raw time-history helper (one curve per node, optional per-node subplots):

```python
fig, meta = nr.plot.plot_TH(
    result_name="DISPLACEMENT", component=2,
    node_ids=[101, 120, 140],
    split_subplots=True, sharey=True, figsize=(8, 3),
)
```

### 4.2 Element plots — `er.plot`

```python
er = ds.elements.get_element_results("section.force",
        element_type="5-ElasticBeam3d", model_stage="MODEL_STAGE[1]")

# Time history of a single raw column for one/many elements:
er.plot.history("Mz_ip0", element_ids=[1, 2, 3], x_axis="time")

# Time history of a canonical name, with fiber/IP/layer reduction.
# Useful for section.fiber.* buckets that expand to n_fibers * n_ip cols.
er.plot.history_canonical("strain_11", reduce="mean")              # over all
er.plot.history_canonical("strain_11", over="fibers", ip_idx=0,
                           reduce="abs_max")                        # fibers @ IP 0
er.plot.history_canonical("stress_11", over="ips", fiber_idx=0,
                           reduce="max", element_ids=[1, 2, 3])     # IPs of fiber 0

# Beam diagram (line elements only): component vs position along element:
er.plot.diagram("bending_moment_z", element_id=1, step=10)
er.plot.diagram("bending_moment_z", element_id=1, step=10, x_in_natural=True)

# Shell/solid IP scatter colored by value at a step (lightweight contour):
er.plot.scatter("membrane_xx", step=10, axes=("x", "y"))
```

`history_canonical` accepts `over ∈ {"all", "fibers", "ips", "layers"}`
and `reduce ∈ {"mean", "median", "max", "min", "abs_max", "sum"}`. When
`over` is anything but `"all"` it requires the partner anchors (e.g.
`over="fibers"` needs `ip_idx=`) so the reduction collapses one axis at
fixed coordinates on the others. `meta["columns_used"]` lists which raw
columns were folded in.

### 4.3 Model / mesh visualization — `ds.plot`

```python
ds.plot.undeformed_shape()
ds.plot.deformed_shape(model_stage="MODEL_STAGE[1]", step=peak, scale=200.0)

# Wireframe backdrop, then overlay an IP scatter on the same axes:
ax, _ = ds.plot.mesh(element_type="203-ASDShellQ4")
er.plot.scatter("membrane_xx", step=10, ax=ax)
# …or the bundled convenience:
ds.plot.mesh_with_contour(er, "membrane_xx", step=10)

# Beams as 3-D extruded section solids (uses the .cdata profile sidecar):
ds.plot.beam_solids(selection_set_name="Columns", edge_color=None)
ds.plot.beam_solids_deformed(model_stage="MODEL_STAGE[1]", step=peak, scale=200.0)
```

### 4.4 Section-cut plots — `cut.plot`

```python
cut.plot.time_history("Fx")                     # one component vs time
cut.plot.orbit("Fx", "Fy")                      # lateral force orbit
cut.plot.envelope_bars(show_minmax=True)        # peak per component
cut.plot.consistency_residual(ds)               # Newton-3rd diagnostic

# Capacity / hysteresis: cut force vs node-pair drift
drift = DriftSpec(top_node=140, bottom_node=1, component=1,
                  normalize_by=30000.0, label="roof drift ratio")
cut.plot.hysteresis("Fx", drift, ds)
```

### 4.5 Sweep plots — `sweep.plot`

```python
# Story-shear-vs-elevation profile (locator on the y-axis by default):
sweep.plot.profile("Fx", agg="peak_abs")        # agg: max | min | peak_abs
sweep.plot.profile("Fx", agg="max", vertical=False)

# Time × plane heatmap (diverging colormap, sign-aware):
sweep.plot.heatmap("Fx")
```

### 4.6 Multi-case cut plots — `MultiCutResult.plot`

When a `SectionCutSpec` is applied across an `MPCOResults` ensemble:

```python
multi.plot.overlay_time_history("Fx")           # one trace per case
multi.plot.case_envelope_bars("Fx", agg="peak_abs")
multi.plot.case_scatter("Fx", "Fy", agg="peak_abs")   # biaxial demand
```

---

## 5. Gotchas & quick reference

| Topic | Watch out for |
|---|---|
| Stage default | `nodes` → all stages; `elements` → first stage only when `model_stage=None`. |
| Components | 1-indexed (1=X, 2=Y, 3=Z), matching OpenSees. |
| Element type | Use the **base** type (`"5-ElasticBeam3d"`); `element_type` is required. |
| Closed-form buckets | `gp_xi is None`, `n_ip == 0`; IP/integration methods raise (by design). |
| Plotting API | It's `nr.plot.xy(...)` / `ds.plot.xy(...)`. No `plot_roof_drift`-style methods or `ds.plot.nodes`. |
| Section cut filter | Inline form needs a plane **and** an element filter; `model_stage` is a required keyword. |
| `bounding_polygon` | Must be convex and lie on the cut plane (validated at construction). |
| `per_fiber` | Requires `per_layer`; only for fiber-decomposed layered shells. |
| Selection sets | `ds.selection_set` is parsed lazily on first access; `print_selection_set_info()` lists them. |
| Heterogeneous layouts | Multi-stage / mixed integration-rule fetches raise `MpcoFormatError` rather than silently NaN-padding — query stages/buckets separately. |
| Pickle | `.gz` suffix auto-enables gzip for every `save_pickle` in the library. |

### Public API

```python
from STKO_to_python import (
    MPCODataSet,                                    # entry point
    NodalResults, ElementResults,                   # result containers
    Plane, SectionCut, SectionCutSpec,              # section cuts
    SectionSweep, MultiCutResult, DriftSpec,
    Aggregator, MPCOResults, MPCO_df,               # aggregation / multi-case
    HDF5Utils, H5RepairTool, AttrDict, PlotSettings,
)
```

### Runnable examples in this repo

| File | Covers |
|---|---|
| `examples/usage_tour.py` | End-to-end tour (intro → fetch → aggregate → plot → pickle). |
| `examples/elastic_frame_example.py` | Nodes + closed-form beam elements. |
| `examples/quad_frame_shell_example.py` | Shell elements + multi-partition. |
| `examples/section_cut_solid_and_layered_example.py` | Solid + layered-shell section cuts and sweeps. |
| `examples/solid_mixed_example.py` | Mixed beam/solid models. |
