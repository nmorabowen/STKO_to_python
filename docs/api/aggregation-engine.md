# AggregationEngine

Stateless engineering-aggregation engine. Holds every drift / envelope /
rocking / torsion / orbit calculation that `NodalResults` exposes as a
forwarder. One shared instance lives at
`NodalResults._aggregation_engine`; callers normally invoke the thin
wrappers on `NodalResults` instead (e.g. `nr.drift(...)` rather than
`engine.drift(nr, ...)`).

The engine holds **no dataset state** and is pickle-safe (not persisted
with `NodalResults`).

---

## Node selection convention

Several methods accept exactly **one** of these four node-selection inputs:

| Parameter | Type | Description |
|---|---|---|
| `node_ids` | `list[int]` | Explicit node IDs |
| `selection_set_id` | `int \| list[int]` | ID from `.cdata` |
| `selection_set_name` | `str \| list[str]` | Name from `.cdata` |
| `coordinates` | `list[list[float]]` | (x,y) or (x,y,z) resolved to nearest nodes |

Providing more than one raises `ValueError`.

---

## Pair-wise utilities

### `delta_u`

```python
nr.delta_u(
    top,           # node ID (int) or coordinates (x,y) / (x,y,z)
    bottom,        # node ID (int) or coordinates
    component,     # result component (int or str)
    result_name="DISPLACEMENT",
    stage=None,    # required for multi-stage results
    signed=True,
    reduce="series",  # "series" → pd.Series | "abs_max" → float
)
```

Relative displacement `u_top(t) - u_bottom(t)`. Returns a `pd.Series`
(one value per step) when `reduce="series"`, or a single `float` when
`reduce="abs_max"`.

```python
du = nr.delta_u(top=42, bottom=10, component=1)
peak = nr.delta_u(top=42, bottom=10, component=1, reduce="abs_max")
```

---

### `drift`

```python
nr.drift(
    top,
    bottom,
    component,
    result_name="DISPLACEMENT",
    stage=None,
    signed=True,
    reduce="series",  # "series" → pd.Series | "abs_max" → float
)
```

Interstory drift ratio between two nodes:

```
drift(t) = (u_top(t) - u_bottom(t)) / (z_top - z_bottom)
```

Raises `ValueError` if `z_top == z_bottom`.

```python
dr = nr.drift(top=42, bottom=10, component=1)
peak = nr.drift(top=42, bottom=10, component=1, reduce="abs_max")

# Coordinates instead of node IDs
dr = nr.drift(top=(10.0, 0.0, 6.0), bottom=(10.0, 0.0, 3.0), component=1)
```

---

### `residual_drift`

```python
nr.residual_drift(
    top,
    bottom,
    component,
    result_name="DISPLACEMENT",
    stage=None,
    signed=True,
    tail=1,       # number of end-of-record steps to average
    agg="mean",   # "mean" or "median" over the tail window
)
```

Returns a single `float` — the drift ratio averaged over the last `tail`
steps. Increasing `tail` reduces end-of-record noise.

```python
rd = nr.residual_drift(top=42, bottom=10, component=1, tail=5, agg="mean")
```

---

## Story-profile methods

All story-profile methods use **z-tolerance clustering**: nodes are
sorted by z-coordinate and grouped into stories when their z-values are
within `dz_tol` of each other. Provide exactly one of
`selection_set_id`, `selection_set_name`, `node_ids`, or `coordinates`.

---

### `interstory_drift_envelope`

```python
nr.interstory_drift_envelope(
    component,
    selection_set_id=None,
    selection_set_name=None,
    node_ids=None,
    coordinates=None,
    result_name="DISPLACEMENT",
    stage=None,
    dz_tol=1e-3,
    representative="min_id",  # "min_id" | "max_abs_peak"
)
```

Returns a `pd.DataFrame` indexed by `(z_lower, z_upper)`:

| Column | Description |
|---|---|
| `z_lower`, `z_upper` | Story elevation bounds (also in index) |
| `lower_node`, `upper_node` | Representative node IDs |
| `dz` | Story height |
| `max_drift` | Maximum (signed) drift over all steps |
| `min_drift` | Minimum (signed) drift |
| `max_abs_drift` | Peak absolute drift |

`representative` controls which node is selected per story:
- `"min_id"` — smallest node ID (deterministic)
- `"max_abs_peak"` — node with the largest absolute peak response

```python
env = nr.interstory_drift_envelope(
    component=1,
    selection_set_name="ControlPoints",
    dz_tol=1e-3,
)
print(env["max_abs_drift"].max())
```

---

### `interstory_drift_envelope_pd`

```python
nr.interstory_drift_envelope_pd(
    component,
    selection_set_name=None,
    selection_set_id=None,
    node_ids=None,
    coordinates=None,
    result_name="DISPLACEMENT",
    stage=None,
    dz_tol=1e-3,
    representative="max_abs",  # "max_abs" | "max" | "min"
)
```

Flat variant of `interstory_drift_envelope` — returns a regular
`pd.DataFrame` (no MultiIndex) sorted by `z_lower`, suitable for
statistics or histograms. Adds a `representative_drift` column chosen
by the `representative` parameter:

| `representative` | `representative_drift` value |
|---|---|
| `"max_abs"` | `max_abs_drift` |
| `"max"` | `max_drift` |
| `"min"` | `min_drift` |

```python
df = nr.interstory_drift_envelope_pd(
    component=1,
    selection_set_name="ControlPoints",
    representative="max_abs",
)
df.plot.barh(x="z_upper", y="representative_drift")
```

---

### `residual_interstory_drift_profile`

```python
nr.residual_interstory_drift_profile(
    component,
    selection_set_id=None,
    selection_set_name=None,
    node_ids=None,
    coordinates=None,
    result_name="DISPLACEMENT",
    stage=None,
    dz_tol=1e-3,
    representative="min_id",
    signed=True,
    tail=1,
    agg="mean",
)
```

Residual interstory drift per story. Returns a `pd.DataFrame` indexed by
`(z_lower, z_upper)` with columns `lower_node`, `upper_node`, `dz`,
`residual_drift`.

```python
prof = nr.residual_interstory_drift_profile(
    component=1,
    selection_set_name="ControlPoints",
    tail=5,
)
```

---

### `residual_drift_envelope`

```python
nr.residual_drift_envelope(
    component,
    selection_set_id=None,
    selection_set_name=None,
    node_ids=None,
    coordinates=None,
    result_name="DISPLACEMENT",
    stage=None,
    dz_tol=1e-3,
    representative="min_id",
    tail=1,
    agg="mean",
)
```

Summary metrics over the residual interstory drift profile. Returns a
`dict`:

```python
{
    "max_abs_residual_story_drift": float,
    "max_pos_residual_story_drift": float,
    "max_neg_residual_story_drift": float,
}
```

```python
res = nr.residual_drift_envelope(
    component=1,
    selection_set_name="ControlPoints",
    tail=5,
)
print(f"Max residual drift: {res['max_abs_residual_story_drift']:.4f}")
```

---

### `story_pga_envelope`

```python
nr.story_pga_envelope(
    component,
    selection_set_id=None,
    selection_set_name=None,
    node_ids=None,
    coordinates=None,
    result_name="ACCELERATION",
    stage=None,
    dz_tol=1e-3,
    to_g=False,           # divide by g_value to convert to g
    g_value=9810,         # mm/s² (set to 9.81 for m/s²)
    reduce_nodes="max_abs",  # "max_abs" | "max" | "min"
)
```

Peak floor acceleration (PFA) profile. Returns a `pd.DataFrame` indexed
by `story_z`:

| Column | Description |
|---|---|
| `n_nodes` | Nodes requested at this story |
| `n_nodes_present` | Nodes found in the results |
| `max_acc` | Max acceleration (signed) |
| `min_acc` | Min acceleration (signed) |
| `pga` | Peak absolute acceleration |
| `ctrl_node_max/min/pga` | Controlling node IDs |

```python
pga = nr.story_pga_envelope(
    component=1,
    selection_set_name="ControlPoints",
    result_name="ACCELERATION",
    to_g=True,
    g_value=9810,
)
print(pga[["pga"]])
```

---

## Torsion and rocking

### `roof_torsion`

```python
nr.roof_torsion(
    node_a_id=None,         # or node_a_coord=(x, y)
    node_b_id=None,         # or node_b_coord=(x, y)
    node_a_coord=None,
    node_b_coord=None,
    result_name="DISPLACEMENT",
    ux_component=1,
    uy_component=2,
    stage=None,
    signed=True,
    reduce="series",          # "series" | "abs_max" | "max" | "min"
    return_residual=False,
    return_quality=False,
)
```

Estimates roof rotation about the vertical axis (z) using the
small-rotation formula from two plan-view nodes A and B:

```
θ(t) = ([du, dv] · [-dy, dx]) / (dx² + dy²)
```

Provide exactly one of `node_a_id` or `node_a_coord` (same for B).

Returns a `pd.Series` (time history of θ in radians) or a `float` when
`reduce != "series"`. When `return_quality=True`, also returns a debug
`pd.DataFrame` with columns `du, dv, du_rot, dv_rot, ru, rv, rel_norm,
res_norm, rigidity_ratio`.

```python
theta = nr.roof_torsion(
    node_a_id=100,
    node_b_id=200,
    reduce="abs_max",
)
print(f"Peak roof rotation: {theta:.6f} rad")

# Full time history + quality check
theta_ts, debug = nr.roof_torsion(
    node_a_coord=(0.0, 0.0),
    node_b_coord=(20.0, 0.0),
    return_quality=True,
)
print(debug[["rigidity_ratio"]].describe())
```

---

### `base_rocking`

```python
nr.base_rocking(
    node_coords_xy,    # list of exactly 3 (x, y) points
    z_coord,           # base elevation; nodes resolved at (x, y, z_coord)
    result_name="DISPLACEMENT",
    uz_component=3,
    stage=None,
    reduce="series",   # "series" → DataFrame | "abs_max" → dict
    det_tol=1e-12,     # singularity tolerance for the geometry matrix
)
```

Estimates foundation rocking from vertical displacements Uz at 3 base
nodes using small-rotation kinematics:

```
w(x,y) = w₀ + θx·y - θy·x
```

When `reduce="series"` returns a `pd.DataFrame` indexed by step with
columns `w0, theta_x_rad, theta_y_rad, theta_mag_rad, is_singular`.
When `reduce="abs_max"` returns a `dict` with
`theta_x_abs_max, theta_y_abs_max, theta_mag_abs_max`.

If the 3 points are collinear or duplicate (singular geometry matrix),
the method falls back gracefully: angles are zero and `is_singular=True`.

```python
rocking = nr.base_rocking(
    node_coords_xy=[(0, 0), (10, 0), (5, 8)],
    z_coord=0.0,
    reduce="abs_max",
)
print(f"Max rocking: {rocking['theta_mag_abs_max']:.6f} rad")
```

---

### `asce_torsional_irregularity`

```python
nr.asce_torsional_irregularity(
    component,
    side_a_top,      # (x, y, z) of top of side-A story
    side_a_bottom,   # (x, y, z) of bottom of side-A story
    side_b_top,      # (x, y, z) of top of side-B story
    side_b_bottom,   # (x, y, z) of bottom of side-B story
    result_name="DISPLACEMENT",
    stage=None,
    reduce_time="abs_max",     # "abs_max" | "max" | "min"
    definition="max_over_avg", # "max_over_avg" | "max_over_min"
    eps=1e-16,
    signed=True,
    tail=None,   # drop last N steps before reducing (None = keep all)
)
```

ASCE 7 torsional irregularity ratio comparing edge drifts at a story.
All four corner points are `(x, y, z)` tuples resolved to the nearest
nodes.

Returns a `dict`:

```python
{
    "drift_A": float,       # reduced drift magnitude at side A
    "drift_B": float,       # reduced drift magnitude at side B
    "drift_avg": float,     # 0.5 * (drift_A + drift_B)
    "drift_max": float,     # max(drift_A, drift_B)
    "ratio": float,         # torsional irregularity ratio
    "ctrl_side": "A" | "B",
    "node_ids": {
        "A_top": int, "A_bottom": int,
        "B_top": int, "B_bottom": int,
    },
    "metadata": { ... },
}
```

`definition` controls the denominator:
- `"max_over_avg"` — ASCE 7 §12.3.2.1: ratio > 1.2 → Type 1a, > 1.4 → Type 1b
- `"max_over_min"` — alternative for stricter checks

```python
result = nr.asce_torsional_irregularity(
    component=1,
    side_a_top=(0.0, 0.0, 6.0),
    side_a_bottom=(0.0, 0.0, 3.0),
    side_b_top=(20.0, 0.0, 6.0),
    side_b_bottom=(20.0, 0.0, 3.0),
)
print(f"Torsional irregularity ratio: {result['ratio']:.3f}")
if result["ratio"] > 1.2:
    print("Type 1a torsional irregularity (ASCE 7 §12.3.2.1)")
```

---

## Orbit

### `orbit`

```python
nr.orbit(
    result_name="DISPLACEMENT",
    x_component=1,
    y_component=2,
    node_ids=None,           # exactly one of these four
    selection_set_id=None,
    selection_set_name=None,
    coordinates=None,
    stage=None,
    reduce_nodes="none",     # "none" | "mean" | "median" | "max_abs"
    signed=True,
    return_nodes=False,
)
```

Builds an x-y displacement orbit `(sx, sy)` from two components of the
same result. Returns `(pd.Series, pd.Series)`, or
`(pd.Series, pd.Series, list[int])` when `return_nodes=True`.

`reduce_nodes` collapses multiple selected nodes into a single pair per
step:
- `"none"` — no reduction; returned Series have MultiIndex `(node_id, step)`
- `"mean"` / `"median"` — average/median over nodes at each step
- `"max_abs"` — per-component node with the largest absolute value at each step

```python
sx, sy = nr.orbit(
    x_component=1,
    y_component=2,
    selection_set_name="RoofCenter",
    reduce_nodes="mean",
)

import matplotlib.pyplot as plt
plt.plot(sx, sy)
plt.xlabel("Ux"); plt.ylabel("Uy")
plt.axis("equal")
plt.show()
```

---

## API reference

::: STKO_to_python.dataprocess.aggregation.AggregationEngine
