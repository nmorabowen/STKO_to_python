# NodalResults — Nodal Results Container

`NodalResults` is a self-contained container for nodal results extracted from MPCO HDF5 files. Once created (via `ds.nodes.get_nodal_results()`), it carries all data, time, metadata, and geometry needed for post-processing — no reference to the original dataset or HDF5 files is required. It can be pickled for persistent storage and reloaded later.

## Creating a NodalResults Object

```python
from STKO_to_python import MPCODataSet

ds = MPCODataSet(r"C:\path\to\results", "mpco")

# Extract results
nr = ds.nodes.get_nodal_results(
    results_name=["DISPLACEMENT", "ACCELERATION"],
    model_stage="MODEL_STAGE[2]",
    selection_set_name="ControlPoints",
)

# From this point on, 'nr' is fully independent of 'ds'
```

You must provide **at least one** of `node_ids`, `selection_set_id`, or `selection_set_name`. If multiple are given, they are unioned. `results_name` defaults to ALL available results if omitted; `model_stage` defaults to ALL stages.

```python
# By explicit node IDs (all results, all stages)
nr = ds.nodes.get_nodal_results(
    node_ids=[10, 20, 30, 40],
)

# By selection set ID, single result
nr = ds.nodes.get_nodal_results(
    results_name="ACCELERATION",
    selection_set_id=5,
)

# Multiple results as a list
nr = ds.nodes.get_nodal_results(
    results_name=["DISPLACEMENT", "ACCELERATION"],
    selection_set_name="ControlPoints",
)
```

## Internal Structure

The core data lives in `nr.df` — a pandas DataFrame with a MultiIndex:

- **Single stage**: index = `(node_id, step)`, columns = MultiIndex `(result, component)`
- **Multiple stages**: index = `(stage, node_id, step)`, columns = MultiIndex `(result, component)`

Time is in `nr.time` (a numpy array for single-stage, or a dict of arrays for multi-stage). Metadata and geometry are in `nr.info` (a `NodalResultsInfo` object).

## Introspection

```python
nr.list_results()
# ('ACCELERATION', 'DISPLACEMENT')

nr.list_components("DISPLACEMENT")
# ('1', '2', '3')

nr.info.nodes_ids       # tuple of node IDs in this result set
nr.info.model_stages    # tuple of stage names
nr.info.nodes_info      # DataFrame with x, y, z coordinates
nr.info.selection_set   # selection set dict (if available)
```

## Fetching Data

### Basic fetch

`fetch()` is the primary data access method. It supports filtering by result name, component, node IDs, selection sets, and coordinates.

```python
# All components of DISPLACEMENT, all nodes
df = nr.fetch("DISPLACEMENT")

# Single component
s = nr.fetch("DISPLACEMENT", component=1)

# Filter by node IDs
s = nr.fetch("DISPLACEMENT", component=1, node_ids=[10, 20])

# Filter by selection set name
s = nr.fetch("DISPLACEMENT", component=1, selection_set_name="RoofNodes")

# Filter by selection set ID
s = nr.fetch("DISPLACEMENT", component=1, selection_set_id=3)

# Filter by coordinates (resolves to nearest nodes)
s = nr.fetch("DISPLACEMENT", component=1, coordinates=[(0.0, 0.0), (10.0, 5.0)])

# Return resolved node IDs alongside data
data, node_ids = nr.fetch("DISPLACEMENT", component=1,
                          selection_set_name="RoofNodes",
                          return_nodes=True)
```

All node sources (node_ids, selection_set_id, selection_set_name, coordinates) are unioned.

### Attribute-style access (ResultView)

Each result type is available as an attribute that returns a `_ResultView` proxy:

```python
# Equivalent to nr.fetch("DISPLACEMENT", component=1)
s = nr.DISPLACEMENT[1]

# All components
df = nr.DISPLACEMENT[:]

# Component + node filter
s = nr.DISPLACEMENT[1, [10, 20]]
df = nr.DISPLACEMENT[:, [10, 20]]
```

### Nearest-node fetch

```python
# Convenience: coordinates -> nearest node -> fetch
s = nr.fetch_nearest(
    points=[(0.0, 0.0, 3.0), (10.0, 5.0, 6.0)],
    result_name="DISPLACEMENT",
    component=1,
)
```

## Drift Analysis

### Relative displacement (delta_u)

```python
# Time history of relative displacement between two nodes
du = nr.delta_u(
    top=42,              # node ID or coordinates (x,y) / (x,y,z)
    bottom=10,
    component=1,         # component number
    result_name="DISPLACEMENT",
    signed=True,
    reduce="series",     # "series" -> pd.Series, "abs_max" -> float
)
```

### Interstory drift

```python
# Drift ratio = (u_top - u_bottom) / (z_top - z_bottom)
dr = nr.drift(
    top=42,
    bottom=10,
    component=1,
    reduce="series",
)
```

Both `top` and `bottom` accept either a node ID (int) or coordinates that are resolved to the nearest node.

### Interstory drift envelope

Automatically clusters nodes into story levels by z-coordinate tolerance, then computes drift envelopes per story:

```python
env = nr.interstory_drift_envelope(
    component=1,
    selection_set_name="ControlPoints",
    dz_tol=1e-3,                    # z-tolerance for story clustering
    representative="min_id",         # or "max_abs_peak"
)
# Returns DataFrame indexed by (z_lower, z_upper):
#   lower_node, upper_node, dz, max_drift, min_drift, max_abs_drift
```

Parameters for node selection (provide exactly one):

- `selection_set_id`
- `selection_set_name`
- `node_ids`
- `coordinates`

The `representative` parameter controls how a single representative node is chosen per story when multiple nodes exist at the same elevation: `"min_id"` picks the smallest node ID (fast, deterministic), while `"max_abs_peak"` picks the node with the largest absolute peak response.

### Interstory drift envelope (alternative)

`interstory_drift_envelope_pd()` returns a flat DataFrame suitable for statistics or histograms:

```python
env_pd = nr.interstory_drift_envelope_pd(
    component=1,
    selection_set_name="ControlPoints",
    representative="max_abs",     # "max_abs", "max", or "min"
)
# Returns DataFrame with columns:
#   z_lower, z_upper, dz, max_drift, min_drift, max_abs_drift,
#   representative_drift, lower_node, upper_node
```

## PGA Envelope

Story-level peak ground acceleration envelope:

```python
pga = nr.story_pga_envelope(
    component=1,
    selection_set_name="ControlPoints",
    result_name="ACCELERATION",
    to_g=True,           # convert to g
    g_value=9810,        # mm/s^2 if your model is in mm
    dz_tol=1e-3,
    reduce_nodes="max_abs",
)
# Returns DataFrame indexed by story_z:
#   n_nodes, n_nodes_present, max_acc, min_acc, pga,
#   ctrl_node_max, ctrl_node_min, ctrl_node_pga
```

## Roof Torsion

Estimates rotation about the vertical axis (z) from two roof nodes using the small-rotation formula:

```python
theta = nr.roof_torsion(
    node_a_id=100,       # or node_a_coord=(x, y)
    node_b_id=200,       # or node_b_coord=(x, y)
    ux_component=1,
    uy_component=2,
    reduce="series",     # "series", "abs_max", "max", "min"
)

# With quality diagnostics
theta, debug_df = nr.roof_torsion(
    node_a_id=100,
    node_b_id=200,
    return_quality=True,
)
# debug_df contains: du, dv, du_rot, dv_rot, ru, rv,
#                     rel_norm, res_norm, rigidity_ratio
```

## Base Rocking

Estimates foundation rocking angles from vertical displacements at 3 base nodes:

```python
rocking = nr.base_rocking(
    node_coords_xy=[(0, 0), (10, 0), (5, 8)],  # 3 plan-view points
    z_coord=0.0,                                  # base elevation
    uz_component=3,
    reduce="series",
)
# Returns DataFrame: w0, theta_x_rad, theta_y_rad, theta_mag_rad, is_singular

# Or summary
summary = nr.base_rocking(
    node_coords_xy=[(0, 0), (10, 0), (5, 8)],
    z_coord=0.0,
    reduce="abs_max",
)
# Returns dict: theta_x_abs_max, theta_y_abs_max, theta_mag_abs_max
```

If the 3 points are collinear or duplicate, the method gracefully falls back to zero rocking (no exception raised).

## Residual Drifts

### Single pair

```python
rd = nr.residual_drift(
    top=42,
    bottom=10,
    component=1,
    tail=5,       # average last 5 steps (reduces end-of-record noise)
    agg="mean",   # or "median"
)
# Returns: float (residual drift ratio)
```

### Profile (per story)

```python
prof = nr.residual_interstory_drift_profile(
    component=1,
    selection_set_name="ControlPoints",
    tail=5,
    agg="mean",
)
# Returns DataFrame indexed by (z_lower, z_upper):
#   lower_node, upper_node, dz, residual_drift
```

### Envelope summary

```python
env = nr.residual_drift_envelope(
    component=1,
    selection_set_name="ControlPoints",
)
# Returns dict:
#   max_abs_residual_story_drift
#   max_pos_residual_story_drift
#   max_neg_residual_story_drift
```

## ASCE Torsional Irregularity

Single-story torsional irregularity check per ASCE 7:

```python
result = nr.asce_torsional_irregularity(
    component=1,
    side_a_top=(0.0, 0.0, 6.0),
    side_a_bottom=(0.0, 0.0, 3.0),
    side_b_top=(20.0, 0.0, 6.0),
    side_b_bottom=(20.0, 0.0, 3.0),
    reduce_time="abs_max",
    definition="max_over_avg",   # or "max_over_min"
)
# Returns dict:
#   drift_A, drift_B, drift_avg, drift_max, ratio, ctrl_side,
#   node_ids: {A_top, A_bottom, B_top, B_bottom},
#   metadata: {component, result_name, ...}
```

All four inputs are `(x, y, z)` tuples resolved to the nearest nodes.

## Orbit (Displacement Path)

Build an x-y displacement orbit from two components:

```python
sx, sy = nr.orbit(
    result_name="DISPLACEMENT",
    x_component=1,
    y_component=2,
    selection_set_name="RoofCenter",
    reduce_nodes="mean",    # "none", "mean", "median", "max_abs"
)

# Plot
import matplotlib.pyplot as plt
plt.plot(sx, sy)
plt.xlabel("Ux"); plt.ylabel("Uy")
plt.title("Roof displacement orbit")
plt.axis("equal")
plt.show()
```

When `reduce_nodes="none"` and multiple nodes are selected, the returned series have a MultiIndex `(node_id, step)`.

## Plotting

`NodalResults` includes a built-in plotter accessed via the `.plot` property:

```python
nr.plot.time_history(result_name="DISPLACEMENT", component=1, node_ids=[10, 20])
nr.plot.envelope(...)
# (See NodalResultsPlotter for full API)
```

## Pickle Serialization

Save and reload results without needing the original HDF5 files:

```python
# Save
nr.save_pickle("my_results.pkl")

# Save compressed
nr.save_pickle("my_results.pkl.gz")   # auto-detects .gz extension

# Reload
from STKO_to_python import NodalResults
nr = NodalResults.load_pickle("my_results.pkl")
nr = NodalResults.load_pickle("my_results.pkl.gz")
```

Compression is auto-detected from the `.gz` extension, or you can force it with `compress=True/False`.

## Geometry Helpers (via nr.info)

The `nr.info` object provides geometry utilities:

```python
# Find nearest node to a point
ids = nr.info.nearest_node_id([(5.0, 3.0, 0.0)])

# Find nearest node within a specific file partition
ids = nr.info.nearest_node_id([(5.0, 3.0)], file_id=0)

# Also return distances
ids, dists = nr.info.nearest_node_id([(5.0, 3.0)], return_distance=True)

# Resolve selection set names to IDs
sids = nr.info.selection_set_ids_from_names("ControlPoints")

# Get node IDs from a selection set
nids = nr.info.selection_set_node_ids_by_name("ControlPoints")
nids = nr.info.selection_set_node_ids(selection_set_id=3)
```

## Full Example: Seismic Post-Processing

```python
from STKO_to_python import MPCODataSet, NodalResults

# --- Load ---
ds = MPCODataSet(r"C:\results\building", "mpco", name="RC_Building")

# --- Extract ---
nr = ds.nodes.get_nodal_results(
    results_name=["DISPLACEMENT", "ACCELERATION"],
    model_stage="MODEL_STAGE[2]",
    selection_set_name="ControlPoints",
)

# --- Save for reuse ---
nr.save_pickle("building_nodal.pkl.gz")

# --- Later: reload without HDF5 files ---
nr = NodalResults.load_pickle("building_nodal.pkl.gz")

# --- Interstory drift envelope ---
drift_env = nr.interstory_drift_envelope(
    component=1,
    selection_set_name="ControlPoints",
)
print("Max interstory drift (X):", drift_env["max_abs_drift"].max())

# --- PGA profile ---
pga = nr.story_pga_envelope(
    component=1,
    selection_set_name="ControlPoints",
    result_name="ACCELERATION",
    to_g=True,
    g_value=9810,
)
print("Peak floor accelerations:\n", pga[["pga"]])

# --- Residual drifts ---
res = nr.residual_drift_envelope(
    component=1,
    selection_set_name="ControlPoints",
    tail=5,
)
print("Max residual drift:", res["max_abs_residual_story_drift"])

# --- Roof torsion ---
theta_max = nr.roof_torsion(
    node_a_id=100,
    node_b_id=200,
    reduce="abs_max",
)
print(f"Max roof torsion: {theta_max:.6f} rad")

# --- ASCE torsional irregularity ---
asce = nr.asce_torsional_irregularity(
    component=1,
    side_a_top=(0.0, 0.0, 6.0),
    side_a_bottom=(0.0, 0.0, 3.0),
    side_b_top=(20.0, 0.0, 6.0),
    side_b_bottom=(20.0, 0.0, 3.0),
)
print(f"Torsional irregularity ratio: {asce['ratio']:.3f}")

# --- Displacement orbit ---
sx, sy = nr.orbit(
    x_component=1,
    y_component=2,
    selection_set_name="RoofCenter",
    reduce_nodes="mean",
)
```
