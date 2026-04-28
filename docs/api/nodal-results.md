# NodalResults

Self-contained view over nodal result data. Returned by
`ds.nodes.get_nodal_results()`. Holds a reference to a
`NodalResultsQueryEngine` and pre-resolved IDs. Every engineering
aggregation (`drift`, `interstory_drift_envelope`, `orbit`, …) is a
forwarder to [AggregationEngine](aggregation-engine.md).

Pickle-stable: saves and reloads without the original HDF5 files.

For the comprehensive usage guide (with full worked examples) see
[Usage guides → NodalResults](../NodalResults.md).

---

## Construction

```python
nr = ds.nodes.get_nodal_results(
    results_name="DISPLACEMENT",
    model_stage="MODEL_STAGE[1]",
    node_ids=[1, 2, 3, 4],           # explicit IDs, or…
    # selection_set_name="roof",      # …named selection set
    # selection_set_id=2,             # …selection set by index
)
```

---

## Key attributes

| Attribute | Type | Description |
|---|---|---|
| `nr.df` | `pd.DataFrame` | MultiIndex `(node_id, step)` × component columns |
| `nr.time` | `np.ndarray \| dict` | Time per step (dict for multi-stage) |
| `nr.node_ids` | `tuple[int, ...]` | Node IDs in this result |
| `nr.n_steps` | `int` | Number of recorded steps |

---

## Introspection

```python
nr.list_results()                     # tuple of result names, e.g. ('DISPLACEMENT',)
nr.list_components('DISPLACEMENT')    # tuple of component labels, e.g. ('1', '2', '3')
```

---

## Fetching data

```python
# Full result as DataFrame
df = nr.fetch(result_name="DISPLACEMENT")

# Single component
s = nr.fetch(result_name="DISPLACEMENT", component=1)

# Filtered by node IDs
s = nr.fetch(result_name="DISPLACEMENT", component=1, node_ids=[1, 4])

# Dynamic attribute style (equivalent to fetch)
view = nr.DISPLACEMENT[1]           # all nodes, component 1
view = nr.DISPLACEMENT[1, [1, 4]]   # nodes [1, 4], component 1
```

---

## Engineering aggregations

All aggregations forward to [AggregationEngine](aggregation-engine.md).

### Drift

```python
ts = nr.drift(top=4, bottom=1, component=1)
# pd.Series indexed by step — relative displacement u_top - u_bottom
```

### Interstory drift envelope

```python
env = nr.interstory_drift_envelope(
    component=1,
    node_ids=[1, 2, 3, 4],
    dz_tol=1e-3,          # Z-coordinate tolerance for story pairing
)
# DataFrame with columns: z_lower, z_upper, lower_node, upper_node,
#                         dz, max_drift, min_drift, max_abs_drift
```

### Residual drift

```python
resid = nr.residual_drift(
    top=4, bottom=1, component=1,
    tail=3,     # number of final steps to average
    agg="mean", # "mean" | "last"
)
# float — mean (or last) drift value at the end of the record
```

### Interstory drift envelope profile

```python
pd_env = nr.interstory_drift_envelope_pd(
    component=1,
    node_ids=[1, 2, 3, 4],
    dz_tol=1e-3,
)
# Wide DataFrame — one column per story pair, rows are time steps
```

### Residual interstory drift profile

```python
profile = nr.residual_interstory_drift_profile(
    component=1,
    node_ids=[1, 2, 3, 4],
    tail=3,
    agg="mean",
)
```

### Story PGA envelope

```python
pga = nr.story_pga_envelope(
    component=1,
    node_ids=[1, 2, 3, 4],
    dz_tol=1e-3,
    to_g=True,
    g_value=9810,
)
```

### Roof torsion

```python
torsion_ts, ratio_ts = nr.roof_torsion(
    node_a_ids=[10, 11],
    node_b_ids=[20, 21],
    component=1,
    dz_tol=1e-3,
)
```

### Base rocking

```python
theta = nr.base_rocking(
    component=3,          # vertical DOF
    node_ids=[1, 2, 3],
    dz_tol=1e-3,
)
```

### ASCE torsional irregularity

```python
result = nr.asce_torsional_irregularity(
    component=1,
    side_a_top=(0.0, 0.0, 6.0),
    side_a_bottom=(0.0, 0.0, 3.0),
    side_b_top=(20.0, 0.0, 6.0),
    side_b_bottom=(20.0, 0.0, 3.0),
)
# dict with keys: ratio, delta_a, delta_b, avg_drift, irregular
```

### Orbit

```python
sx, sy = nr.orbit(node_ids=1, x_component=1, y_component=2)
# Two Series — displacement trajectory for a single node
```

### Residual drift envelope

```python
env = nr.residual_drift_envelope(
    component=1,
    node_ids=[1, 2, 3, 4],
    tail=3,
    agg="mean",
)
```

---

## Plotting

```python
ax, meta = nr.plot.xy(
    y_results_name="DISPLACEMENT",
    y_direction=1,
    y_operation="Max",
    x_results_name="TIME",
)

fig, meta = nr.plot.plot_TH(
    result_name="DISPLACEMENT",
    component=1,
    node_ids=[1, 2, 3, 4],
    split_subplots=True,
)
```

See [Plotting](plotting.md) for the full parameter reference.

---

## Pickle serialization

```python
nr.save_pickle("nr.pkl")
nr.save_pickle("nr.pkl.gz")          # compressed

from STKO_to_python.results.nodal_results_dataclass import NodalResults
nr = NodalResults.load_pickle("nr.pkl")
```

---

## API reference

::: STKO_to_python.results.nodal_results_dataclass.NodalResults

## The dynamic-view proxy

`NodalResults` builds an internal `_ResultView` per result name so
you can write `nr.ACCELERATION[1, [14, 25]]` instead of
`nr.fetch(result_name="ACCELERATION", component=1, node_ids=[14, 25])`.

::: STKO_to_python.results.nodal_results_dataclass._ResultView
    options:
      filters: []
      show_root_heading: true
