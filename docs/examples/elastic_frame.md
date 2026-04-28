# Elastic frame

Demonstrates the core workflow against the single-partition
`5-ElasticBeam3d` fixture — the recommended first read after the
[Usage tour](usage_tour.md).

**Script:** [`examples/elastic_frame_example.py`](https://github.com/nmorabowen/STKO_to_python/blob/main/examples/elastic_frame_example.py)

```bash
python examples/elastic_frame_example.py
```

---

## Fixture

`stko_results_examples/elasticFrame/results` — single partition,
recorder `results`. 4 nodes, 3 `5-ElasticBeam3d` elements, 2 model
stages (`MODEL_STAGE[1]`, `MODEL_STAGE[2]`), 10 steps each.

---

## Sections covered

### 1. Dataset introspection

```python
ds = MPCODataSet(str(DATASET_DIR), "results")
ds.model_stages           # ['MODEL_STAGE[1]', 'MODEL_STAGE[2]']
ds.node_results_names     # DISPLACEMENT, REACTION, ACCELERATION, …
ds.element_results_names  # force, localForce
ds.unique_element_types   # ['5-ElasticBeam3d[1:0:0]']
```

### 2. Nodal results and engineering aggregations

```python
nr = ds.nodes.get_nodal_results(
    results_name="DISPLACEMENT",
    model_stage="MODEL_STAGE[1]",
    node_ids=[1, 2, 3, 4],
)

ts    = nr.drift(top=4, bottom=1, component=1)
env   = nr.interstory_drift_envelope(component=1, node_ids=[1,2,3,4], dz_tol=1e-3)
resid = nr.residual_drift(top=4, bottom=1, component=1, tail=3, agg="mean")
sx, sy = nr.orbit(node_ids=1, x_component=1, y_component=2)
```

### 3. XY plotting

```python
nr.plot.xy(y_results_name="DISPLACEMENT", y_direction=1,
           y_operation="Max", x_results_name="TIME")

ds.plot.xy(model_stage="MODEL_STAGE[1]", results_name="DISPLACEMENT",
           node_ids=[1,2,3,4], y_direction=1, y_operation="Max",
           x_results_name="TIME")
```

### 4. Closed-form beam results

`force` and `localForce` are closed-form buckets — 12 components each,
no integration points (`gp_xi=None`, `n_ip=0`).

```python
er = ds.elements.get_element_results(
    results_name="force",
    element_type="5-ElasticBeam3d",
    model_stage="MODEL_STAGE[1]",
    element_ids=[1, 2, 3],
)
# er.df.shape == (30, 12)  — 3 elements × 10 steps
# er.list_components() == ('Px_1', 'Py_1', ..., 'Mz_2')
```

### 5. Canonical names on closed-form results

```python
er_local = ds.elements.get_element_results("localForce", ...)
er_local.list_canonicals()
# ('axial_force', 'bending_moment_y', 'bending_moment_z',
#  'shear_y', 'shear_z', 'torsion')

er_local.canonical_columns("axial_force")   # ('N_1', 'N_2')
df = er_local.canonical("bending_moment_z") # columns: Mz_1, Mz_2
```

!!! note "integrate_canonical on closed-form"
    Closed-form buckets have no integration points and no `gp_weights`,
    so `integrate_canonical()` raises `ValueError`. This is by design —
    the result is already a nodal-force summary, not a field quantity.

### 6. Pickle round-trips

```python
nr.save_pickle("nr.pkl")
loaded = NodalResults.load_pickle("nr.pkl")

er.save_pickle("er.pkl")
loaded = ElementResults.load_pickle("er.pkl")
```
