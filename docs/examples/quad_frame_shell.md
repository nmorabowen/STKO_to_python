# Quad-frame shells

Demonstrates multi-partition datasets and the full Gauss-point
integration API against the `203-ASDShellQ4` fixture.

**Script:** [`examples/quad_frame_shell_example.py`](https://github.com/nmorabowen/STKO_to_python/blob/main/examples/quad_frame_shell_example.py)

```bash
python examples/quad_frame_shell_example.py
```

---

## Fixture

`stko_results_examples/elasticFrame/QuadFrame_results` — two
partition files (`results.part-0.mpco`, `results.part-1.mpco`),
recorder `results`. 676 nodes, 625 `203-ASDShellQ4` shells + 75
`5-ElasticBeam3d` columns, 2 model stages, 10 steps.

The library merges the partitions transparently — all element IDs
appear in a single `ds.elements_info["dataframe"]`.

---

## Sections covered

### 1. Multi-partition introspection

```python
ds = MPCODataSet(str(DATASET_DIR), "results")
ds.unique_element_types   # ['203-ASDShellQ4[...]', '5-ElasticBeam3d[...]']
len(ds.elements_info["dataframe"])  # 700 total elements across both partitions
```

### 2. Discover shell element IDs

```python
df = ds.elements_info["dataframe"]
shell_ids = [int(i) for i in
    df.query("element_type == '203-ASDShellQ4'")["element_id"].head(5)]
```

### 3. Shell `section.force` — 4 Gauss points

```python
er = ds.elements.get_element_results(
    results_name="section.force",
    element_type="203-ASDShellQ4",
    model_stage="MODEL_STAGE[1]",
    element_ids=shell_ids,
)
# er.n_ip == 4
# er.gp_natural.shape == (4, 2)   — 2×2 Gauss-Legendre in ξη
# er.gp_weights == [1., 1., 1., 1.]  (sum=4, one integration over [-1,1]²)
```

### 4. Canonical names for shell section forces

```python
er.list_canonicals()
# ('bending_moment_xx', 'bending_moment_xy', 'bending_moment_yy',
#  'membrane_xx', 'membrane_xy', 'membrane_yy',
#  'transverse_shear_xz', 'transverse_shear_yz')

er.canonical_columns("membrane_xx")
# ('Fxx_ip0', 'Fxx_ip1', 'Fxx_ip2', 'Fxx_ip3')
```

### 5. Per-Gauss-point slicing

```python
sub0 = er.at_ip(0)   # 8 columns: Fxx_ip0, Fyy_ip0, …, Vyz_ip0
sub3 = er.at_ip(3)   # 8 columns: Fxx_ip3, …, Vyz_ip3
```

### 6. Physical coordinates and element areas

```python
phys = er.physical_coords()    # (n_e, 4, 3) — physical (x,y,z) per IP
dets = er.jacobian_dets()      # (n_e, 4) — surface Jacobian per IP

# Element area via quadrature: A = sum(w_i * |J_i|)
areas = (er.gp_weights[None, :] * dets).sum(axis=1)
```

### 7. Surface-integrated membrane force

```python
# Σ Fxx_ip * w_ip * |J_ip|  over all 4 IPs — per element, per step
s = er.integrate_canonical("membrane_xx")

# Reshape to (n_steps, n_elements) matrix
mtx = s.unstack("element_id")
```

### 8. Spatial scatter plot

```python
ax, meta = er.plot.scatter("membrane_xx", step=5, axes=("x", "z"))
ax.figure.colorbar(meta["scatter"], ax=ax)
```

### 9. Pickle — `gp_natural` survives

```python
er.save_pickle("shells.pkl")
er2 = ElementResults.load_pickle("shells.pkl")
# er2.gp_natural preserved → integrate_canonical still works
```
