# Solid + fiber beam

Demonstrates the full integration-point API on a mixed-element dataset
containing `56-Brick` continuum solids and `64-DispBeamColumn3d` beams
with compressed fiber sections.

**Script:** [`examples/solid_mixed_example.py`](https://github.com/nmorabowen/STKO_to_python/blob/main/examples/solid_mixed_example.py)

```bash
python examples/solid_mixed_example.py
```

---

## Fixture

`stko_results_examples/solid_partition_example` — two partition files
(`Recorder.part-0.mpco`, `Recorder.part-1.mpco`), recorder `Recorder`.
1720 nodes, 647 `56-Brick` + 712 `64-DispBeamColumn3d` elements,
1 model stage (`MODEL_STAGE[1]`), 1667 steps.

---

## Part 1 — Brick continuum (`56-Brick`)

### Material stress — 8 Gauss points

```python
er = ds.elements.get_element_results(
    results_name="material.stress",
    element_type="56-Brick",
    model_stage="MODEL_STAGE[1]",
    element_ids=brick_ids,
)
# er.n_ip == 8
# er.gp_dim == 3
# er.gp_xi is None   — multi-D bucket, no 1-D ξ
# er.gp_natural.shape == (8, 3)   — (ξ, η, ζ) at ±1/√3 per axis
# er.gp_weights.sum() == 8.0      — 2×2×2 Gauss-Legendre over [-1,1]³
```

Column layout: `sigma11_ip0, sigma22_ip0, …, sigma13_ip7`
(6 stress components × 8 IPs = 48 columns).

### Per-IP stress slice

```python
sub0 = er.at_ip(0)
# columns: ['sigma11_ip0', 'sigma22_ip0', 'sigma33_ip0',
#           'sigma12_ip0', 'sigma23_ip0', 'sigma13_ip0']
```

### Physical coordinates inside the element bounding box

```python
phys = er.physical_coords()    # (n_e, 8, 3)
# All 8 IPs verified to lie inside element bounding box

dets  = er.jacobian_dets()     # (n_e, 8)
# V = sum(w_i * |J_i|) matches bounding-box volume for axis-aligned bricks
vols  = (er.gp_weights[None, :] * dets).sum(axis=1)
```

### Volume-integrated stress

```python
s = er.integrate_canonical("stress_11")
# Series indexed (element_id, step), name='integral_stress_11'
# Σ σ₁₁_ip * w_ip * |J_ip|

mtx = s.unstack("element_id")   # (n_steps, n_elements) matrix
```

### Spatial scatter at a step

```python
ax, meta = er.plot.scatter("stress_11", step=100, axes=("x", "z"))
```

---

## Part 2 — Displacement-based beam (`64-DispBeamColumn3d`)

### Section forces — 2-IP Lobatto end-stations

```python
er_sf = ds.elements.get_element_results(
    results_name="section.force",
    element_type="64-DispBeamColumn3d",
    model_stage="MODEL_STAGE[1]",
    element_ids=beam_ids,
)
# er_sf.n_ip == 2
# er_sf.gp_xi == array([-1.0, 1.0])   — Lobatto rule: ξ=-1 → node i, ξ=+1 → node j
# er_sf.gp_natural is None              — line elements use gp_xi, not gp_natural
# er_sf.gp_weights is None              — no catalog weights for custom-rule beams
```

Column layout: `P_ip0, Mz_ip0, My_ip0, T_ip0, P_ip1, Mz_ip1, My_ip1, T_ip1`.

```python
sub_i = er_sf.at_ip(0)    # section forces at node-i end
sub_j = er_sf.at_ip(1)    # section forces at node-j end
```

### Moment diagram along the beam

```python
ax, meta = er_sf.plot.diagram("bending_moment_z", element_id=1, step=100)
```

### Compressed fiber sections — 6 fibers × 2 IPs

```python
er_fib = ds.elements.get_element_results(
    results_name="section.fiber.stress",
    element_type="64-DispBeamColumn3d",
    model_stage="MODEL_STAGE[1]",
    element_ids=beam_ids,
)
# n_components == 12   (6 fibers × 2 IPs)
# gp_xi == array([-1.0, 1.0])

sub_ip0 = er_fib.at_ip(0)
# shape: (n_elements × n_steps, 6)
# columns: sigma11_f0_ip0, sigma11_f1_ip0, …, sigma11_f5_ip0
```

### Why `integrate_canonical` is not available for fiber buckets

Fiber buckets have `gp_weights=None` (the custom Lobatto/Gauss rule
used by `DispBeamColumn3d` isn't written to the MPCO file). Calling
`integrate_canonical()` raises `ValueError` with a pointer to manual
integration. For element-level fiber integration supply your own
fiber-area weights:

```python
sub_ip0 = er_fib.at_ip(0)           # (n_el × n_steps, 6 fibers)
fiber_areas = np.array([...])        # (6,) — fiber areas from the model
result = (sub_ip0.to_numpy() * fiber_areas[None, :]).sum(axis=1)
```
