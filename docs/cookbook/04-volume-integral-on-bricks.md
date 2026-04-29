# Volume integral of σ_11 over a brick subdomain

> Integrate the axial stress σ_11 over the volume of every brick
> element above an elevation cut, summed at every step. The result is
> a single scalar per step — a resultant axial force across the cut
> region.

The engineering question is: *for the part of the model above
`z = 5.0`, what is the total axial force carried at every step?* For a
hex element, OpenSees writes σ_11 at each Gauss point in the
`material.stress` recorder. The volume integral is the standard
quadrature sum:

$$
\int_{\Omega_e} \sigma_{11}\,\mathrm{d}V
\;\approx\;
\sum_{i=1}^{n_\mathrm{IP}} \sigma_{11,i}\, w_i\, |J_i|
$$

— exactly what `er.integrate_canonical("stress_11")` computes for
every element and step.

The fixture this tutorial targets is
`stko_results_examples/solid_partition_example`, a two-partition mesh
with `56-Brick` continuum and `64-DispBeamColumn3d` beams. **It is
gitignored**, so the script below short-circuits gracefully if the
fixture is absent locally — drop your own brick fixture in to run it.

```python
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from STKO_to_python import MPCODataSet

REPO_ROOT = Path("path/to/STKO_to_python")
DATASET = REPO_ROOT / "stko_results_examples" / "solid_partition_example"

if not (DATASET / "Recorder.part-0.mpco").exists():
    sys.exit(f"Fixture not present at {DATASET}; skipping.")

ds = MPCODataSet(str(DATASET), "Recorder", verbose=False)
print(ds.unique_element_types)
# ['56-Brick[...]', '64-DispBeamColumn3d[...]']
```

## 1. Select the bricks above the cut

`get_elements_at_z_levels` returns every element whose `z_min …
z_max` range straddles a requested level — useful for picking out the
elements that *cross* a given elevation. To take *everything above*
the cut, pair it with a centroid filter:

```python
Z_CUT = 5.0

cross_cut = ds.elements.get_elements_at_z_levels(
    list_z=[Z_CUT],
    element_type="56-Brick",
)
print(f"{len(cross_cut)} bricks straddle z = {Z_CUT}")

edf = ds.elements_info["dataframe"]
above_cut = edf[
    (edf["element_type"] == "56-Brick")
    & (edf["centroid_z"] > Z_CUT)
]
brick_ids = above_cut["element_id"].astype(int).tolist()
print(f"{len(brick_ids)} bricks above z = {Z_CUT}")
```

## 2. Fetch `material.stress` for those bricks

```python
er = ds.elements.get_element_results(
    results_name="material.stress",
    element_type="56-Brick",
    model_stage="MODEL_STAGE[1]",
    element_ids=brick_ids,
)
print(er.list_canonicals())
# ('stress_11', 'stress_22', 'stress_33', 'stress_12', 'stress_23', 'stress_13')
print(er.n_ip, er.gp_dim, er.gp_natural.shape, er.gp_weights.shape)
# 8  3  (8, 3)  (8,)
```

Standard 2×2×2 Gauss–Legendre: eight points at `±1/√3` along each
axis, every weight 1.0.

## 3. Confirm the catalog is wired up

`integrate_canonical` needs three things to be present:
`gp_weights` (carried by the catalog), `jacobian_dets()` (computed
from element node coords + shape functions), and a canonical that
resolves to exactly `n_ip` columns (one σ_11 column per Gauss point).

```python
dets = er.jacobian_dets()        # (n_elements, 8)
print("All |J| > 0:", bool(np.all(dets > 0)))

# Element volumes via quadrature ( ∫ 1 dV = Σ w_i |J_i| )
volumes = (er.gp_weights[None, :] * dets).sum(axis=1)
print(f"V range: [{volumes.min():.3f}, {volumes.max():.3f}]")
```

## 4. Integrate σ_11 over each brick at every step

```python
sigma_int = er.integrate_canonical("stress_11")    # Series (element_id, step)
print(sigma_int.head())
# element_id  step
# 100         0      ...
# 100         1      ...
# ...
```

`integrate_canonical` returns a `Series` with the standard
`(element_id, step)` MultiIndex — same shape as everything else, so
all the usual pandas reshaping idioms apply:

```python
per_step = sigma_int.unstack("element_id")    # rows = step, cols = element_id
print(per_step.head())
```

## 5. Sum across the subdomain to get a single time history

```python
total_axial = per_step.sum(axis=1)
total_axial.index = er.time
total_axial.name = "axial_force_above_cut"

fig, ax = plt.subplots()
ax.plot(total_axial.index, total_axial.values, lw=1.5)
ax.axhline(0.0, color="k", lw=0.5)
ax.set_xlabel("Time")
ax.set_ylabel(r"$\int \sigma_{11}\,dV$ above z = " + f"{Z_CUT}")
ax.set_title(f"Resultant axial force above z = {Z_CUT}")
plt.show()
```

## 6. Manual quadrature, for full control

`integrate_canonical` is a thin wrapper over the multiply-and-sum.
For audit purposes — or to integrate a custom quantity like
`σ_11 · σ_22` — drop down to the underlying arrays:

```python
cols = list(er.canonical_columns("stress_11"))
weights = er.gp_weights                             # (n_ip,)
dets = er.jacobian_dets()                           # (n_e, n_ip)

step = 5
eid = brick_ids[0]
eidx = er.element_ids.index(eid)

sigma = er.df.xs((eid, step))[cols].to_numpy()      # (n_ip,)
manual = float((sigma * weights * dets[eidx]).sum())
helper = float(sigma_int.loc[eid, step])
print(f"manual = {manual:.6e}")
print(f"helper = {helper:.6e}")
print(f"match  = {np.isclose(manual, helper)}")
```

## Variations

- **Different stress component**: replace `"stress_11"` with
  `"stress_22"`, `"stress_33"`, `"stress_12"`, etc. The full set is
  in `er.list_canonicals()`.
- **Subdomain by selection set instead of z-cut**: pass
  `selection_set_name="MyRegion"` to `get_element_results` and skip
  the z-filter entirely.
- **Element-by-element table** (e.g. ranked by axial force at the
  residual step): `per_step.iloc[-1].sort_values()`.
- **Strain instead of stress**: fetch `"material.strain"` and use
  `"strain_11"`. The integral measures volumetric strain energy
  density × volume.
- **Tetrahedral or triangular subdomain**: as long as the element
  class is in `utilities/gauss_points.py` and
  `utilities/shape_functions.py`, the same `integrate_canonical` call
  works. Adding a new class is a one-line catalog entry; see the
  ElementResults reference §
  ["Multi-dimensional integration points"](../ElementResults.md#multi-dimensional-integration-points-shells-solids).
- **Shell surface integral** (one dimension lower):
  `er_shell.integrate_canonical("membrane_xx")` — same idiom,
  surface measure instead of volume measure. Worked through in
  [Quad-frame shells §8](../examples/quad_frame_shell.md).

For the broader brick / fiber-beam tour see
[Solid + fiber beam](../examples/solid_mixed.md).
