# Peak moment per beam over a pushover

> For every beam in the model, find the largest bending moment that
> ever occurs along its length over the whole pushover. Return a
> sorted table and a histogram so the heavily-loaded members are
> obvious at a glance.

The engineering question is: *which beams are working hardest, and by
how much?* For a force-/displacement-based beam each step writes
section forces at every integration station along the element. The
peak bending moment of an element is then the maximum of `|M|` taken
across **all IPs** and **all steps**.

The fixture used here is
`stko_results_examples/elasticFrame/elasticFrame_mesh_displacementBased_results`:
11 `64-DispBeamColumn3d` elements, each with five Lobatto integration
stations (`gp_xi = [-1, -0.65, 0, +0.65, +1]`). The pushover lives in
`MODEL_STAGE[2]`.

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from STKO_to_python import MPCODataSet

REPO_ROOT = Path("path/to/STKO_to_python")
DATASET = REPO_ROOT / "stko_results_examples" / "elasticFrame" / "elasticFrame_mesh_displacementBased_results"
PUSHOVER_STAGE = "MODEL_STAGE[2]"

ds = MPCODataSet(str(DATASET), "results", verbose=False)
print(ds.unique_element_types)         # ['64-DispBeamColumn3d[...]']
print(ds.number_of_steps)              # {'MODEL_STAGE[1]': 10, 'MODEL_STAGE[2]': 10}
```

## 1. Fetch `section.force` for every beam

```python
edf = ds.elements_info["dataframe"]
beam_ids = edf.query(
    "element_type == '64-DispBeamColumn3d'"
)["element_id"].astype(int).tolist()

er = ds.elements.get_element_results(
    results_name="section.force",
    element_type="64-DispBeamColumn3d",
    model_stage=PUSHOVER_STAGE,
    element_ids=beam_ids,
)
print(er.n_ip, er.gp_xi)
# 5  [-1.  -0.6546  0.  0.6546  1.]
print(er.list_canonicals())
# ('axial_force', 'bending_moment_y', 'bending_moment_z', 'torsion')
```

## 2. Pick the canonical and inspect its columns

`canonical_columns` returns the IP-stamped shortnames for whichever
engineering quantity you ask for:

```python
er.canonical_columns("bending_moment_z")
# ('Mz_ip0', 'Mz_ip1', 'Mz_ip2', 'Mz_ip3', 'Mz_ip4')
er.canonical_columns("bending_moment_y")
# ('My_ip0', 'My_ip1', 'My_ip2', 'My_ip3', 'My_ip4')
```

For this 2-D plane frame the lateral push activates `M_y` (bending
about the local-y axis — out of the local-z direction the load is
applied in). For a different orientation swap it for
`bending_moment_z`. Quick sanity check:

```python
for canon in ("bending_moment_y", "bending_moment_z"):
    cols = er.canonical_columns(canon)
    peak = float(er.df[list(cols)].abs().max().max())
    print(f"  {canon:18s} peak |·|: {peak:.3g}")
# bending_moment_y   peak |·|: 2.701e+07
# bending_moment_z   peak |·|: 0
```

## 3. Collapse to one peak per beam

The peak moment of a beam is the max of `|M_y|` taken across IPs *and*
steps. `peak_abs` already does the per-element collapse for you, so
the only manual step is the row-wise max across IP columns:

```python
canon = "bending_moment_y"
cols = list(er.canonical_columns(canon))

# peak_abs() returns one column per data column with the "_peak_abs"
# suffix; we keep just the IP columns for our canonical and reduce
# across them.
peak_all = er.peak_abs()
peak_per_ip = peak_all[[f"{c}_peak_abs" for c in cols]]
# columns: My_ip0_peak_abs, My_ip1_peak_abs, ...

peak_per_beam = peak_per_ip.max(axis=1)
peak_per_beam.name = f"peak_abs_{canon}"
peak_per_beam.index.name = "element_id"
print(peak_per_beam.head())
```

## 4. Sorted ranking table

```python
ranking = (
    peak_per_beam
    .sort_values(ascending=False)
    .to_frame()
)
ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
print(ranking.to_string())
#             rank  peak_abs_bending_moment_y
# element_id
# 3              1               2.701401e+07
# 11             2               2.701401e+07
# 8              3               1.704946e+07
# 9              4               1.339346e+07
# ...
```

Half the population is concentrated at the top — typical for a
pushover where the column–beam connection regions absorb most of the
demand.

## 5. Histogram

```python
fig, ax = plt.subplots()
ax.hist(peak_per_beam.values / 1e6, bins=10, edgecolor="k", color="C0")
ax.set_xlabel("Peak |M_y| (MN·m)")
ax.set_ylabel("Number of beams")
ax.set_title("Distribution of peak bending moment over the pushover")
plt.show()
```

## 6. Cross-check with `summary()`

`summary()` dumps every per-element statistic in a single DataFrame
(`max`, `min`, `peak_abs`, `residual`, `mean`) — a quick way to verify
you're not double-counting:

```python
summ = er.summary()
print(summ.filter(like="My_ip4").head())
#             My_ip4_max  My_ip4_min  My_ip4_peak_abs  My_ip4_residual  My_ip4_mean
# element_id
# 1                 ...         ...              ...              ...          ...
```

## Variations

- **Use bending about the z-axis instead**: change `canon` to
  `"bending_moment_z"` — the rest of the code is identical.
- **Cyclic / dynamic loading**: `peak_abs` already takes the absolute
  value, so the same recipe gives the *cyclic envelope*. For the
  *signed* maximum, replace it with `er.df[cols].max(axis=0).max()`
  (or use `time_of_peak(..., abs=False)`).
- **Where along the element does the peak occur?** Per-element, the
  IP that wins the row-wise max is the location:
  ``peak_per_ip.idxmax(axis=1)``. Pair with `er.gp_xi` to get ξ ∈ [-1, +1]
  or with `er.physical_x(beam_length=L)` for a position in
  `[0, L]`.
- **Per-element diagram at the peak step**: ``er.plot.diagram("bending_moment_y",
  element_id=eid, step=int(er.time_of_peak(f"My_ip{idx}").loc[eid]))`` —
  see [Solid + fiber beam](../examples/solid_mixed.md) §11.
- **Save and reuse**: `er.save_pickle("beams.pkl.gz")` carries
  `gp_xi`, `gp_weights`, and `element_node_coords` through the
  round-trip, so all the analytics above work after a reload without
  re-opening the MPCO.

For the broader IP-level API see the
[Quad-frame shells example](../examples/quad_frame_shell.md) (2-D
Gauss points) and the
[Solid + fiber beam example](../examples/solid_mixed.md) (3-D bricks
plus fiber sections).
