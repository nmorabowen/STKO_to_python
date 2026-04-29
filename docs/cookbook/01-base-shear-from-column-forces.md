# Base shear from column forces

> Compute the total base shear of a frame as a time history by
> summing the shear in every column at the foundation.

The engineering question is: *for each step, what horizontal force is
the foundation transmitting?* In OpenSees the answer lives in the
`localForce` recorder on the column elements at `Z = 0` — pull the
shear at the base node of every base column and add them up.

The fixture used here is `stko_results_examples/elasticFrame/elasticFrame_mesh_results`:
a single-bay frame meshed into 11 `5-ElasticBeam3d` segments, with a
horizontal pushover applied in `MODEL_STAGE[2]`.

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from STKO_to_python import MPCODataSet

REPO_ROOT = Path("path/to/STKO_to_python")          # adjust as needed
DATASET = REPO_ROOT / "stko_results_examples" / "elasticFrame" / "elasticFrame_mesh_results"
PUSHOVER_STAGE = "MODEL_STAGE[2]"                   # gravity is stage 1

ds = MPCODataSet(str(DATASET), "results", verbose=False)
print(ds.model_stages, ds.unique_element_types, ds.number_of_steps)
```

## 1. Identify the base columns

A base column is any `5-ElasticBeam3d` element whose lower node sits
at `Z = 0`. The element index already carries connectivity
(`node_list`) and the node table carries coordinates — joining them
gives the answer.

```python
edf = ds.elements_info["dataframe"]
ndf = ds.nodes_info["dataframe"]

beams = edf[edf["element_type"] == "5-ElasticBeam3d"]
node_z = ndf.set_index("node_id")["z"]

def has_base_node(node_list, tol=1e-6):
    return any(abs(node_z[int(n)]) < tol for n in node_list)

base_ids = [
    int(eid)
    for eid, nodes in zip(beams["element_id"], beams["node_list"])
    if has_base_node(nodes)
]
print(f"Base columns: {base_ids}")
```

For this fixture: `[1, 4]` — the bottom segment of each of the two
columns.

## 2. Fetch `localForce` on the base columns

`localForce` is the closed-form, element-local force/moment bucket for
beams. It carries axial, two shears, torsion and two moments at each
end of the element — twelve columns total.

```python
er = ds.elements.get_element_results(
    results_name="localForce",
    element_type="5-ElasticBeam3d",
    model_stage=PUSHOVER_STAGE,
    element_ids=base_ids,
)
print(er.list_components())     # ('N_1', 'Vy_1', 'Vz_1', ..., 'Mz_2')
print(er.list_canonicals())     # ('axial_force', 'shear_y', 'shear_z', ...)
```

## 3. Pick the right shear via the canonical layer

The library's *canonical* layer maps engineering quantities to whichever
shortnames the bucket actually carries:

```python
er.canonical_columns("shear_y")   # ('Vy_1', 'Vy_2')
er.canonical_columns("shear_z")   # ('Vz_1', 'Vz_2')
```

For this frame the lateral push is in the local-`z` direction of the
columns (see the orientation of the section assigned in `sections.tcl`),
so the active shear lives in `Vz_*`. Quick sanity check:

```python
for canon in ("shear_y", "shear_z"):
    cols = er.canonical_columns(canon)
    peak = float(er.df[list(cols)].abs().max().max())
    print(f"  {canon:8s} peak |·|: {peak:.3g}")
# shear_y   peak |·|: 0.0
# shear_z   peak |·|: 17582.106
```

If your model pushes in local-`y` instead, swap `"shear_z"` for
`"shear_y"` below — the canonical layer hides the column-name
difference.

## 4. Sum the end-1 shear into a base-shear time history

`Vz_1` is the local-z shear at end 1 — the foundation end of each
column. Sign convention is element-local; for a single push direction
all columns carry the same sign and a simple sum gives the resultant.

```python
shear_per_col = er.df["Vz_1"].unstack("element_id")   # rows=step, cols=element_id
print(shear_per_col)
#         1            4
# step
# 0    -10000.0  -1000.0
# 1    -10500.0  -1500.0
# ...

base_shear = shear_per_col.sum(axis=1)
base_shear.index = er.time
base_shear.name = "base_shear"
print(base_shear.head())
```

## 5. Plot the history

```python
fig, ax = plt.subplots()
ax.plot(base_shear.index, base_shear.values, lw=1.5)
ax.axhline(0.0, color="k", lw=0.5)
ax.set_xlabel("Time")
ax.set_ylabel("Base shear  (sum of Vz at column bases)")
ax.set_title("Pushover base-shear history")
plt.show()
```

For a one-shot plot of a single column's shear, the per-result
`history` helper does the same job without the intermediate DataFrame:

```python
ax, _ = er.plot.history("Vz_1", element_ids=base_ids)
ax.set_ylabel("Column-base shear, Vz_1")
```

## Variations

- **Absolute base shear** (cyclic / dynamic loading): use
  `shear_per_col.abs().sum(axis=1)` so signs from columns pushing
  opposite ways don't cancel — useful for envelope plots but not for a
  resultant.
- **Frame oriented in local-y**: swap every `Vz_1` for `Vy_1` and
  every `"shear_z"` for `"shear_y"`. The rest of the recipe is
  identical.
- **Multi-component base shear**: stack the two shears with
  `np.hypot(er.df["Vy_1"], er.df["Vz_1"])` before summing for a magnitude.
- **Residual base shear** (nonlinear pushover): replace `base_shear` with
  `base_shear.iloc[-3:].mean()` to get the value at the residual step
  rather than the time history.
- **Direct from globalForce**: `results_name="force"` carries the same
  information in *global* axes (`Px_1`, `Py_1`, `Pz_1`); skip the
  canonical layer and sum the global x or y force directly.

For the broader API tour on this fixture see the
[Elastic frame example](../examples/elastic_frame.md).
