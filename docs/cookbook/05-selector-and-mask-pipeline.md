# Find the highly-loaded beams in a region — selector + mask pipeline

> Locate every beam-column inside a chosen plan region whose absolute
> peak bending moment over the seismic window exceeds a yield-style
> threshold, then plot the survivors. Demonstrates the
> chainable-selector + result-mask pipeline end-to-end.

The new pipeline has two layers, executed in this order:

1. **`ds.elements.select()` — pre-fetch.** Chainable, lazy queries
   over the cached element index. Decides *which elements* to read
   from disk before any HDF5 access. Spatial primitives, type/selection
   anchors, and `&` / `|` / `~` boolean composition.
2. **`er.where(...)` — post-fetch.** Builds a per-element boolean
   `ResultMask` from a value condition over a time window. Compose
   masks with `&` / `|` / `~`; apply with `er[mask]` to get a fresh
   `ElementResults` trimmed to the matched elements.

The example below uses the small
`stko_results_examples/elasticFrame/elasticFrame_mesh_displacementBased_results`
fixture: 11 `64-DispBeamColumn3d` elements with five Lobatto integration
stations each. Pushover stage is `MODEL_STAGE[2]`.

```python
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from STKO_to_python import MPCODataSet

REPO_ROOT = Path("path/to/STKO_to_python")
DATASET = (
    REPO_ROOT / "stko_results_examples" / "elasticFrame"
    / "elasticFrame_mesh_displacementBased_results"
)
PUSHOVER_STAGE = "MODEL_STAGE[2]"

ds = MPCODataSet(str(DATASET), "results", verbose=False)
```

---

## 1. Pre-fetch: pick beams in a region

The model is small enough that we *could* fetch every beam — but the
recipe stays exactly the same on a 100k-element job. Build a selector,
inspect the resolved id list, then hand it to the fetch path.

```python
sel = (ds.elements.select()
       .of_type("DispBeamColumn3d")          # universe anchor
       .within_box(min=(-0.5, -0.5, 0.0),    # ground-floor band
                   max=( 9.5,  9.5, 4.0))
       .centroid_in("z", lo=0.0, hi=4.0))    # belt-and-braces

print(sel)
# ElementSelector(of_type='DispBeamColumn3d', WithinBoxOp, CentroidInOp)
print(sel.count(), "beams matched")
print(sel.ids().tolist())
```

`sel.df()` returns the matching rows of the element-index DataFrame
(useful for sanity checking — element_id, decorated_type, node_list,
centroids).

### Composing two anchors

The same shape works with named selection sets — a selector can mix
spatial geometry with STKO selection sets via `&`/`|`. Each leaf needs
its own anchor:

```python
near_origin = (ds.elements.select()
               .of_type("DispBeamColumn3d")
               .within_distance(point=(0.0, 0.0, 0.0), radius=3.0))

corner_set = (ds.elements.select()
              .of_type("DispBeamColumn3d")
              .from_selection("CornerBeams"))   # if present

both = near_origin & corner_set
either = near_origin | corner_set
not_corner = (ds.elements.select()
              .of_type("DispBeamColumn3d")
              .from_selection("CornerBeams"))    # universe = corner beams
just_other_corners = ~not_corner   # same universe, complement → other beams
```

The `~` rule is strict: a leaf must declare its universe via
`.of_type(...)`, `.from_selection(...)`, or `.with_ids(...)` for
negation to be defined. An unanchored `~sel` raises a
`ValueError` rather than silently negating against every element in
the model.

---

## 2. Fetch results, scoped by the selector

`get_element_results` accepts a selector directly — its anchor's
`of_type` becomes the fetch's `element_type`, and its resolved ids
become the `element_ids` filter:

```python
er = ds.elements.get_element_results(
    results_name="section.force",
    selector=sel,
    model_stage=PUSHOVER_STAGE,
)
print(er)
# ElementResults(results_name='section.force',
#                element_type='DispBeamColumn3d',
#                n_elements=…, n_steps=10, n_components=20, n_ip=5, …)
```

You can still pass `element_type=` and `element_ids=` explicitly; the
selector's ids are unioned with anything else you pass.

---

## 3. Build a result mask

Imagine we want to flag every beam whose absolute peak `M_y` over the
back half of the pushover exceeds a yield-style threshold of
`5e6 N·m` at *any* IP. We'll handle the per-IP question in two ways
to show both styles.

### 3a. Pick one IP and threshold its abs-peak

`component(name)` is the simplest path — pick a single column and
reduce over time:

```python
m_ip2 = (er.where(time=(0.0, 5.0))                # default window
         .component("My_ip2")                      # mid-element station
         .abs_peak()                               # |peak| over window
         .gt(5e6))

print(m_ip2)            # ResultMask(n_true=…, n_total=…)
print(m_ip2.ids())      # element ids that pass
hot = er[m_ip2]         # fresh ElementResults
```

### 3b. Combine masks across IPs

For "any IP exceeds the threshold", build one mask per IP and OR them
together:

```python
ip_cols = er.canonical_columns("bending_moment_y")    # ('My_ip0', …, 'My_ip4')

masks = [
    er.where().component(col).abs_peak().gt(5e6)
    for col in ip_cols
]
any_ip_hot = masks[0]
for m in masks[1:]:
    any_ip_hot = any_ip_hot | m

hot = er[any_ip_hot]
print(hot.element_ids, "out of", er.element_ids)
```

The ergonomic shortcut for "any step where any IP exceeds X" is the
`predicate` escape hatch — the full `(element_id, step)` index lets you
bake your own per-row boolean and the chain reduces it via `any` per
element:

```python
any_ip_any_step = er.where().predicate(
    lambda df: df[list(ip_cols)].abs().max(axis=1) > 5e6
)
hot = er[any_ip_any_step]
```

---

## 4. Compose: selector AND mask in one shot

Spatial filter (Layer A) and value filter (Layer B) compose naturally
because they live in different objects: layer A produces `element_ids`
that go into the fetch; layer B produces a `ResultMask` that goes into
`er[...]`.

```python
sel = (ds.elements.select()
       .of_type("DispBeamColumn3d")
       .centroid_in("z", lo=0.0, hi=4.0)
       .nearest_to((4.5, 4.5, 2.0), k=8))           # 8 beams near a column line

er = ds.elements.get_element_results(
    "section.force", selector=sel, model_stage=PUSHOVER_STAGE,
)

mask = (er.where(time=(2.0, 5.0))
        .component("My_ip2").abs_peak().gt(5e6)
        & er.where().component("N_ip2").trough().lt(0.0))   # also in compression

hot = er[mask]
print(f"{hot.n_elements} beams: in box, near (4.5,4.5,2), |My_ip2| > 5e6, axially compressed")
```

`hot` is a fully-formed `ElementResults` — pickle-able, plottable,
indexable. `er.plot.history`, `er.plot.diagram`, `er.physical_coords`,
and `er.integrate_canonical` all work on it.

---

## 5. Inspect intermediate values for tuning

Every reduction is also exposed as a `pd.Series` so you can pick a
threshold informed by the data instead of hard-coding it:

```python
peaks = er.where().component("My_ip2").abs_peak().values()
print(peaks.describe())     # min/max/mean/std per element
threshold = peaks.quantile(0.75)    # top 25%

mask = er.where().component("My_ip2").abs_peak().gt(threshold)
hot = er[mask]
```

The `over_threshold(v)` reduction returns the *fraction* of steps
above `v`, so chain a comparator to find elements that spend a
significant share of the window above some level:

```python
mask = (er.where(time=(0.0, 5.0))
        .component("My_ip2")
        .over_threshold(3e6)
        .gt(0.25))           # at least 25% of the window above 3 MN·m
```

---

## 6. Plot the survivors

The plotting helpers carry over verbatim — `hot` behaves exactly like
the original `er`, just with fewer rows:

```python
fig, ax = plt.subplots(figsize=(7, 4))
hot.plot.history("My_ip2", ax=ax)
ax.axhline( 5e6, color="k", lw=0.7, ls="--")
ax.axhline(-5e6, color="k", lw=0.7, ls="--")
ax.set_ylabel("M_y at IP 2 (N·m)")
ax.set_title(f"{hot.n_elements} beams above the threshold")
plt.show()
```

---

## Cheat sheet

| Goal | One-liner |
|---|---|
| All beams of a class | `ds.elements.select().of_type("DispBeamColumn3d").ids()` |
| In a 3-D AABB | `…of_type("…").within_box(min=(0,0,0), max=(10,10,10)).ids()` |
| Within radius of a point | `…within_distance((5,5,5), r=2.0).ids()` |
| k nearest to a point | `…nearest_to((5,5,5), k=10).ids()` |
| Crossing a plane | `…on_plane(z=2.5).ids()` (or `point=, normal=`) |
| Inside a slab | `…centroid_in("z", lo=2.0, hi=4.0).ids()` |
| In named selection set | `…from_selection("Walls").ids()` |
| Combine | `(a & b).ids()`, `(a | b).ids()`, `(~a).ids()` |
| Threshold abs-peak | `er.where().component(col).abs_peak().gt(v)` |
| At a step | `er.where().component(col).at_step(s).between(lo, hi)` |
| Over a time window | `er.where(time=(t0, t1)).component(col).peak().gt(v)` |
| Over time as fraction | `er.where().component(col).over_threshold(v).gt(0.25)` |
| Composing masks | `m1 & m2`, `m1 | m2`, `~m1` |
| Apply | `er[mask]` → fresh `ElementResults` |

For the full API reference and the time-spec grammar see
[ElementResults — Element selectors](../api/element-results.md#element-selectors-pre-fetch)
and [ElementResults — Result masks](../api/element-results.md#result-masks-post-fetch).
