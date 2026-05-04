# Selector + mask pipeline — complete guide

> A comprehensive walkthrough of `ds.elements.select()` (pre-fetch
> spatial / metadata queries) and `er.where(...)` (post-fetch threshold
> / time-window filters), with worked numerical output on a small
> fixture so you can see what each call returns. Use this as a
> reference whenever you're building filter chains.

The pipeline has two layers:

```
┌──────────────────────────────────────────────────────────────────┐
│ Layer A — ElementSelector  (pre-fetch, ds.elements.select())     │
│   Lazy, immutable, runs against the cached element index.        │
│   Returns:  np.ndarray[int64] of element_ids                     │
│   Filters:  type, selection-set, spatial (point/line/plane/box)  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼  feeds element_ids into the fetch
                  get_element_results(selector=...)
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Layer B — ResultMask  (post-fetch, er.where(...))                │
│   Operates on the (element_id, step) × component DataFrame.      │
│   Returns:  ResultMask (fresh ElementResults via er[mask])       │
│   Filters:  component-threshold, time-window, predicate          │
└──────────────────────────────────────────────────────────────────┘
```

The two layers compose naturally because they live in different
objects. Selectors produce ids that go into the fetch; masks produce
boolean filters that go into `er[...]`. You can use either layer
alone, or both.

---

## 0. When to use what

| You want… | Use |
|---|---|
| Beams in a region (no results read yet) | Layer A |
| Beams in a named selection set | Layer A |
| Beams whose peak `M_y` exceeded 50 kN·m | Layer B |
| Beams in a region AND whose peak exceeded a threshold | Both |
| Custom spatial logic (e.g. annular ring) | Layer A `.where(fn)` |
| Custom value logic (e.g. spent ≥40% of run above threshold) | Layer B `.over_threshold(...)` or `.predicate(fn)` |

Decision rule: if the filter only needs geometry / metadata, do it in
Layer A — it's free (no HDF5 reads). If it needs the time-history
values, it has to be Layer B.

---

## 1. Fixture and setup

Every code block below is runnable against the checked-in
`elasticFrame` fixture:

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from STKO_to_python import MPCODataSet

REPO_ROOT = Path("path/to/STKO_to_python")
DATASET = (
    REPO_ROOT
    / "stko_results_examples"
    / "elasticFrame"
    / "elasticFrame_mesh_displacementBased_results"
)
PUSHOVER_STAGE = "MODEL_STAGE[2]"

ds = MPCODataSet(str(DATASET), "results", verbose=False)
```

The fixture is small and predictable — 11 `64-DispBeamColumn3d`
elements with five Lobatto integration stations each (`gp_xi =
[-1, -0.65, 0, +0.65, +1]`). This makes it easy to reason about
what each filter should produce.

### 1.1 Discovery — what's in the file?

```python
print(ds.unique_element_types)
# ['64-DispBeamColumn3d[1000:1]']

print(list(ds.number_of_steps.items()))
# [('MODEL_STAGE[1]', 10), ('MODEL_STAGE[2]', 10)]

print(ds.elements_info["dataframe"].head())
#    element_id  element_idx  file_id          element_type ...    centroid_x  centroid_y  centroid_z
# 0           1            0        0  64-DispBeamColumn3d ...           0.0         0.0         1.5
# 1           2            1        0  64-DispBeamColumn3d ...           0.0         0.0         4.5
# ...
```

The element-index DataFrame is the universe Layer A queries against.
Columns: `element_id`, `element_idx`, `file_id`, `element_type`,
`decorated_type`, `node_list`, `num_nodes`, `centroid_x/y/z`. Spatial
primitives use the centroid columns; node-level filters explode
`node_list` against the node-coordinate table.

---

## 2. Layer A — ElementSelector

### 2.1 The empty selector

```python
sel = ds.elements.select()
print(sel)
# ElementSelector()
print(sel.count())                 # 11
print(sel.ids().tolist())          # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

An unanchored selector resolves over **every element in the index**
(both element classes if your model has more than one).

### 2.2 Anchors — set the universe

Three ways to anchor:

```python
# By element class (decorated [bracket] suffix is stripped automatically)
beams = ds.elements.select().of_type("DispBeamColumn3d")
beams = ds.elements.select().of_type("64-DispBeamColumn3d")            # also OK
beams = ds.elements.select().of_type("64-DispBeamColumn3d[1000:1]")    # also OK
print(beams.count())                # 11

# By selection set (name or set-id)
walls = ds.elements.select().from_selection("Walls")          # if present
walls = ds.elements.select().from_selection(7)                # by id
walls = ds.elements.select().from_selection(["Walls", "Slabs"])  # several

# By explicit ids
some = ds.elements.select().with_ids([1, 5, 9])
print(some.count())                 # 3
```

Anchors are the universe used by negation (`~`) — see §3.3.

You can stack anchors (intersection):

```python
sel = (ds.elements.select()
       .of_type("DispBeamColumn3d")
       .with_ids([1, 5, 9, 99]))
print(sel.ids().tolist())   # [1, 5, 9]   ← 99 is not in the index
```

### 2.3 Spatial primitives

Every primitive returns a new selector with one filter op appended.
Chaining is AND-narrowing, in the source order you wrote.

#### 2.3.1 `within_box`

```python
sel = (ds.elements.select()
       .of_type("DispBeamColumn3d")
       .within_box(min=(-0.1, -0.1, 0.0), max=(0.1, 0.1, 6.0)))
print(sel.count(), sel.ids().tolist())
# 4 [1, 2, 3, 4]    ← left column of the frame, all stories
```

`mode=` controls how "inside" is decided:

| `mode` | Meaning |
|---|---|
| `"centroid"` (default) | Element centroid ∈ box |
| `"any_node"` | At least one node ∈ box |
| `"all_nodes"` | Every node ∈ box |

The centroid mode uses pre-computed columns and is the fastest. The
node modes need to look up coordinates per node — slower but lets
you catch elements that *cross* the box even if their centroid is
outside:

```python
# Tiny box at the origin — only one node is here.
narrow = dict(min=(-0.05, -0.05, -0.05), max=(0.05, 0.05, 0.05))

print(ds.elements.select().of_type("DispBeamColumn3d")
      .within_box(**narrow, mode="centroid").count())   # 0
print(ds.elements.select().of_type("DispBeamColumn3d")
      .within_box(**narrow, mode="any_node").count())   # 1
print(ds.elements.select().of_type("DispBeamColumn3d")
      .within_box(**narrow, mode="all_nodes").count())  # 0
```

#### 2.3.2 `within_distance`

Centroid distance to a point.

```python
sel = (ds.elements.select()
       .of_type("DispBeamColumn3d")
       .within_distance(point=(0.0, 0.0, 1.5), radius=0.1))
print(sel.ids().tolist())   # [1]   ← only the column whose centroid is at (0,0,1.5)
```

#### 2.3.3 `nearest_to`

k-NN by centroid. Returns rows sorted by ascending distance with a
stable tie-break:

```python
sel = (ds.elements.select()
       .of_type("DispBeamColumn3d")
       .nearest_to(point=(0.0, 0.0, 0.0), k=3))
print(sel.ids().tolist())
# [1, 5, 2]   ← element 1 is closest, then 5, then 2
```

`k=0` gives an empty selector. `k > n_elements` returns all elements,
sorted by distance.

#### 2.3.4 `on_plane`

Element *crosses* a plane (any node on each side, or any node on the
plane within `tol`). Two forms — axis-aligned and general:

```python
# Axis-aligned: pick exactly one of x=, y=, z=
sel = (ds.elements.select()
       .of_type("DispBeamColumn3d")
       .on_plane(z=3.0, tol=1e-6))
print(sel.ids().tolist())   # [1, 2, 3, 4]   ← columns straddle z=3 between stories

# General plane via point + normal
sel = (ds.elements.select()
       .of_type("DispBeamColumn3d")
       .on_plane(point=(2.5, 0.0, 0.0), normal=(1.0, 0.0, 0.0)))
print(sel.count())   # ← beams whose nodes straddle x=2.5
```

Mixing axis-aligned and `point`/`normal` raises `ValueError`. A
zero-length normal raises too.

#### 2.3.5 `near_line`

Centroid distance to a line *segment* (clamped at the endpoints).

```python
sel = (ds.elements.select()
       .of_type("DispBeamColumn3d")
       .near_line(p0=(0.0, 0.0, 0.0), p1=(0.0, 0.0, 9.0), radius=0.1))
print(sel.ids().tolist())   # [1, 2, 3]   ← left column line
```

#### 2.3.6 `centroid_in`

One- or two-sided range on a single centroid axis. Cleaner than
`within_box` when you only care about one dimension:

```python
# Slab between z=3 and z=6
mid = (ds.elements.select()
       .of_type("DispBeamColumn3d")
       .centroid_in("z", lo=3.0, hi=6.0))
print(mid.count())          # ← beams whose centroid_z ∈ [3, 6]

# One-sided
ground = ds.elements.select().of_type("DispBeamColumn3d").centroid_in("z", hi=3.0)
upper  = ds.elements.select().of_type("DispBeamColumn3d").centroid_in("z", lo=6.0)
```

`lo=hi=None` raises — pass at least one bound. The axis is `"x"`,
`"y"`, or `"z"` (case-insensitive).

#### 2.3.7 `where(fn)` — predicate escape hatch

When a primitive doesn't fit your shape, drop down to pandas. The
function takes the element-index DataFrame (already narrowed by the
anchors) and must return a boolean array of the same length.

```python
# Annular region between r=1 and r=2 in the xy-plane
def annulus(df):
    r = np.sqrt(df["centroid_x"]**2 + df["centroid_y"]**2)
    return ((r >= 1.0) & (r <= 2.0)).to_numpy()

sel = (ds.elements.select()
       .of_type("DispBeamColumn3d")
       .where(annulus))
print(sel.count())

# Even-numbered elements only (silly but legal)
sel = (ds.elements.select()
       .of_type("DispBeamColumn3d")
       .where(lambda df: (df["element_id"] % 2 == 0).to_numpy()))
```

If `fn` returns a wrong shape it raises `ValueError("predicate
returned shape (N,), expected (M,)")`.

### 2.4 Inspection methods

```python
sel = (ds.elements.select()
       .of_type("DispBeamColumn3d")
       .within_box(min=(-0.1, -0.1, 0.0), max=(0.1, 0.1, 6.0)))

sel.ids()        # np.ndarray[int64]   matched element_ids
sel.df()         # pd.DataFrame        rows from the element index
sel.mask()       # pd.Series[bool]     indexed by element_id over the universe
sel.count()      # int
repr(sel)        # 'ElementSelector(of_type=..., WithinBoxOp)'
```

`mask()` indexes by the universe, not the full model — `True` where
the element matched, `False` for the rest of the same `of_type` set.

### 2.5 Order of operations matters

Chained primitives are applied left-to-right.
`within_box(...) → nearest_to(p, k=10)` returns "the 10 nearest
elements *that are also in the box*" (could be fewer than 10).
`nearest_to(p, k=10) → within_box(...)` returns "the 10 nearest
globally, intersected with the box" (could be 0). Pick the order that
expresses your question.

```python
# 5 nearest beams that are inside the bounding box
sel = (ds.elements.select()
       .of_type("DispBeamColumn3d")
       .within_box(min=(-0.1, -0.1, 0.0), max=(0.1, 0.1, 6.0))
       .nearest_to((0.0, 0.0, 1.5), k=5))

# Of the 5 nearest globally, which are inside the box?
sel2 = (ds.elements.select()
        .of_type("DispBeamColumn3d")
        .nearest_to((0.0, 0.0, 1.5), k=5)
        .within_box(min=(-0.1, -0.1, 0.0), max=(0.1, 0.1, 6.0)))
```

---

## 3. Boolean composition

`&`, `|`, `~` operate on the resolved id sets — `np.intersect1d`,
`np.union1d`, `np.setdiff1d`.

### 3.1 Intersection

```python
left_column = (ds.elements.select()
               .of_type("DispBeamColumn3d")
               .within_box(min=(-0.1, -0.1, 0.0), max=(0.1, 0.1, 9.0)))

mid_height  = (ds.elements.select()
               .of_type("DispBeamColumn3d")
               .centroid_in("z", lo=3.0, hi=6.0))

both = left_column & mid_height
print(both.ids().tolist())   # left-column beams in the mid story
```

### 3.2 Union

```python
ground = ds.elements.select().of_type("DispBeamColumn3d").centroid_in("z", hi=3.0)
roof   = ds.elements.select().of_type("DispBeamColumn3d").centroid_in("z", lo=6.0)
endpoints = ground | roof          # every beam at the ends
```

### 3.3 Negation — the universe rule

`~sel` is "everything in `sel`'s anchor universe that isn't in `sel`"
— **never** the complement against the whole model. The universe is
always the intersection of `of_type` ∩ `from_selection` ∩ `with_ids`.

```python
hot_box = (ds.elements.select()
           .of_type("DispBeamColumn3d")
           .within_box(min=(-0.1, -0.1, 0.0), max=(0.1, 0.1, 6.0)))

print(hot_box.count())          # 4
print((~hot_box).count())       # 7   ← the OTHER beams (not all 11 in the model)
```

If you negate without an anchor, the call raises:

```python
~ds.elements.select().within_box(...)
# ValueError: Cannot negate a selector without an of_type/from_selection/
# with_ids anchor — call .of_type(...) first to define the universe.
```

This is intentional. "Everything except this 5-beam set" without
declaring whether you mean *of which class* would silently include
shells, bricks, links — almost never what the user meant.

### 3.4 Composing combinators

`&` / `|` / `~` chain to any depth. The combinator's *own* universe is
derived recursively:

| Op | Universe |
|---|---|
| `(a & b)._universe` | `a._universe ∩ b._universe` |
| `(a | b)._universe` | `a._universe ∪ b._universe` |
| `(~a)._universe`    | `a._universe` |

So `~(a & b)` is "elements in a's universe AND b's universe, that are
not in (a∩b)" — well-defined. `~(a | b)` is "elements in a's universe
OR b's universe, that are not in (a∪b)" — also well-defined, even
when `a` and `b` anchor to different element classes.

```python
beams = ds.elements.select().of_type("DispBeamColumn3d").within_box(...)
shells = ds.elements.select().of_type("ASDShellQ4").within_distance(...)
print((~(beams | shells)).count())   # everything else in the union universe
```

### 3.5 What you can't do on a combinator

A combinator has no anchor or filter chain of its own — chaining
primitives or anchors on `(a & b)` raises:

```python
combo = a & b
combo.of_type("OtherClass")
# TypeError: Cannot call .of_type() on a combined selector; anchor each
# leaf selector before combining.
```

If you need that, anchor on the leaves and combine again. This rule
is what makes the universe deterministic.

---

## 4. Wiring the selector into the fetch

`get_element_results(selector=...)` reads the selector's ids and
inherits its `of_type` anchor (if `element_type=` is omitted):

```python
sel = (ds.elements.select()
       .of_type("DispBeamColumn3d")
       .centroid_in("z", lo=3.0, hi=6.0))

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

You can still pass `element_ids=` and `element_type=` explicitly; the
selector's ids are *unioned* with `element_ids=`. If both the selector
has an `of_type` anchor and you pass `element_type=`, the explicit
`element_type=` wins.

### 4.1 Without a selector

The classical signature still works — selectors are an *addition*,
not a replacement:

```python
er = ds.elements.get_element_results(
    results_name="section.force",
    element_type="DispBeamColumn3d",
    selection_set_name="Walls",
    element_ids=[1, 5, 9],         # unioned with the set
    model_stage=PUSHOVER_STAGE,
)
```

---

## 5. Layer B — ResultMask

`er.where(time=...)` returns a query object. Chain a column choice, a
reduction over time, and a comparator to produce a `ResultMask`.

### 5.1 The shape of the chain

```
er.where(time=window)          # default time window for the chain
   .component("Mz_ip0")          # OR  .canonical("axial_force")  OR  .predicate(fn)
   .abs_peak(time=...)           # reduction → one scalar / element
   .gt(50.0)                     # comparator → ResultMask
```

Every step is a fresh immutable object — chains are safe to share
or branch.

### 5.2 Picking the column

#### 5.2.1 `component(name)` — exact column name

```python
m = er.where().component("Mz_ip0").abs_peak().gt(50.0)
```

The name must match a column in `er.df` exactly. If not it raises:

```python
er.where().component("nope")
# ValueError: component 'nope' not in this ElementResults.
# Available: ['N_ip0', 'Vy_ip0', ..., 'Mz_ip4']
```

#### 5.2.2 `canonical(name)` — engineering-friendly name

For canonicals that resolve to exactly one column (e.g. closed-form
buckets where `axial_force` → just `N_1`), `canonical` is the cleanest
form. For multi-IP buckets where the canonical resolves to several
columns, you must pick a specific column with `component`:

```python
print(er.canonical_columns("bending_moment_y"))
# ('My_ip0', 'My_ip1', 'My_ip2', 'My_ip3', 'My_ip4')

er.where().canonical("bending_moment_y")
# ValueError: canonical 'bending_moment_y' resolves to 5 columns
# (['My_ip0', 'My_ip1', 'My_ip2', 'My_ip3', 'My_ip4']);
# pick one via .component(name).
```

(Multi-column reduction is on the roadmap as an explicit
`.over_ips(...)` step. For now, build one mask per IP and OR them
— see §7.4.)

### 5.3 Reductions over time

Each reduction collapses the (element_id, step) DataFrame to one
scalar per element.

| Reduction | Definition |
|---|---|
| `at_step(s)` | value at step `s` |
| `at_time(t)` | value at the step nearest to `t` |
| `peak(time=...)` | signed max over the window |
| `trough(time=...)` | signed min over the window |
| `abs_peak(time=...)` | max of `|·|` over the window |
| `mean(time=...)` | arithmetic mean over the window |
| `residual(time=...)` | last step in the window |
| `over_threshold(v, time=...)` | fraction of steps with value > `v` |

The reductions return a `_ScalarPerElement` — an internal type that
exposes `.values()` (the underlying Series for inspection) and the
comparator methods.

```python
peaks = er.where().component("Mz_ip0").abs_peak().values()
print(peaks)
# element_id
# 1     7.823e+05
# 2     1.244e+06
# 3     2.701e+07
# ...
print(peaks.describe())     # min, max, mean, std, quantiles
```

Use `.values()` to pick a data-driven threshold rather than hard-coding
one (§9).

### 5.4 Comparators

`_ScalarPerElement.<op>(...)` returns the final `ResultMask`.

| Comparator | Semantics |
|---|---|
| `.gt(v)` | `value > v` |
| `.lt(v)` | `value < v` |
| `.ge(v)` | `value ≥ v` |
| `.le(v)` | `value ≤ v` |
| `.between(lo, hi, inclusive=True)` | `lo ≤ value ≤ hi` (or strict) |
| `.outside(lo, hi, inclusive=False)` | `value < lo` or `value > hi` |
| `.eq(v, atol=0)` | exact equality (`atol > 0` falls through to `.near`) |
| `.near(v, atol)` | `|value - v| ≤ atol` |

```python
hot   = er.where().component("Mz_ip2").abs_peak().gt(5e6)
cold  = er.where().component("Mz_ip2").abs_peak().lt(1e5)
band  = er.where().component("Mz_ip2").peak().between(-5e6, 5e6)
extreme = er.where().component("Mz_ip2").peak().outside(-5e6, 5e6)
exact = er.where().component("Mz_ip2").at_step(0).eq(0.0)        # initial-rest check
near0 = er.where().component("Mz_ip2").at_step(0).near(0.0, atol=1e-9)
```

### 5.5 Time-spec grammar

The `time=` argument on `er.where(...)` and on every reduction
accepts:

| Spec | Meaning |
|---|---|
| `None` | all steps in `er.time` |
| `int` | one step index (negative wraps from the end) |
| `float` | step nearest to that time value |
| `slice(t0, t1)` | half-open *time* range: `t0 ≤ time < t1` |
| `(t0, t1)` tuple | same as the slice form (Python sugar) |
| `list[int]` / `np.ndarray[int]` | explicit step indices |
| `list[float]` / `np.ndarray[float]` | nearest step for each |

A few worked examples on a 5-step fixture with `time = [0, 1, 2, 3, 4]`:

```python
from STKO_to_python.elements.result_mask import resolve_step_indices
import numpy as np

t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

resolve_step_indices(None, t)              # [0, 1, 2, 3, 4]    all steps
resolve_step_indices(2, t)                 # [2]                step 2
resolve_step_indices(-1, t)                # [4]                last step
resolve_step_indices(2.4, t)               # [2]                nearest to 2.4
resolve_step_indices(slice(1.0, 3.0), t)   # [1, 2]             1 ≤ time < 3
resolve_step_indices((1.0, 4.0), t)        # [1, 2, 3]          tuple form
resolve_step_indices([0, 2, 4], t)         # [0, 2, 4]          list of int
resolve_step_indices([0.4, 2.6], t)        # [0, 3]             nearest of each float
```

Slices and tuples are *time* values, not step indices. To slice by
step index, pass an explicit list. (Mixing the two would be a
disaster; the dispatch is type-driven.)

### 5.6 Default vs explicit time window

`er.where(time=window)` sets a default window for *every* reduction in
the chain. Each reduction can override with its own `time=`.

```python
# Default window: first half of the pushover (time 0-5)
ew = er.where(time=(0.0, 5.0))

m1 = ew.component("Mz_ip0").peak().gt(1e6)            # uses default
m2 = ew.component("N_ip0").trough().lt(-1e3)          # uses default
m3 = ew.component("Mz_ip0").peak(time=None).gt(1e6)   # OVERRIDE: all steps
m4 = ew.component("Mz_ip0").peak(time=(2.0, 4.0)).gt(1e6)  # OVERRIDE: narrow
```

`er.where()` with no argument → default is `None` (all steps).

---

## 6. Mask composition

Masks compose with `& / | / ~` — the same algebra as selectors, but
operating on per-element bool series instead of id sets.

```python
m_peak = er.where().component("Mz_ip2").abs_peak().gt(5e6)
m_axial = er.where().component("N_ip2").trough().lt(-1e4)

both = m_peak & m_axial               # AND
either = m_peak | m_axial             # OR
not_peak = ~m_peak                    # complement vs all elements in `er`
```

Negation on a mask doesn't need an anchor — the universe is fixed
(every element in the parent `ElementResults`). Combining masks across
**different** `ElementResults` instances raises `ValueError`. If you
need to compare results from two fetches, build the masks separately
and combine the *id sets* manually.

A mask exposes:

```python
m = er.where().component("Mz_ip0").abs_peak().gt(1e6)

m.ids()       # np.ndarray[int64] of matched ids
m.mask()      # pd.Series[bool] indexed by element_id (over the parent's full id set)
m.count()     # int
m.apply()     # fresh ElementResults (== er[m])
len(m)        # same as m.count()
repr(m)       # 'ResultMask(n_true=…, n_total=…)'
```

---

## 7. The predicate escape hatch

When the value condition doesn't fit a `component → reduction →
comparator` shape, drop into pandas via `er.where().predicate(fn)`.

### 7.1 Per-element form

The function returns a 1-D boolean array of length `n_elements`,
aligned with `er.element_ids`. Used directly as the mask:

```python
m = er.where().predicate(
    lambda df: np.array([eid % 2 == 0 for eid in er.element_ids])
)
print(m.ids().tolist())   # [2, 4, 6, 8, 10]
```

### 7.2 Full-index form

The function operates on the full `(element_id, step)` DataFrame and
returns a bool Series of the same length. The chain reduces it via
**`any` per element** to a per-element mask:

```python
# Any element where |Mz_ip2| > 5e6 at *any* step
ip_cols = er.canonical_columns("bending_moment_z")
m = er.where().predicate(
    lambda df: df[list(ip_cols)].abs().max(axis=1) > 5e6
)
print(m.ids())
```

If you need *all* steps satisfied, replace the `any` with `all`
explicitly — call `.values()` on a comparator or just do the reduction
yourself before the predicate.

### 7.3 Bad shape → loud error

```python
er.where().predicate(lambda df: np.array([True, False]))   # wrong length
# ValueError: predicate(fn): returned shape (2,); expected (11,) or (110,).
```

### 7.4 Pattern: "any IP exceeds threshold X" the long way

When you want each IP individually thresholded and OR'ed (so you can
mix thresholds per IP):

```python
ip_cols = er.canonical_columns("bending_moment_y")
masks = [
    er.where().component(col).abs_peak().gt(5e6)
    for col in ip_cols
]

any_ip_hot = masks[0]
for m in masks[1:]:
    any_ip_hot = any_ip_hot | m

hot = er[any_ip_hot]
print(hot.element_ids)
```

The predicate-form is shorter when the threshold is uniform across IPs:

```python
hot = er[
    er.where().predicate(
        lambda df: df[list(ip_cols)].abs().max(axis=1) > 5e6
    )
]
```

---

## 8. Apply: `er[mask]`

`er[mask]` returns a fresh `ElementResults` trimmed to the matched
ids:

```python
mask = er.where().component("Mz_ip2").abs_peak().gt(5e6)
hot = er[mask]
```

What's preserved in the subset:

| Attribute | Behavior |
|---|---|
| `df` | trimmed to matched element_ids (all steps, all components) |
| `element_ids` | sorted tuple of matched ids |
| `element_node_coords`, `element_node_ids` | trimmed and aligned to the new id order |
| `time` | preserved as-is |
| `gp_xi`, `gp_natural`, `gp_weights` | preserved |
| `model_stage`, `model_stages`, `stage_step_ranges` | preserved |
| `name`, `element_type`, `results_name` | preserved |

The result is a fully-formed `ElementResults` — pickle-able,
plottable, indexable, integrable, and you can run **another**
`er.where(...)` chain on it to refine further:

```python
# Stage 1: spatial filter via Layer A
sel = ds.elements.select().of_type("DispBeamColumn3d").centroid_in("z", lo=3.0, hi=6.0)
er = ds.elements.get_element_results("section.force", selector=sel,
                                     model_stage=PUSHOVER_STAGE)

# Stage 2: peak filter via Layer B
hot = er[er.where().component("Mz_ip2").abs_peak().gt(5e6)]

# Stage 3: ALSO under axial compression
hot2 = hot[hot.where().component("N_ip2").trough().lt(0.0)]

print(hot2)
```

`er[non_mask]` raises `TypeError` — column access still goes through
`er.<colname>` (e.g. `er.Mz_ip0`), not `er["Mz_ip0"]`.

---

## 9. Inspecting intermediate values for tuning

Instead of guessing thresholds, look at the underlying scalars first:

```python
peaks = er.where().component("Mz_ip2").abs_peak().values()
print(peaks.sort_values(ascending=False).head(10))
# element_id
# 3     2.701e+07
# 11    2.701e+07
# 8     1.705e+07
# ...

q75 = peaks.quantile(0.75)
print(f"top 25% threshold: {q75:.3g}")

mask = er.where().component("Mz_ip2").abs_peak().gt(q75)
hot = er[mask]
```

`.values()` returns a `pd.Series` indexed by `element_id` — full
pandas semantics, so `.describe()`, `.quantile()`, `.hist()`, etc.
are all available.

---

## 10. End-to-end pipeline example

The full picture: pre-fetch spatial filter → fetch → post-fetch
threshold filter → plot.

```python
# 0. Open dataset
ds = MPCODataSet(str(DATASET), "results", verbose=False)

# 1. Pre-fetch: locate beams in the mid story
sel = (ds.elements.select()
       .of_type("DispBeamColumn3d")
       .centroid_in("z", lo=3.0, hi=6.0))
print(f"selector matched {sel.count()} beams: {sel.ids().tolist()}")

# 2. Fetch section.force only for those beams, only for the pushover
er = ds.elements.get_element_results(
    results_name="section.force",
    selector=sel,
    model_stage=PUSHOVER_STAGE,
)
print(er)

# 3. Tune the threshold from the data
peaks = er.where().component("My_ip2").abs_peak().values()
threshold = peaks.quantile(0.5)        # median
print(f"threshold (median |My_ip2|): {threshold:.3g}")

# 4. Build the mask: above-threshold AND in axial compression
mask = (er.where(time=(0.0, 5.0))
        .component("My_ip2").abs_peak().gt(threshold)
        & er.where().component("N_ip2").trough().lt(0.0))

# 5. Apply
hot = er[mask]
print(f"survivors: {hot.element_ids}")

# 6. Plot the survivors' history
fig, ax = plt.subplots(figsize=(8, 4))
hot.plot.history("My_ip2", ax=ax)
ax.axhline( threshold, color="k", ls="--", lw=0.7, label=f"+{threshold:.2g}")
ax.axhline(-threshold, color="k", ls="--", lw=0.7)
ax.set_ylabel("M_y at IP 2 (N·m)")
ax.set_title(f"{hot.n_elements} mid-story beams above the median peak (and in compression)")
ax.legend(loc="upper right")
plt.show()
```

---

## 11. Common patterns

### 11.1 Threshold as fraction of run

`over_threshold(v)` returns a fraction in `[0, 1]`. Chain a comparator
to flag elements that spent enough time above `v`:

```python
# At least 25% of the window above 3e6 (positive direction only)
mask = (er.where(time=(0.0, 5.0))
        .component("My_ip2")
        .over_threshold(3e6)
        .gt(0.25))
```

### 11.2 Find when each element peaked

`time_of_peak` (a method on `ElementResults`) is the right tool:

```python
peak_steps = er.time_of_peak("My_ip2", abs=True)
# Series indexed by element_id, values are step indices
peak_times = pd.Series(er.time, name="time")[peak_steps.values].values
```

### 11.3 "First step where the threshold is breached"

`peak` and `time_of_peak` find the *largest* value. For "the first
step where the value crosses some level", drop into pandas:

```python
def first_breach_step(series_per_elem, threshold):
    first = (series_per_elem.abs() > threshold).idxmax()
    # idxmax of an all-False series returns the first index — guard:
    return first if (series_per_elem.abs() > threshold).any() else pd.NA

cols_per_elem = er.df["My_ip2"].abs().unstack("step")  # element_id × step
first_breach = cols_per_elem.apply(
    lambda row: row.idxmax() if row.max() > 5e6 else np.nan, axis=1
)
```

### 11.4 Chaining selectors and masks symmetrically

Both selectors and masks expose `&`, `|`, `~`, `.ids()`, `.count()`.
You can mirror the same combinator on both sides of the fetch:

```python
# Region A: low and on the left ; Region B: high and on the right
A = (ds.elements.select().of_type("DispBeamColumn3d")
     .centroid_in("z", hi=3.0).centroid_in("x", hi=2.5))
B = (ds.elements.select().of_type("DispBeamColumn3d")
     .centroid_in("z", lo=6.0).centroid_in("x", lo=2.5))

er = ds.elements.get_element_results("section.force",
                                     selector=A | B,
                                     model_stage=PUSHOVER_STAGE)

# In the result, separately mask each region's hot set
hotA = er[er.where().component("My_ip2").abs_peak().gt(5e6)
          & er.where().predicate(lambda df: df.index.get_level_values("element_id").isin(A.ids()))]
```

---

## 12. Common mistakes and the errors you'll see

| Mistake | Error |
|---|---|
| `~ds.elements.select().within_box(...)` | `ValueError: Cannot negate a selector without an of_type/from_selection/with_ids anchor — call .of_type(...) first to define the universe.` |
| Calling `.of_type(...)` on a combinator | `TypeError: Cannot call .of_type() on a combined selector; anchor each leaf selector before combining.` |
| `er.where().canonical("bending_moment_y")` (multi-IP) | `ValueError: canonical 'bending_moment_y' resolves to 5 columns; pick one via .component(name).` |
| `er.where().component("nope")` | `ValueError: component 'nope' not in this ElementResults. Available: [...]` |
| Combining masks across two `ElementResults` | `ValueError: Cannot AND masks from different ElementResults instances.` |
| `er[42]` | `TypeError: ElementResults[...] expects a ResultMask; got int. Use attribute access for columns.` |
| Predicate returns wrong length | `ValueError: predicate(fn): returned shape (N,); expected (n_elements,) or (n_rows,).` |
| `centroid_in("z")` with no bounds | `ValueError: centroid_in: pass at least one of lo=, hi=.` |
| Negative `radius` | `ValueError: within_distance: radius must be non-negative.` |
| `on_plane(z=2, point=..., normal=...)` | `ValueError: on_plane: cannot mix x/y/z= with point/normal.` |

---

## 13. Performance notes

- **Layer A is free.** Selectors run against the cached element-index
  DataFrame in memory. Building one — even with several primitives —
  is microsecond-scale on a 100k-element model. The cost is in `.ids()`
  / `.df()`, which actually evaluates the chain. Build selectors
  lazily, evaluate once, reuse the id array.
- **Centroid mode beats node mode by ~10×.** `within_box(mode="centroid")`
  uses three vectorized comparisons on the centroid columns. The
  `any_node` / `all_nodes` modes have to look up coordinates per node
  (a Python-level loop today). Use centroid mode unless you genuinely
  need node-level semantics.
- **Layer B is bounded by the size of `er.df`.** Once the `ElementResults`
  is built, masks are pure pandas reductions on it. They scale with
  `n_elements × n_steps × n_components`, which is usually small after
  Layer A has pruned the model.
- **Cache the fetch.** The dataset's query engine has an LRU cache on
  `get_element_results`; identical fetch arguments hit the cache.
  Selectors with the same resolved id array produce the same cache
  key. Don't fetch the whole model "to be safe" — fetch the smallest
  superset you'll need, then mask.
- **Reusing a mask.** `m.apply()` / `er[m]` rebuilds an
  `ElementResults` each call (cheap — it's a pandas slice). For
  expensive downstream work, do it once and bind the result.

---

## 14. Cheat sheet

```python
# ---- Layer A ----------------------------------------------------------
sel = ds.elements.select().of_type("Beam")            # anchor by class
       .from_selection("Walls")                        # anchor by set
       .with_ids([1, 2, 3])                            # anchor by ids
       .within_box(min=, max=, mode="centroid")        # AABB
       .within_distance(point=, radius=)               # sphere
       .nearest_to(point=, k=)                         # k-NN
       .on_plane(z=2.5)                                # plane (axis form)
       .on_plane(point=, normal=, tol=)                # plane (general)
       .near_line(p0=, p1=, radius=)                   # cylinder around segment
       .centroid_in("z", lo=, hi=)                     # axis range
       .where(lambda df: ...)                          # predicate

ids   = sel.ids();   df  = sel.df();   m = sel.mask();   n = sel.count()

(a & b).ids();   (a | b).ids();   (~a).ids()           # boolean algebra
                                                         # ~ requires anchor

er = ds.elements.get_element_results("name", selector=sel, model_stage=...)

# ---- Layer B ----------------------------------------------------------
ew = er.where(time=(0.0, 5.0))                          # default window

m = ew.component("Mz_ip0").abs_peak().gt(50)            # one column
m = ew.canonical("axial_force").peak().gt(50)            # one canonical (1 col)
m = ew.predicate(lambda df: ...)                         # escape hatch

# Reductions
.at_step(s)  .at_time(t)
.peak(time=)  .trough(time=)  .abs_peak(time=)
.mean(time=)  .residual(time=)
.over_threshold(v, time=)        # → fraction; chain a comparator

# Comparators
.gt(v)  .lt(v)  .ge(v)  .le(v)
.between(lo, hi, inclusive=)  .outside(lo, hi, inclusive=)
.eq(v, atol=)  .near(v, atol=)

# Inspection / composition
m.ids();   m.mask();   m.count();   m.apply()
m1 & m2;   m1 | m2;   ~m1
er[m]                                                   # fresh ElementResults
```

For the underlying type signatures see
[ElementResults — Element selectors](../api/element-results.md#element-selectors-pre-fetch)
and [ElementResults — Result masks](../api/element-results.md#result-masks-post-fetch).
