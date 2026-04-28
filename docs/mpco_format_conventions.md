# MPCO format conventions & gotchas

Lessons captured while wiring beam-element topologies into a sibling
library's MPCO reader (apeGmsh Phase 11b: line-stations + nodal-forces).
This document focuses on **format-level** findings — things STKO_to_python
can use directly when extending its `.mpco` parsing.

Each item is a discovery that would have saved time if known upfront.

---

## 1. `GP_X` lives on connectivity, not on results buckets

Custom-rule (force-/disp-based beam-column) integration-point coordinates
are an *attribute* on the connectivity dataset:

```
MODEL_STAGE[<n>]/MODEL/ELEMENTS/<class_tag>-<class_name>[<rule>:<cust>]
    @GP_X         (n_IP,)   float64   — natural ξ in [-1, +1]
    @INTEGRATION_RULE = [1000]
    @CUSTOM_INTEGRATION_RULE = [1]
    @CUSTOM_INTEGRATION_RULE_DIMENSION = [1]
    @GEOMETRY = [1]   # line
```

Note the **2-field bracket** `[<rule>:<cust>]` on connectivity. Results
buckets at `RESULTS/ON_ELEMENTS/section.force/<bracket>` use the **3-field**
form `[<rule>:<cust>:<hdr>]`. The `<rule>:<cust>` pair is the common key
linking them.

`GP_X` values are in natural `[-1, +1]`, **not** physical `[0, L]`. This is
counterintuitive because `ops.eleResponse(eid, "integrationPoints")` (live
openseespy) returns physical `pts[i] * L` instead — see §7.

---

## 2. META/COMPONENTS string format

A single byte-string per bucket. Two distinct shapes depending on topology:

### Line-stations (force-/disp-based beams under `section.force`)

```
"<pathInts>.<compsCsv>;<pathInts>.<compsCsv>;..."
```

`;`-separated, **one segment per integration point**. Each segment is
`<pathInts>.<compsCsv>`, where:

- `<pathInts>` = dot-separated descriptor codes from
  `ElementOutputDescriptorType` (`0=Element, 1=Gauss, 2=Section,
  3=Fiber, 4=Material`). For `section.force` on beams it's `0.1.2`
  (Element→Gauss→Section).
- `<compsCsv>` = comma-separated section response component names
  (`P`, `Mz`, `My`, `T`, `Vy`, `Vz`).

Example for 5-IP Lobatto + aggregated 6-comp section:
```
"0.1.2.P,Mz,My,T,Vy,Vz;0.1.2.P,Mz,My,T,Vy,Vz;0.1.2.P,Mz,My,T,Vy,Vz;0.1.2.P,Mz,My,T,Vy,Vz;0.1.2.P,Mz,My,T,Vy,Vz"
```

To parse: split on `;`, then for each segment take the substring after the
**last** `.` (path ints can be more than 3 deep for fibers — see §16). For
homogeneous-section beams, every segment is identical.

### Nodal forces (closed-form elastic beams under `globalForce`/`localForce`)

A **single segment** with node-suffixed names, node-major:

```
globalForce 3D: "0.Px_1,Py_1,Pz_1,Mx_1,My_1,Mz_1,Px_2,Py_2,Pz_2,Mx_2,My_2,Mz_2"
localForce  3D: "0.N_1,Vy_1,Vz_1,T_1,My_1,Mz_1,N_2,Vy_2,Vz_2,T_2,My_2,Mz_2"
globalForce 2D: "0.Px_1,Py_1,Mz_1,Px_2,Py_2,Mz_2"
localForce  2D: "0.N_1,Vy_1,Mz_1,N_2,Vy_2,Mz_2"
```

The `_1` / `_2` suffix is the element-node index. Path is just `0.` (no
Gauss/Section level — closed-form means no integration points).

---

## 3. Closed-form META has a `GAUSS_IDS=[[-1]]` sentinel

Distinct from gauss-level buckets where `GAUSS_IDS = [[0], [1], ..., [n-1]]`.
For closed-form line elements:

```
META/MULTIPLICITY     = [[1]]
META/GAUSS_IDS        = [[-1]]            ← sentinel: "no integration point"
META/NUM_COMPONENTS   = [[<flat_size>]]   ← whole flat width in one block
NUM_COLUMNS attribute = <flat_size>       ← n_nodes × n_components_per_node
```

So validation paths must accept `-1` as a valid GAUSS_ID for nodal-forces
buckets, while still rejecting it for gauss-level buckets.

---

## 4. The `customRuleIdx` axis groups elements by `(rule_type, x_vector)` identity

Two `forceBeamColumn` elements with the same `beamIntegration` AND the
same length share a `customRuleIdx`. With different lengths or rules,
they get different ones. So:

- A given bucket's `GP_X` is **uniform across its elements**.
- A model with heterogeneous beam-integrations spawns **multiple buckets per element class**.

When stitching slabs, group by bucket, not by class — same class can land
in multiple buckets.

---

## 5. `SECTION_RESPONSE_*` codes are stable ABI

Fixed in `OpenSees/SRC/material/section/SectionForceDeformation.h:52–57`:

```
1 = SECTION_RESPONSE_MZ
2 = SECTION_RESPONSE_P
3 = SECTION_RESPONSE_VY
4 = SECTION_RESPONSE_MY
5 = SECTION_RESPONSE_VZ
6 = SECTION_RESPONSE_T
```

Plus `15 = MYY`, `18 = VYZ` for warping/asymmetric sections (rarely used,
worth flagging if encountered).

The MPCO short names in META map 1:1:

| MPCO name | Code | Meaning |
|---|---|---|
| `P` | 2 | Axial force |
| `Mz` | 1 | Bending moment about local z |
| `My` | 4 | Bending moment about local y |
| `T` | 6 | Torsion |
| `Vy` | 3 | Shear in local y |
| `Vz` | 5 | Shear in local z |

---

## 6. Section response order matches `getType()` order, NOT any canonical layout

A bare `FiberSection3d` has `getType() = [P, Mz, My, T]` (codes
`(2, 1, 4, 6)`, that order). `SectionAggregator` appends in user-listed
order: `section Aggregator 11 100 Vy 101 Vz -section 1` produces
`[P, Mz, My, T, Vy, Vz]` (codes `(2, 1, 4, 6, 3, 5)`).

User-defined aggregation orders are also valid — there's no enforced
canonical layout. **Read the actual codes from META/COMPONENTS** rather
than assuming one. The `n_components` count alone is insufficient: a
4-comp section could be `(P, Mz, My, T)` OR `(P, Mz, Vy, T)` if someone
aggregated weirdly.

---

## 7. `ops.eleResponse(eid, "integrationPoints")` returns physical `pts[i] * L`

Per `ForceBeamColumn3d.cpp:3338–3346`, the live openseespy call returns
positions along the beam in **physical length** (`[0, L]`), NOT natural
parent coordinates. MPCO's `GP_X` attribute, in contrast, is in **natural
ξ ∈ [-1, +1]**.

To bridge:

```python
xi_natural = 2.0 * xi_phys / L - 1.0
# where L = ||coords[node_j] - coords[node_i]|| from connectivity
```

Easy to miss because both APIs talk about "integration points" without
flagging the unit difference. Burned 30 minutes in a debugging session
before we caught it.

---

## 8. Disp-based beam-columns don't expose `integrationPoints` (live)

`DispBeamColumn{2d,3d}` and family don't implement
`getResponse("integrationPoints")` in OpenSees v3.7.x. MPCO writes their
`GP_X` to disk anyway via internal Tier 2/3 probing in
`MPCORecorder.cpp:4089–4265` (a `setResponse(["section", "dummy"], ...)`
trick), so `.mpco` files contain the data. But any tool doing **live**
openseespy introspection (e.g. an in-memory recorder) will silently miss
disp-based beams.

For STKO_to_python: not directly relevant if you only read `.mpco` files,
but worth knowing if you ever extend to live capture.

---

## 9. `globalForce` / `localForce` use different component-name conventions

For the same closed-form 3D beam:

| Frame | Per-node names |
|---|---|
| `globalForce` | `Px, Py, Pz, Mx, My, Mz` (global axes) |
| `localForce` | `N, Vy, Vz, T, My, Mz` (element-local axes) |

Same physical DOFs, different naming. The `My` / `Mz` symbols **collide**
between frames; only the keyword (`globalForce` vs `localForce`)
disambiguates them. `N`, `Vy`, `Vz`, `T` are local-frame-only; `Px`, `Py`,
`Pz`, `Mx` are global-frame-only.

---

## 10. Path ints in META/COMPONENTS are descriptor types, not array indices

`<pathInts>` like `0.1.2.` is **NOT** `(elem_idx, gauss_idx, section_idx)`
— it's `ElementOutputDescriptorType` codes:

```
0 = Element
1 = Gauss
2 = Section
3 = Fiber
4 = Material
```

So `0.1.2.` means "Element→Gauss→Section" hierarchy. The Gauss-point
INDEX is encoded by the **segment's position** in the `;`-separated list,
NOT by anything in `pathInts`. Don't read `0.1.2.` as "element 0, gauss 1,
section 2".

For deeper levels like fiber probing, expect paths like `0.1.2.3.` (adds
Fiber) or `0.1.2.4.` (adds Material).

---

## 11. `NUM_COLUMNS` should equal the META block sum

Invariant: `sum(MULTIPLICITY[i] * NUM_COMPONENTS[i]) == NUM_COLUMNS`.
For closed-form (single block): `1 * NUM_COMPONENTS[0] == NUM_COLUMNS`.
For gauss-level: e.g. `5 * 6 == 30` for 5-IP × 6-comp aggregated section.

This invariant catches MPCO version drift early. If your reader sees a
mismatch, the bucket layout assumption is wrong and you should stop
rather than producing silently-misaligned data.

---

## 12. `MULTIPLICITY > 1` indicates compressed META

Adjacent blocks with identical structure (e.g. all fibers of one section
sharing the same component layout) get collapsed: a `MULTIPLICITY[i] = N`
means N adjacent column-groups share `NUM_COMPONENTS[i]` and the same
component names (the COMPONENTS string repeats once for each, but
sometimes MPCO compresses the shared structure). Worth checking when
parsing fiber buckets in Phase 11c-equivalent work.

---

## 13. Catalog architecture: split "fixed by class" vs "varies per instance"

When wiring beam-columns we found a clean separation:

- **`ResponseLayout`** (fixed by class): `(class, rule, token) → (n_GP,
  natural_coords, component_layout)`. Works for continuum solids, shells,
  trusses — anything with class-determined response shape.
- **`CustomRuleLayout`** (per-instance): just `(class_tag, coord_system)`.
  The actual `(n_IP, GP_X, component_layout)` is resolved at read time
  from `(GP_X attribute, META/COMPONENTS section codes)`.

Two parallel catalogs, non-overlapping membership. Keeps the
"declarative" catalog small and pushes the variable shape into a
narrow runtime resolver. This pattern likely scales to fibers and
layered shells — anywhere the response shape depends on per-element
metadata (assigned section, layered material, etc.).

---

## 14. Topology-aware routing avoids canonical-name collisions

Same canonical user-facing name routes to different on-disk groups by
*topology context*:

| Canonical | Topology | MPCO keyword |
|---|---|---|
| `axial_force` | gauss-points | `axialForce` (Truss scalar) |
| `axial_force` | line-stations | `section.force` (beam vector, first comp) |
| `bending_moment_y` | shells (gauss) | `stresses` (resultant) |
| `bending_moment_y` | line-stations | `section.force` (Mz column) |

Lets the user keep ONE canonical vocabulary even when the same name maps
to different on-disk groups depending on which element type carries it.
The reader takes a `topology=` hint; the lookup table is per-topology.

For STKO_to_python, this is relevant if you expose user-friendly names
across multiple element families that might otherwise collide.

---

## 15. META is the source of truth; the catalog is a hint

Always validate `NUM_COLUMNS` against your assumed layout and fail loudly
if they disagree. Don't trust the catalog blindly — MPCO files can come
from section configurations the catalog wasn't designed for, and silently
mis-naming columns is worse than raising.

We added a per-bucket `validate_*_meta(bucket_grp, layout, ...)` helper
for each topology that cross-checks NUM_COLUMNS, MULTIPLICITY shape, and
GAUSS_IDS sequence/sentinel against the resolved layout before any data
is read.

---

## 16. Three-way agreement testing catches convention bugs

For each topology we built three independent paths to the same data:

1. **Read from `.mpco`** (file format)
2. **Capture live via openseespy** (in-process)
3. **Transcode from `.out` text recorder** (file format, different
   conventions than MPCO)

Then ran an identical OpenSees model and asserted the three paths
produce the **same numbers**. Most bugs we caught were path-disagreements
on column order or component naming — not numeric errors. The three-way
test is the strongest signal for "convention bugs are absent."

For STKO_to_python: even a two-way version (MPCO vs `.out`-via-Tcl-script)
would catch most format-side bugs.

---

## 17. Inspect a real fixture before writing reader code

The apeGmsh Phase 11b plan called the META/COMPONENTS format "pseudo"
because nobody had inspected an actual file. We saved real time by:

1. Writing a tiny Tcl script (`recorder mpco run.mpco -E section.force ...`)
2. Running it through OpenSees once
3. Dumping the HDF5 tree with `h5py` directly (skipping the
   `inspect_mpco.py` script which had a bug on this file)

Turned three guesses about the format into one verified ground truth.
Cost ~5 minutes; saved at least an hour of "why doesn't my parser work."

---

## Quick-reference summary

| Topology | MPCO group | GP_X location | META segments | Sentinel |
|---|---|---|---|---|
| Continuum gauss | `stresses` / `strains` | (catalog lookup; not per-bucket) | per-GP, sequential | `GAUSS_IDS=[0..n-1]` |
| Line-stations (beam-columns) | `section.force` | `MODEL/ELEMENTS/<bracket-2-field>/@GP_X` | per-IP `0.1.2.<comps>` | `GAUSS_IDS=[0..n_IP-1]` |
| Nodal forces (closed-form beams) | `globalForce` / `localForce` | n/a (no IPs) | single block `0.<node-suffixed names>` | `GAUSS_IDS=[[-1]]` |
| Fibers | `section.fiber.stress` | per-section | per-fiber within per-IP segments | (Phase 11c — not yet codified) |
| Layers (layered shells) | `material.fiber.stress` (with shell keyword swap) | per-surface-IP | per-layer × per-sub-IP | (Phase 11c — not yet codified) |
