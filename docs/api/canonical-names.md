# Canonical element result names

The on-disk MPCO shortnames for the same engineering quantity vary by
element family: `P` for axial force in a line-station beam bucket, `N`
in a closed-form `localForce` bucket, `sigma11` in a continuum stress
bucket. The canonical-name layer provides a stable vocabulary that works
across all families.

Source: [`STKO_to_python/elements/canonical.py`](https://github.com/nmorabowen/STKO_to_python/blob/main/src/STKO_to_python/elements/canonical.py)

---

## The canonical map

| Canonical name | MPCO shortnames | Description |
|---|---|---|
| `axial_force` | `P`, `N` | Beam axial / normal force |
| `bending_moment_z` | `Mz` | Bending moment about local z |
| `bending_moment_y` | `My` | Bending moment about local y |
| `torsion` | `T` | Torsional moment |
| `shear_y` | `Vy` | Shear force in local y |
| `shear_z` | `Vz` | Shear force in local z |
| `axial_strain` | `eps` | Beam section axial strain |
| `curvature_z` | `kappaZ` | Beam section curvature about z |
| `curvature_y` | `kappaY` | Beam section curvature about y |
| `twist` | `theta` | Beam section twist |
| `membrane_xx` | `Fxx` | Shell membrane force Fxx |
| `membrane_yy` | `Fyy` | Shell membrane force Fyy |
| `membrane_xy` | `Fxy` | Shell membrane force Fxy |
| `bending_moment_xx` | `Mxx` | Shell bending moment Mxx |
| `bending_moment_yy` | `Myy` | Shell bending moment Myy |
| `bending_moment_xy` | `Mxy` | Shell bending moment Mxy |
| `transverse_shear_xz` | `Vxz` | Shell transverse shear Vxz |
| `transverse_shear_yz` | `Vyz` | Shell transverse shear Vyz |
| `membrane_strain_xx` | `epsXX` | Shell membrane strain |
| `membrane_strain_yy` | `epsYY` | Shell membrane strain |
| `membrane_strain_xy` | `epsXY` | Shell membrane strain |
| `curvature_xx` | `kappaXX` | Shell curvature |
| `curvature_yy` | `kappaYY` | Shell curvature |
| `curvature_xy` | `kappaXY` | Shell curvature |
| `transverse_shear_strain_xz` | `gammaXZ` | Shell transverse shear strain |
| `transverse_shear_strain_yz` | `gammaYZ` | Shell transverse shear strain |
| `stress_11` | `sigma11` | Continuum normal stress σ₁₁ |
| `stress_22` | `sigma22` | Continuum normal stress σ₂₂ |
| `stress_33` | `sigma33` | Continuum normal stress σ₃₃ |
| `stress_12` | `sigma12` | Continuum shear stress σ₁₂ |
| `stress_23` | `sigma23` | Continuum shear stress σ₂₃ |
| `stress_13` | `sigma13` | Continuum shear stress σ₁₃ |
| `strain_11` | `eps11` | Continuum normal strain ε₁₁ |
| `strain_22` | `eps22` | Continuum normal strain ε₂₂ |
| `strain_33` | `eps33` | Continuum normal strain ε₃₃ |
| `strain_12` | `eps12` | Continuum shear strain ε₁₂ |
| `strain_23` | `eps23` | Continuum shear strain ε₂₃ |
| `strain_13` | `eps13` | Continuum shear strain ε₁₃ |
| `damage_pos` | `d+` | Tensile damage variable |
| `damage_neg` | `d-` | Compressive damage variable |
| `plastic_strain_pos` | `PLE+` | Tensile plastic strain |
| `plastic_strain_neg` | `PLE-` | Compressive plastic strain |
| `crack_width` | `cw` | Crack width |
| `force_x_global` | `Px` | Closed-form global X force |
| `force_y_global` | `Py` | Closed-form global Y force |
| `force_z_global` | `Pz` | Closed-form global Z force |
| `moment_x_global` | `Mx` | Closed-form global X moment |

!!! note "My / Mz collision"
    `My` and `Mz` appear in both `globalForce` (global axes) and
    `localForce` (element-local axes). These are **not** registered as
    canonical names to avoid ambiguity. Fetch the correct bucket
    (`globalForces` vs `localForce`) and access columns directly by
    shortname.

---

## Column-name suffix conventions

Column names produced by the META parser end in one of three suffix
patterns; `shortname_of()` strips them to recover the MPCO shortname:

| Suffix | Pattern | Example |
|---|---|---|
| Closed-form (per node) | `_<int>` | `Px_1`, `N_2` |
| Line-station / Gauss-level | `_ip<int>` | `P_ip0`, `sigma11_ip7` |
| Compressed fiber | `_f<int>_ip<int>` | `sigma11_f3_ip0` |

---

## Public functions

### `available_canonicals`

```python
from STKO_to_python.elements.canonical import available_canonicals

names = available_canonicals()
# ('axial_force', 'axial_strain', 'bending_moment_xx', ...)
```

Returns a sorted tuple of all registered canonical names.

---

### `shortname_of`

```python
from STKO_to_python.elements.canonical import shortname_of

shortname_of("Px_1")          # "Px"
shortname_of("P_ip3")         # "P"
shortname_of("sigma11_f5_ip0") # "sigma11"
shortname_of("Mz_2")          # "Mz"
```

Strips the suffix from a flat column name and returns the MPCO shortname.
Works on all three suffix patterns.

---

### `match_canonical_columns`

```python
from STKO_to_python.elements.canonical import match_canonical_columns

# Line-station beam: columns are P_ip0..P_ip4, Mz_ip0..Mz_ip4, ...
cols = ["P_ip0", "Mz_ip0", "My_ip0", "T_ip0", "P_ip1", "Mz_ip1"]
match_canonical_columns("axial_force", cols)
# ["P_ip0", "P_ip1"]

match_canonical_columns("bending_moment_z", cols)
# ["Mz_ip0", "Mz_ip1"]
```

Returns the subset of `columns` whose MPCO shortname matches the
registered shortnames for the given canonical name. Raises `ValueError`
if the canonical name is unknown.

---

### `list_canonical_for_columns`

```python
from STKO_to_python.elements.canonical import list_canonical_for_columns

cols = ["P_ip0", "Mz_ip0", "My_ip0", "T_ip0"]
list_canonical_for_columns(cols)
# ('axial_force', 'bending_moment_y', 'bending_moment_z', 'torsion')
```

Returns the canonical names whose shortnames are present in the given
column list. Used by `ElementResults.list_canonicals()`.

---

## Via `ElementResults`

These functions are also exposed as methods on `ElementResults`:

```python
er.list_canonicals()
# ('axial_force', 'bending_moment_z', ...)

er.canonical_columns("axial_force")
# ('P_ip0', 'P_ip1', 'P_ip2', 'P_ip3', 'P_ip4')

er.canonical("bending_moment_z")
# pd.DataFrame with columns Mz_ip0 ... Mz_ip4
```

See [ElementResults.md](../ElementResults.md#canonical-engineering-friendly-names)
for the full workflow.

---

## API reference

::: STKO_to_python.elements.canonical
