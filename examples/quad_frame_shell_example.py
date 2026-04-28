"""Quad-frame shell example — multi-partition ASDShellQ4 fixture.

Demonstrates multi-partition datasets and Gauss-point integration
against the checked-in ``stko_results_examples/elasticFrame/QuadFrame_results``
fixture (two .part-N.mpco files, ``203-ASDShellQ4`` shell elements).

    1.  Multi-partition dataset construction and introspection.
    2.  Discovering shell element IDs from the element index.
    3.  Shell ``section.force`` — 4 Gauss points, gp_natural (4, 2).
    4.  Canonical names for shell section forces (membrane, bending,
        transverse shear).
    5.  ``at_ip()`` — per-Gauss-point DataFrame slice.
    6.  ``physical_coords()`` — IP physical positions in 3-D space.
    7.  ``jacobian_dets()`` — surface Jacobian for each IP.
    8.  Element area computation via numerical quadrature.
    9.  ``integrate_canonical("membrane_xx")`` — surface-integrated
        axial membrane force per element per step.
   10.  ElementResults pickle round-trip (gp_natural survives).

Run with::

    python examples/quad_frame_shell_example.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from STKO_to_python import MPCODataSet
from STKO_to_python.elements.element_results import ElementResults

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = (
    REPO_ROOT / "stko_results_examples" / "elasticFrame" / "QuadFrame_results"
)
RECORDER = "results"
STAGE = "MODEL_STAGE[1]"
N_SHELL_ELEMENTS = 5  # number of shells to fetch for the demo


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Dataset construction and introspection
# ---------------------------------------------------------------------------
def section_1_introspect(ds: MPCODataSet) -> None:
    section("1. Multi-partition dataset introspection")

    print(f"model_stages:          {ds.model_stages}")
    print(f"node_results_names:    {ds.node_results_names}")
    print(f"element_results_names: {ds.element_results_names}")
    print(f"unique_element_types:  {ds.unique_element_types}")
    print(f"total elements:        {len(ds.elements_info['dataframe'])}")
    print(f"total nodes:           {len(ds.nodes_info['dataframe'])}")

    # Because there are two partition files (part-0 and part-1), the library
    # merges them transparently. The user never needs to know about partitions.
    avail = ds.elements.get_available_element_results()
    print("\nAvailable element results by partition:")
    for part_id, mapping in avail.items():
        for rname, types in mapping.items():
            print(f"  partition {part_id}: {rname!r} -> {types}")


# ---------------------------------------------------------------------------
# 2. Discover shell element IDs
# ---------------------------------------------------------------------------
def _get_shell_ids(ds: MPCODataSet, n: int) -> list[int]:
    df = ds.elements_info["dataframe"]
    ids = (
        df.query("element_type == '203-ASDShellQ4'")["element_id"]
        .head(n)
        .tolist()
    )
    return [int(i) for i in ids]


def section_2_discover_elements(ds: MPCODataSet) -> list[int]:
    section("2. Discovering shell element IDs from the element index")

    df = ds.elements_info["dataframe"]
    shell_count = (df["element_type"] == "203-ASDShellQ4").sum()
    print(f"Total 203-ASDShellQ4 elements: {shell_count}")

    shell_ids = _get_shell_ids(ds, N_SHELL_ELEMENTS)
    print(f"Using first {N_SHELL_ELEMENTS} IDs for demos: {shell_ids}")
    return shell_ids


# ---------------------------------------------------------------------------
# 3. Fetch shell section.force with 4 Gauss points
# ---------------------------------------------------------------------------
def section_3_fetch_section_force(
    ds: MPCODataSet, shell_ids: list[int]
) -> ElementResults:
    section("3. Shell section.force — 4-IP Gauss-level bucket")

    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="203-ASDShellQ4",
        model_stage=STAGE,
        element_ids=shell_ids,
    )
    print(repr(er))
    print(f"df.shape:              {er.df.shape}")
    print(f"n_elements={er.n_elements}, n_steps={er.n_steps}, "
          f"n_components={er.n_components}")
    print(f"n_ip:                  {er.n_ip}")
    print(f"gp_xi:                 {er.gp_xi}  (None for 2-D shells)")
    print(f"gp_natural.shape:      {er.gp_natural.shape}")
    print(f"gp_natural (2x2 Gauss-Legendre in xi-eta):\n{er.gp_natural}")
    print(f"gp_weights:            {er.gp_weights}  (sum={er.gp_weights.sum():.3f})")
    print(f"list_components():     {er.list_components()}")
    return er


# ---------------------------------------------------------------------------
# 4. Canonical names for shell section forces
# ---------------------------------------------------------------------------
def section_4_canonical_names(er: ElementResults) -> None:
    section("4. Canonical engineering names (shell section forces)")

    print(f"list_canonicals(): {er.list_canonicals()}")
    for canon in er.list_canonicals():
        cols = er.canonical_columns(canon)
        print(f"  {canon!r:30s} -> {cols}")

    # Pull a full DataFrame for membrane_xx across all elements and steps
    df_mxx = er.canonical("membrane_xx")
    print(f"\ncanonical('membrane_xx') shape: {df_mxx.shape}")
    print(f"columns: {list(df_mxx.columns)}")
    print(df_mxx.head(3))


# ---------------------------------------------------------------------------
# 5. Per-IP slicing via at_ip()
# ---------------------------------------------------------------------------
def section_5_at_ip(er: ElementResults) -> None:
    section("5. Per-Gauss-point slicing — at_ip()")

    for ip in range(er.n_ip):
        sub = er.at_ip(ip)
        print(f"at_ip({ip}): shape={sub.shape}, columns={list(sub.columns)}")

    # Detailed look at IP 0 for the first element
    sub0 = er.at_ip(0)
    eid = er.element_ids[0]
    print(f"\nIP 0 data for element {eid} (first 3 steps):")
    print(sub0.xs(eid, level="element_id").head(3))


# ---------------------------------------------------------------------------
# 6. Physical coordinates of Gauss points
# ---------------------------------------------------------------------------
def section_6_physical_coords(er: ElementResults) -> None:
    section("6. physical_coords() — IP physical positions in 3-D space")

    phys = er.physical_coords()   # (n_e, 4, 3)
    print(f"physical_coords() shape: {phys.shape}")

    # Verify each IP lies inside its element's node bounding box
    all_inside = True
    for i, eid in enumerate(er.element_ids):
        nc = er.element_node_coords[i]   # (4 nodes, 3)
        lo = nc.min(axis=0) - 1e-9
        hi = nc.max(axis=0) + 1e-9
        for ip_xyz in phys[i]:
            if not (np.all(ip_xyz >= lo) and np.all(ip_xyz <= hi)):
                all_inside = False
    print(f"All IPs inside their element bounding box: {all_inside}")

    # Show physical coords for the first element
    eid = er.element_ids[0]
    print(f"\nElement {eid} node coords:\n{er.element_node_coords[0]}")
    print(f"Element {eid} Gauss-point physical coords:\n{phys[0]}")


# ---------------------------------------------------------------------------
# 7. Jacobian determinants (surface Jacobian)
# ---------------------------------------------------------------------------
def section_7_jacobian_dets(er: ElementResults) -> None:
    section("7. jacobian_dets() — surface Jacobian determinant per IP")

    dets = er.jacobian_dets()   # (n_e, 4)
    print(f"jacobian_dets() shape: {dets.shape}")
    print(f"All dets positive: {np.all(dets > 0)}")

    # Compute element areas via quadrature: A = Σ w_i * |J_i|
    areas = (er.gp_weights[np.newaxis, :] * dets).sum(axis=1)  # (n_e,)
    print(f"\nElement areas (from quadrature):")
    for eid, area in zip(er.element_ids, areas):
        print(f"  element {eid:4d}: area = {area:.6f}")


# ---------------------------------------------------------------------------
# 8. Surface-integrated membrane force
# ---------------------------------------------------------------------------
def section_8_integrate_canonical(er: ElementResults) -> None:
    section("8. integrate_canonical('membrane_xx') — surface resultant")

    s = er.integrate_canonical("membrane_xx")
    print(f"integrate_canonical series: shape={s.shape}, name={s.name!r}")
    print(f"Index names: {list(s.index.names)}")

    # Reshape to (n_steps, n_elements) matrix
    mtx = s.unstack("element_id")
    print(f"\nStep × element matrix shape: {mtx.shape}")
    print(f"(rows = time steps, columns = element IDs)")
    print(mtx.head(4).to_string())

    # Manual verification for one element at step 1
    cols = list(er.canonical_columns("membrane_xx"))
    dets = er.jacobian_dets()
    eid = er.element_ids[0]
    step = 1
    eid_idx = er.element_ids.index(eid)
    sigma = er.df.xs((eid, step))[cols].to_numpy(dtype=np.float64)
    manual = float((sigma * er.gp_weights * dets[eid_idx]).sum())
    helper = float(s.loc[eid, step])
    match = abs(manual - helper) < 1e-12 * max(abs(manual), 1e-12)
    print(f"\nManual verification (element {eid}, step {step}):")
    print(f"  manual = {manual:.6e}, integrate_canonical = {helper:.6e}")
    print(f"  Match: {match}")


# ---------------------------------------------------------------------------
# 9. Pickle round-trip — gp_natural survives
# ---------------------------------------------------------------------------
def section_9_pickle(er: ElementResults) -> None:
    section("9. ElementResults pickle round-trip (gp_natural survives)")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shell_er.pkl"
        er.save_pickle(path)
        loaded = ElementResults.load_pickle(path)
        print(f"saved: {path.stat().st_size} bytes")
        print(f"loaded: {loaded!r}")
        print(f"df match:             {loaded.df.shape == er.df.shape}")
        print(f"gp_natural preserved: "
              f"{np.allclose(loaded.gp_natural, er.gp_natural)}")
        print(f"gp_weights preserved: "
              f"{np.allclose(loaded.gp_weights, er.gp_weights)}")
        # integrate_canonical still works after reload
        s_after = loaded.integrate_canonical("membrane_xx")
        print(f"integrate_canonical after pickle: shape={s_after.shape}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    if not (DATASET_DIR / "results.part-0.mpco").exists():
        raise FileNotFoundError(
            f"QuadFrame fixture not found at {DATASET_DIR}.\n"
            "Check that stko_results_examples/ is present in the repo root."
        )

    ds = MPCODataSet(str(DATASET_DIR), RECORDER, verbose=False)

    section_1_introspect(ds)
    shell_ids = section_2_discover_elements(ds)
    er = section_3_fetch_section_force(ds, shell_ids)
    section_4_canonical_names(er)
    section_5_at_ip(er)
    section_6_physical_coords(er)
    section_7_jacobian_dets(er)
    section_8_integrate_canonical(er)
    section_9_pickle(er)

    print()
    print("quad_frame_shell_example complete.")


if __name__ == "__main__":
    main()
