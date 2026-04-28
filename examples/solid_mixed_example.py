"""Solid + fiber beam example — mixed-element multi-partition fixture.

Demonstrates the full integration-point API against the checked-in
``stko_results_examples/solid_partition_example`` fixture:

    Two-partition dataset containing:
    * ``56-Brick`` — 8-node hexahedral continuum, 8 Gauss points (3-D).
    * ``64-DispBeamColumn3d`` — displacement-based beam, 2 integration
      stations (Lobatto end-points), with compressed fiber sections.

Sections:

    1.  Dataset construction and element discovery.
    2.  Brick ``material.stress`` — 8 Gauss points, gp_natural (8, 3).
    3.  ``at_ip()`` on Brick — per-IP stress slice.
    4.  ``physical_coords()`` on Brick — IP positions inside element bbox.
    5.  ``jacobian_dets()`` and element volume via quadrature.
    6.  ``integrate_canonical("stress_11")`` — volume-integrated sigma_11.
    7.  Beam ``section.force`` — 2-IP line-station, gp_xi at +-1.
    8.  Beam fiber section — ``section.fiber.stress``, 6 fibers x 2 IPs.
    9.  ``at_ip()`` on fiber result — all 6 fibers at one station.
   10.  Why ``integrate_canonical`` is not available for fiber buckets.
   11.  ``plot.diagram()`` — bending moment diagram along a beam element.
   12.  ``plot.scatter()`` — Gauss-point stress scatter for a brick element.
   13.  ElementResults pickle round-trip.

Run with::

    python examples/solid_mixed_example.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from STKO_to_python import MPCODataSet
from STKO_to_python.elements.element_results import ElementResults

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "stko_results_examples" / "solid_partition_example"
RECORDER = "Recorder"
STAGE = "MODEL_STAGE[1]"
N_BRICK = 3     # number of Brick elements to fetch
N_BEAM = 2      # number of DispBeamColumn3d elements to fetch


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Dataset construction and element discovery
# ---------------------------------------------------------------------------
def section_1_introspect(ds: MPCODataSet) -> tuple[list[int], list[int]]:
    section("1. Dataset construction and element discovery")

    print(f"model_stages:          {ds.model_stages}")
    print(f"element_results_names: {ds.element_results_names}")
    print(f"unique_element_types:  {ds.unique_element_types}")

    df = ds.elements_info["dataframe"]
    brick_count = (df["element_type"] == "56-Brick").sum()
    beam_count = (df["element_type"] == "64-DispBeamColumn3d").sum()
    print(f"56-Brick elements:              {brick_count}")
    print(f"64-DispBeamColumn3d elements:   {beam_count}")

    brick_ids = [int(i) for i in df.query("element_type == '56-Brick'")
                 ["element_id"].head(N_BRICK).tolist()]
    beam_ids = [int(i) for i in df.query("element_type == '64-DispBeamColumn3d'")
                ["element_id"].head(N_BEAM).tolist()]
    print(f"Brick IDs for demos:  {brick_ids}")
    print(f"Beam IDs for demos:   {beam_ids}")

    avail = ds.elements.get_available_element_results()
    print("\nAvailable element results by partition:")
    for part_id, mapping in avail.items():
        for rname, types in mapping.items():
            print(f"  partition {part_id}: {rname!r} -> {types}")

    return brick_ids, beam_ids


# ---------------------------------------------------------------------------
# 2. Brick material.stress — 8 Gauss points
# ---------------------------------------------------------------------------
def section_2_brick_stress(ds: MPCODataSet, brick_ids: list[int]) -> ElementResults:
    section("2. Brick material.stress - 8 Gauss points, gp_natural (8, 3)")

    er = ds.elements.get_element_results(
        results_name="material.stress",
        element_type="56-Brick",
        model_stage=STAGE,
        element_ids=brick_ids,
    )
    print(repr(er))
    print(f"df.shape:           {er.df.shape}")
    print(f"n_ip:               {er.n_ip}")
    print(f"gp_dim:             {er.gp_dim}")
    print(f"gp_xi:              {er.gp_xi}  (None - multi-D, no 1-D xi)")
    print(f"gp_natural.shape:   {er.gp_natural.shape}")
    print(f"gp_natural (8 Gauss-Legendre pts in xi-eta-zeta):")
    print(er.gp_natural)
    print(f"\ngp_weights:         {er.gp_weights}")
    print(f"gp_weights.sum():   {er.gp_weights.sum():.3f}  (= 8 for [-1,1]^3)")

    # Standard 2x2x2 Gauss-Legendre rule: all coords are +-1/sqrt(3)
    expected_coords = 1.0 / np.sqrt(3.0)
    print(f"\nAll |gp_natural| approx 1/sqrt(3) ({expected_coords:.6f}): "
          f"{np.allclose(np.abs(er.gp_natural), expected_coords)}")
    print(f"list_canonicals():  {er.list_canonicals()}")
    return er


# ---------------------------------------------------------------------------
# 3. at_ip() — per-Gauss-point stress slice
# ---------------------------------------------------------------------------
def section_3_at_ip_brick(er: ElementResults) -> None:
    section("3. at_ip() on Brick - per-IP stress slice")

    # Show column layout for IP 0 and IP 7
    sub0 = er.at_ip(0)
    sub7 = er.at_ip(7)
    print(f"at_ip(0) columns: {list(sub0.columns)}")
    print(f"at_ip(7) columns: {list(sub7.columns)}")
    print(f"at_ip(0) shape:   {sub0.shape}")

    # Examine stress at Gauss point 0 for the first element
    eid = er.element_ids[0]
    step = 1
    print(f"\nsigma_11 at each Gauss point for element {eid}, step {step}:")
    for ip in range(er.n_ip):
        sub = er.at_ip(ip)
        col = [c for c in sub.columns if "sigma11" in c]
        if col:
            val = float(sub.loc[(eid, step), col[0]])
            print(f"  IP {ip}: sigma_11 = {val:.4e}")


# ---------------------------------------------------------------------------
# 4. physical_coords() — IP positions in physical space
# ---------------------------------------------------------------------------
def section_4_physical_coords(er: ElementResults) -> None:
    section("4. physical_coords() - Gauss-point physical positions")

    phys = er.physical_coords()    # (n_e, 8, 3)
    print(f"physical_coords() shape: {phys.shape}")

    # Verify all IPs lie inside the element bounding box
    all_inside = True
    for i, eid in enumerate(er.element_ids):
        nc = er.element_node_coords[i]   # (8 nodes, 3) for Brick
        lo = nc.min(axis=0) - 1e-9
        hi = nc.max(axis=0) + 1e-9
        for ip_xyz in phys[i]:
            if not (np.all(ip_xyz >= lo) and np.all(ip_xyz <= hi)):
                all_inside = False
    print(f"All IPs inside element bounding box: {all_inside}")

    eid = er.element_ids[0]
    print(f"\nElement {eid} node bounding box:")
    nc = er.element_node_coords[0]
    print(f"  x: [{nc[:, 0].min():.4f}, {nc[:, 0].max():.4f}]")
    print(f"  y: [{nc[:, 1].min():.4f}, {nc[:, 1].max():.4f}]")
    print(f"  z: [{nc[:, 2].min():.4f}, {nc[:, 2].max():.4f}]")
    print(f"Element {eid} Gauss-point physical coords:\n{phys[0]}")


# ---------------------------------------------------------------------------
# 5. jacobian_dets() and element volume
# ---------------------------------------------------------------------------
def section_5_volumes(er: ElementResults) -> None:
    section("5. jacobian_dets() - element volume via numerical quadrature")

    dets = er.jacobian_dets()   # (n_e, 8)
    print(f"jacobian_dets() shape: {dets.shape}")
    print(f"All determinants positive: {np.all(dets > 0)}")

    # V = sum(w_i * |J_i|) - integrates the constant function 1 over the element
    vols = (er.gp_weights[np.newaxis, :] * dets).sum(axis=1)  # (n_e,)
    print(f"\nElement volumes (from quadrature):")
    for eid, vol in zip(er.element_ids, vols):
        # Cross-check with bounding box (exact for axis-aligned bricks)
        i = er.element_ids.index(eid)
        nc = er.element_node_coords[i]
        bbox_vol = float(
            (nc[:, 0].max() - nc[:, 0].min())
            * (nc[:, 1].max() - nc[:, 1].min())
            * (nc[:, 2].max() - nc[:, 2].min())
        )
        print(f"  element {eid:4d}: V_quadrature={vol:.6f}, V_bbox={bbox_vol:.6f}")


# ---------------------------------------------------------------------------
# 6. integrate_canonical("stress_11") — volume-integrated σ₁₁
# ---------------------------------------------------------------------------
def section_6_integrate_canonical(er: ElementResults) -> None:
    section("6. integrate_canonical('stress_11') — sum sigma_11 * w * |J| over IPs")

    s = er.integrate_canonical("stress_11")
    print(f"Series shape: {s.shape}, name: {s.name!r}")
    print(f"Index names:  {list(s.index.names)}")

    # Reshape to (n_steps, n_elements) for easy inspection
    mtx = s.unstack("element_id")
    print("\nStep x element matrix:")
    print(mtx.to_string())

    # Manual verification for element[0], step 1
    cols = list(er.canonical_columns("stress_11"))
    dets = er.jacobian_dets()
    eid = er.element_ids[0]
    step = 1
    eid_idx = er.element_ids.index(eid)
    sigma = er.df.xs((eid, step))[cols].to_numpy(dtype=np.float64)
    manual = float((sigma * er.gp_weights * dets[eid_idx]).sum())
    helper = float(s.loc[eid, step])
    print(f"\nManual check (element {eid}, step {step}):")
    print(f"  sum sigma_11 * w * |J| = {manual:.6e}")
    print(f"  integrate_canonical = {helper:.6e}")
    print(f"  Match: {abs(manual - helper) < 1e-12 * max(abs(manual), 1e-12)}")


# ---------------------------------------------------------------------------
# 7. Beam section.force — 2-IP line-station
# ---------------------------------------------------------------------------
def section_7_beam_section_force(
    ds: MPCODataSet, beam_ids: list[int]
) -> ElementResults:
    section("7. Beam section.force - 2-IP line-station (gp_xi at +-1)")

    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="64-DispBeamColumn3d",
        model_stage=STAGE,
        element_ids=beam_ids,
    )
    print(repr(er))
    print(f"n_ip:               {er.n_ip}")
    print(f"gp_xi:              {er.gp_xi}")
    print(f"  (Lobatto end-points: xi=-1 is node i, xi=+1 is node j)")
    print(f"gp_natural:         {er.gp_natural}  (None for line elements)")
    print(f"gp_weights:         {er.gp_weights}  (None for custom-rule beams)")
    print(f"list_components():  {er.list_components()}")
    print(f"list_canonicals():  {er.list_canonicals()}")

    # at_ip() slices by integration station
    sub_i = er.at_ip(0)   # section forces at node i end
    sub_j = er.at_ip(1)   # section forces at node j end
    print(f"\nat_ip(0) columns: {list(sub_i.columns)}")
    print(f"at_ip(1) columns: {list(sub_j.columns)}")

    eid = beam_ids[0]
    step = 1
    print(f"\nAxial force P at element {eid}, step {step}:")
    print(f"  node-i end (IP 0): {float(sub_i.loc[(eid, step), 'P_ip0']):.4e}")
    print(f"  node-j end (IP 1): {float(sub_j.loc[(eid, step), 'P_ip1']):.4e}")
    return er


# ---------------------------------------------------------------------------
# 8. Beam section.fiber.stress — 6 fibers × 2 integration stations
# ---------------------------------------------------------------------------
def section_8_fiber_stress(ds: MPCODataSet, beam_ids: list[int]) -> ElementResults:
    section("8. section.fiber.stress - compressed fiber bucket (6 fibers x 2 IPs)")

    er = ds.elements.get_element_results(
        results_name="section.fiber.stress",
        element_type="64-DispBeamColumn3d",
        model_stage=STAGE,
        element_ids=beam_ids,
    )
    print(repr(er))
    print(f"df.shape:          {er.df.shape}")
    print(f"n_ip:              {er.n_ip}")
    print(f"gp_xi:             {er.gp_xi}")
    print(f"list_components(): {er.list_components()}")
    print(f"  Column naming:   sigma11_f<fiber>_ip<station>")
    print(f"  6 fibers x 2 stations = {er.n_components} columns total")
    return er


# ---------------------------------------------------------------------------
# 9. at_ip() on fiber result — all fibers at one station
# ---------------------------------------------------------------------------
def section_9_at_ip_fiber(er: ElementResults) -> None:
    section("9. at_ip() on fiber result — all fibers at station 0 (xi=-1)")

    sub0 = er.at_ip(0)   # all 6 fibers at node-i end
    sub1 = er.at_ip(1)   # all 6 fibers at node-j end
    print(f"at_ip(0) shape:   {sub0.shape}  (n_el x n_steps, 6 fibers)")
    print(f"at_ip(0) columns: {list(sub0.columns)}")
    print(f"at_ip(1) columns: {list(sub1.columns)}")

    # Show fiber stresses at station 0 for the first element, step 1
    eid = er.element_ids[0]
    step = 1
    fiber_vals = sub0.loc[(eid, step)]
    print(f"\nFiber stresses at xi=-1 (IP 0), element {eid}, step {step}:")
    for col, val in fiber_vals.items():
        print(f"  {col}: {float(val):.4e}")


# ---------------------------------------------------------------------------
# 10. Why integrate_canonical is not available for fiber buckets
# ---------------------------------------------------------------------------
def section_10_fiber_integrate_raises(er: ElementResults) -> None:
    section("10. integrate_canonical - not available for fiber buckets")

    print("Fiber buckets have gp_weights=None (custom rule, no standard weights).")
    print("integrate_canonical raises ValueError to explain this clearly.\n")
    try:
        er.integrate_canonical("stress_11")
    except ValueError as exc:
        print(f"ValueError: {exc}")
    print(
        "\nFor manual fiber integration, iterate over at_ip(k) and apply"
        " your own fiber-area weights:\n"
        "    sub_ip0 = er.at_ip(0)\n"
        "    # fiber_areas is a (6,) array you provide\n"
        "    result = (sub_ip0.to_numpy() * fiber_areas[None, :]).sum(axis=1)"
    )


# ---------------------------------------------------------------------------
# 11. plot.diagram() — bending moment diagram along a beam element
# ---------------------------------------------------------------------------
def section_11_beam_diagram(er_sf: ElementResults) -> None:
    section("11. plot.diagram() - bending moment diagram along a beam")

    # er_sf is the section.force result (n_ip=2, gp_xi=[-1, +1]).
    # diagram() requires gp_dim==1 and a canonical that maps to exactly n_ip cols.
    eid = er_sf.element_ids[0]
    step = 1

    # Physical x-axis (requires element_node_coords)
    ax, meta = er_sf.plot.diagram(
        "bending_moment_z",
        element_id=eid,
        step=step,
    )
    print(f"diagram('bending_moment_z', element={eid}, step={step})")
    print(f"  x (physical position): {meta['x']}")
    print(f"  y (Mz values):         {meta['y']}")
    print(f"  columns used:          {meta['columns']}")
    plt.close("all")

    # Natural xi axis (no node coords needed)
    ax, meta = er_sf.plot.diagram(
        "axial_force",
        element_id=eid,
        step=step,
        x_in_natural=True,
    )
    print(f"\ndiagram('axial_force', x_in_natural=True)")
    print(f"  xi values: {meta['x']}")
    print(f"  P values:  {meta['y']}")
    plt.close("all")


# ---------------------------------------------------------------------------
# 12. plot.scatter() — Gauss-point stress scatter for a brick element
# ---------------------------------------------------------------------------
def section_12_brick_scatter(er_brick: ElementResults) -> None:
    section("12. plot.scatter() - stress_11 at Gauss points (x-z view)")

    step = 1
    ax, meta = er_brick.plot.scatter(
        "stress_11",
        step=step,
        axes=("x", "z"),
    )
    print(f"scatter('stress_11', step={step}, axes=('x','z'))")
    print(f"  x shape:      {meta['x'].shape}  ({er_brick.n_elements} elements x {er_brick.n_ip} IPs)")
    print(f"  values shape: {meta['values'].shape}")
    print(f"  values range: [{meta['values'].min():.3e}, {meta['values'].max():.3e}]")
    # Add a colorbar to the figure for a complete publication-ready result:
    #   ax.figure.colorbar(meta["scatter"], ax=ax, label="sigma_11")
    plt.close("all")


# ---------------------------------------------------------------------------
# 13. Pickle round-trip
# ---------------------------------------------------------------------------
def section_13_pickle(er_brick: ElementResults) -> None:
    section("13. ElementResults pickle round-trip")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "brick.pkl"
        er_brick.save_pickle(path)
        loaded = ElementResults.load_pickle(path)
        print(f"saved: {path.stat().st_size} bytes")
        print(f"loaded: {loaded!r}")
        print(f"df match:             {loaded.df.shape == er_brick.df.shape}")
        print(f"gp_natural preserved: "
              f"{np.allclose(loaded.gp_natural, er_brick.gp_natural)}")
        print(f"gp_weights preserved: "
              f"{np.allclose(loaded.gp_weights, er_brick.gp_weights)}")
        print(f"element_node_coords preserved: "
              f"{np.allclose(loaded.element_node_coords, er_brick.element_node_coords)}")
        # integrate_canonical still works after reload
        s = loaded.integrate_canonical("stress_11")
        print(f"integrate_canonical after pickle: shape={s.shape}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    if not (DATASET_DIR / "Recorder.part-0.mpco").exists():
        raise FileNotFoundError(
            f"solid_partition fixture not found at {DATASET_DIR}.\n"
            "Check that stko_results_examples/ is present in the repo root."
        )

    ds = MPCODataSet(str(DATASET_DIR), RECORDER, verbose=False)

    brick_ids, beam_ids = section_1_introspect(ds)
    er_brick = section_2_brick_stress(ds, brick_ids)
    section_3_at_ip_brick(er_brick)
    section_4_physical_coords(er_brick)
    section_5_volumes(er_brick)
    section_6_integrate_canonical(er_brick)
    er_beam = section_7_beam_section_force(ds, beam_ids)
    er_fiber = section_8_fiber_stress(ds, beam_ids)
    section_9_at_ip_fiber(er_fiber)
    section_10_fiber_integrate_raises(er_fiber)
    section_11_beam_diagram(er_beam)
    section_12_brick_scatter(er_brick)
    section_13_pickle(er_brick)

    print()
    print("solid_mixed_example complete.")


if __name__ == "__main__":
    main()
