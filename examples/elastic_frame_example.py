"""Elastic frame example — single-partition ElasticBeam3d fixture.

Demonstrates the core STKO_to_python workflow against the checked-in
``stko_results_examples/elasticFrame/results`` fixture:

    1.  Dataset construction and introspection.
    2.  Fetching nodal DISPLACEMENT results.
    3.  Engineering aggregations: drift, interstory envelope, residual
        drift, orbit.
    4.  XY plotting (per-result and dataset-level facade).
    5.  NodalResults pickle round-trip.
    6.  Fetching beam ``force`` and ``localForce`` (closed-form) results.
    7.  ElementResults broker API: envelope, at_step, to_dataframe.
    8.  Canonical engineering names on closed-form beam results.
    9.  ElementResults pickle round-trip.

Run with::

    python examples/elastic_frame_example.py
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
from STKO_to_python.results.nodal_results_dataclass import NodalResults

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "stko_results_examples" / "elasticFrame" / "results"
RECORDER = "results"

NODE_IDS = [1, 2, 3, 4]
ELEMENT_IDS = [1, 2, 3]
STAGE = "MODEL_STAGE[1]"


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Dataset construction and introspection
# ---------------------------------------------------------------------------
def section_1_introspect(ds: MPCODataSet) -> None:
    section("1. Dataset introspection")

    print(f"model_stages:           {ds.model_stages}")
    print(f"number_of_steps:        {ds.number_of_steps}")
    print(f"node_results_names:     {ds.node_results_names}")
    print(f"element_results_names:  {ds.element_results_names}")
    print(f"unique_element_types:   {ds.unique_element_types}")

    avail = ds.elements.get_available_element_results()
    for part_id, mapping in avail.items():
        for rname, types in mapping.items():
            print(f"  partition {part_id}: {rname!r} -> {types}")


# ---------------------------------------------------------------------------
# 2. Fetch nodal DISPLACEMENT
# ---------------------------------------------------------------------------
def section_2_fetch_nodal(ds: MPCODataSet) -> NodalResults:
    section("2. Fetch NodalResults — DISPLACEMENT")

    nr = ds.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage=STAGE,
        node_ids=NODE_IDS,
    )
    print(repr(nr))
    print(f"df.shape:         {nr.df.shape}")
    print(f"df.index.names:   {list(nr.df.index.names)}")
    print(f"time (first 3):   {nr.time[:3]}")
    print(f"list_results():   {nr.list_results()}")
    print(f"list_components('DISPLACEMENT'): {nr.list_components('DISPLACEMENT')}")
    return nr


# ---------------------------------------------------------------------------
# 3. Engineering aggregations
# ---------------------------------------------------------------------------
def section_3_aggregations(nr: NodalResults) -> None:
    section("3. Engineering aggregations")

    # Relative drift between two nodes
    ts = nr.drift(top=4, bottom=1, component=1)
    print(f"drift(top=4, bottom=1, comp=1):   length={ts.size}, "
          f"max|.|={abs(ts).max():.3e}")

    # Interstory envelope — uses Z-coordinate sorting internally
    env = nr.interstory_drift_envelope(
        component=1, node_ids=NODE_IDS, dz_tol=1e-3,
    )
    print(f"interstory_drift_envelope:  {len(env)} story pair(s)")
    print(f"  columns: {list(env.columns)}")
    print(env.to_string())

    # Residual drift at the end of the record
    resid = nr.residual_drift(top=4, bottom=1, component=1, tail=3, agg="mean")
    print(f"residual_drift (mean of last 3 steps): {resid:.3e}")

    # Orbit — displacement trajectory for a single node
    sx, sy = nr.orbit(node_ids=1, x_component=1, y_component=2)
    print(f"orbit(node=1, x=1, y=2):    x-length={len(sx)}, y-length={len(sy)}")


# ---------------------------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------------------------
def section_4_plotting(ds: MPCODataSet, nr: NodalResults) -> None:
    section("4. Plotting")

    # Per-result XY plot
    ax, meta = nr.plot.xy(
        y_results_name="DISPLACEMENT",
        y_direction=1,
        y_operation="Max",
        x_results_name="TIME",
    )
    print(f"nr.plot.xy -> ax={ax}, meta keys={list(meta.keys())}")
    plt.close("all")

    # Dataset-level one-shot plot
    ax, meta = ds.plot.xy(
        model_stage=STAGE,
        results_name="DISPLACEMENT",
        node_ids=NODE_IDS,
        y_direction=1,
        y_operation="Max",
        x_results_name="TIME",
    )
    print(f"ds.plot.xy -> ax={ax}, meta keys={list(meta.keys())}")
    plt.close("all")


# ---------------------------------------------------------------------------
# 5. NodalResults pickle round-trip
# ---------------------------------------------------------------------------
def section_5_pickle_nodal(nr: NodalResults) -> None:
    section("5. NodalResults pickle round-trip")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nr.pkl"
        nr.save_pickle(path)
        loaded = NodalResults.load_pickle(path)
        print(f"saved:  {path.stat().st_size} bytes")
        print(f"loaded: {loaded!r}")
        print(f"shapes match: {loaded.df.shape == nr.df.shape}")


# ---------------------------------------------------------------------------
# 6. Fetch beam `force` results (closed-form)
# ---------------------------------------------------------------------------
def section_6_fetch_element(ds: MPCODataSet) -> ElementResults:
    section("6. Fetch ElementResults — force (closed-form ElasticBeam3d)")

    er = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        model_stage=STAGE,
        element_ids=ELEMENT_IDS,
    )
    print(repr(er))
    print(f"df.shape:          {er.df.shape}")
    print(f"df.index.names:    {list(er.df.index.names)}")
    print(f"list_components(): {er.list_components()}")
    print(f"n_elements:        {er.n_elements}")
    print(f"n_steps:           {er.n_steps}")
    print(f"n_components:      {er.n_components}")
    print(f"gp_xi:             {er.gp_xi}  (None = closed-form, no IPs)")
    return er


# ---------------------------------------------------------------------------
# 7. ElementResults broker API
# ---------------------------------------------------------------------------
def section_7_element_broker(er: ElementResults) -> None:
    section("7. ElementResults broker API")

    # fetch() — explicit component + element filter
    sub = er.fetch(component="Pz_1", element_ids=[1, 2])
    print(f"fetch('Pz_1', [1,2]):   Series length {len(sub)}")

    # Dynamic attribute access
    view = er.Pz_1[[1, 2]]
    print(f"er.Pz_1[[1,2]]:         type={type(view).__name__}")

    # Envelope over all elements and all steps
    env = er.envelope(component="Pz_1")
    print(f"envelope('Pz_1'):       columns={list(env.columns)}")
    print(env)

    # Step snapshot
    snap = er.at_step(5)
    print(f"\nat_step(5):  shape={snap.shape}")

    # Time snapshot (nearest recorded step to t=0.5)
    snap_t = er.at_time(0.5)
    print(f"at_time(0.5): shape={snap_t.shape}")

    # Flat DataFrame with time column included
    flat = er.to_dataframe(include_time=True)
    print(f"to_dataframe(): first 5 cols={list(flat.columns)[:5]}, "
          f"total={len(flat.columns)}")


# ---------------------------------------------------------------------------
# 8. Canonical engineering names
# ---------------------------------------------------------------------------
def section_8_canonical_names(ds: MPCODataSet) -> None:
    section("8. Canonical engineering names")

    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        model_stage=STAGE,
        element_ids=ELEMENT_IDS,
    )
    print(f"localForce columns:    {er.list_components()}")
    print(f"list_canonicals():     {er.list_canonicals()}")

    # Per-canonical column mapping
    for canon in er.list_canonicals():
        cols = er.canonical_columns(canon)
        print(f"  {canon!r:30s} -> {cols}")

    # Full DataFrame for a canonical quantity
    first_canon = er.list_canonicals()[0]
    df = er.canonical(first_canon)
    print(f"\ncanonical({first_canon!r}) shape: {df.shape}")
    print(df.head(3))

    # closed-form → integrate_canonical raises (no gp_weights)
    try:
        er.integrate_canonical("axial_force")
    except ValueError as exc:
        print(f"\nintegrate_canonical on closed-form raises: {exc}")


# ---------------------------------------------------------------------------
# 9. ElementResults pickle round-trip
# ---------------------------------------------------------------------------
def section_9_pickle_element(er: ElementResults) -> None:
    section("9. ElementResults pickle round-trip")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "er.pkl"
        er.save_pickle(path)
        loaded = ElementResults.load_pickle(path)
        print(f"saved:  {path.stat().st_size} bytes")
        print(f"loaded: {loaded!r}")
        print(f"shapes match: {loaded.df.shape == er.df.shape}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    if not (DATASET_DIR / "results.mpco").exists():
        raise FileNotFoundError(
            f"elasticFrame fixture not found at {DATASET_DIR}.\n"
            "Check that stko_results_examples/ is present in the repo root."
        )

    ds = MPCODataSet(str(DATASET_DIR), RECORDER, verbose=False)

    section_1_introspect(ds)
    nr = section_2_fetch_nodal(ds)
    section_3_aggregations(nr)
    section_4_plotting(ds, nr)
    section_5_pickle_nodal(nr)
    er = section_6_fetch_element(ds)
    section_7_element_broker(er)
    section_8_canonical_names(ds)
    section_9_pickle_element(er)

    print()
    print("elastic_frame_example complete.")


if __name__ == "__main__":
    main()
