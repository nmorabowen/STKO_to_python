from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union
import re

import numpy as np
import pandas as pd
import h5py

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet
    from ..results.nodal_results_dataclass import NodalResults

logger = logging.getLogger(__name__)


def _flatten_node_ids(
    node_ids: Union[int, Sequence[Any], np.ndarray],
) -> np.ndarray:
    """Flatten scalar / 1-D / nested-sequence node IDs into a 1-D int64 array.

    Preserves legacy behavior: ``[[1,2],[3]]`` is a valid input and yields
    ``[1, 2, 3]``. A bare ``int`` becomes ``[int]``.
    """
    if isinstance(node_ids, (int, np.integer)):
        return np.asarray([int(node_ids)], dtype=np.int64)
    arr = np.asarray(node_ids, dtype=object)
    if arr.dtype == object:
        flat: list[int] = []
        for x in node_ids:  # type: ignore[union-attr]
            if isinstance(x, (list, tuple, np.ndarray)):
                flat.extend(int(v) for v in x)
            else:
                flat.append(int(x))
        return np.asarray(flat, dtype=np.int64)
    return np.asarray(node_ids, dtype=np.int64)


class NodeManager:
    """
    High-performance MPCO nodal reader and domain manager.

    Canonical name for the refactored Layer 3 domain manager. The legacy
    name ``Nodes`` is preserved as an alias at the bottom of this module
    for back-compat; new code should prefer ``NodeManager``.

    Public API:
        - _get_all_nodes_ids()
        - get_nodal_results()

    Optimizations:
        - One HDF5 open per (file, stage)
        - Multi-result read in a single pass over steps
        - Sorted fancy indexing + order restoration
        - One DataFrame per file per stage (not per result, not per step)
    """

    def __init__(self, dataset: "MPCODataSet") -> None:
        self.dataset = dataset
        self._node_index_df: Optional[pd.DataFrame] = None
        self._node_index_arr: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Node index (required by MPCODataSet)
    # ------------------------------------------------------------------

    @staticmethod
    def _node_dtype() -> np.dtype:
        return np.dtype(
            [
                ("node_id", "i8"),
                ("file_id", "i8"),
                ("index", "i8"),
                ("x", "f8"),
                ("y", "f8"),
                ("z", "f8"),
            ]
        )

    def _get_all_nodes_ids(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Build canonical node index: one row per node_id.
        """
        dtype = self._node_dtype()
        stage0 = self.dataset.model_stages[0]
        policy = self.dataset._format_policy
        chunks: list[np.ndarray] = []

        for file_id in self.dataset.results_partitions:
            with self.dataset._pool.with_partition(file_id) as h5:
                g = h5.get(policy.model_nodes_path(stage0))
                if g is None:
                    continue

                for key in g.keys():
                    if not key.startswith("ID"):
                        continue

                    node_ids = g[key][...].astype(np.int64, copy=False)
                    coord_key = key.replace("ID", "COORDINATES")
                    if coord_key not in g:
                        continue
                    coords = g[coord_key][...]

                    out = np.empty(len(node_ids), dtype=dtype)
                    out["node_id"] = node_ids
                    out["file_id"] = int(file_id)
                    out["index"] = np.arange(len(node_ids), dtype=np.int64)

                    if coords.shape[1] == 3:
                        out["x"], out["y"], out["z"] = coords.T
                    elif coords.shape[1] == 2:
                        out["x"], out["y"] = coords.T
                        out["z"] = 0.0
                    else:
                        raise ValueError(f"Unexpected COORDINATES shape: {coords.shape}")

                    chunks.append(out)

        if chunks:
            arr = np.concatenate(chunks)
            df = pd.DataFrame.from_records(arr)
            df = (
                df.sort_values(["node_id", "file_id", "index"], kind="mergesort")
                  .drop_duplicates("node_id", keep="first")
                  .sort_values("node_id", kind="mergesort")
                  .reset_index(drop=True)
            )
        else:
            arr = np.empty((0,), dtype=dtype)
            df = pd.DataFrame(columns=["node_id", "file_id", "index", "x", "y", "z"])

        self._node_index_arr = arr
        self._node_index_df = df

        if verbose:
            print(f"[Nodes] Indexed {len(df)} unique nodes")

        return {"array": arr, "dataframe": df}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_node_index_df(self) -> pd.DataFrame:
        if self._node_index_df is not None:
            return self._node_index_df

        ni = getattr(self.dataset, "nodes_info", None)
        if isinstance(ni, dict) and isinstance(ni.get("dataframe"), pd.DataFrame):
            self._node_index_df = ni["dataframe"]
            return self._node_index_df

        self._get_all_nodes_ids()
        return self._node_index_df  # type: ignore

    @staticmethod
    def _normalize_stages(stages, all_stages) -> Tuple[str, ...]:
        if stages is None:
            return tuple(all_stages)
        if isinstance(stages, str):
            return (stages,)
        return tuple(stages)

    def _normalize_results(self, results) -> Tuple[str, ...]:
        if results is None:
            return tuple(sorted(self.dataset.node_results_names))
        if isinstance(results, str):
            return (results,)
        return tuple(results)

    def _resolve_node_ids(
        self,
        *,
        node_ids: Union[int, Sequence[int], Sequence[Sequence[int]], np.ndarray, None],
        selection_set_id: Union[int, Sequence[int], None],
        selection_set_name: Union[str, Sequence[str], None],
    ) -> np.ndarray:
        resolver = self.dataset._selection_resolver
        explicit = _flatten_node_ids(node_ids) if node_ids is not None else None
        return resolver.resolve_nodes(
            names=selection_set_name,
            ids=selection_set_id,
            explicit_ids=explicit,
        )

    def _node_file_map(self, node_ids: np.ndarray) -> pd.DataFrame:
        df = self._ensure_node_index_df()
        sub = df[df["node_id"].isin(node_ids)][["node_id", "file_id", "index"]]
        if sub.empty:
            raise ValueError("No node IDs found in dataset.")

        return (
            sub.sort_values(["node_id", "file_id", "index"], kind="mergesort")
               .drop_duplicates("node_id", keep="first")
               .sort_values("node_id", kind="mergesort")
               .reset_index(drop=True)
        )

    @staticmethod
    def _sort_step_keys(keys: Sequence[str]) -> list[str]:
        try:
            return [k for _, k in sorted((int(k), k) for k in keys)]
        except Exception:
            rx = re.compile(r"(\d+)(?!.*\d)")
            try:
                return [k for _, k in sorted((int(rx.search(k).group(1)), k) for k in keys)]
            except Exception:
                return list(keys)

    # ------------------------------------------------------------------
    # Core optimized reader
    # ------------------------------------------------------------------

    @staticmethod
    def _read_multi_results_all_steps(
        *,
        h5: h5py.File,
        stage: str,
        results: Sequence[str],
        node_ids: np.ndarray,
        local_idx: np.ndarray,
    ) -> pd.DataFrame:
        order = np.argsort(local_idx, kind="mergesort")
        idx_sorted = local_idx[order]
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)

        ref = h5[f"{stage}/RESULTS/ON_NODES/{results[0]}/DATA"]
        step_names = Nodes._sort_step_keys(ref.keys())

        n_nodes = len(idx_sorted)
        n_steps = len(step_names)

        blocks = []
        lvl0, lvl1 = [], []

        for r in results:
            g = h5[f"{stage}/RESULTS/ON_NODES/{r}/DATA"]
            sample = g[step_names[0]][idx_sorted[:1]]
            ncomp = sample.shape[1]

            out = np.empty((n_steps * n_nodes, ncomp))
            for s, step in enumerate(step_names):
                arr = g[step][idx_sorted][:, :]
                out[s * n_nodes : (s + 1) * n_nodes, :] = arr[inv]

            blocks.append(out)
            lvl0.extend([r] * ncomp)
            lvl1.extend(range(1, ncomp + 1))

        big = np.hstack(blocks)
        cols = pd.MultiIndex.from_arrays([lvl0, lvl1], names=("result", "component"))

        df = pd.DataFrame(big, columns=cols)
        df["node_id"] = np.tile(node_ids, n_steps)
        df["step"] = np.repeat(np.arange(n_steps), n_nodes)
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_nodal_results(
        self,
        *,
        results_name: Union[str, Sequence[str], None] = None,
        model_stage: Union[str, Sequence[str], None] = None,
        node_ids: Union[int, Sequence[int], Sequence[Sequence[int]], np.ndarray, None] = None,
        selection_set_id: Union[int, Sequence[int], None] = None,
        selection_set_name: Union[str, Sequence[str], None] = None,
    ) -> "NodalResults":
        """Route through the dataset-owned query engine so every call
        benefits from the LRU cache. Thin wrapper; the actual read lives
        in :meth:`_fetch_nodal_results_uncached`.
        """
        engine = getattr(self.dataset, "_nodal_query_engine", None)
        if engine is not None:
            return engine.fetch(
                results_name=results_name,
                model_stage=model_stage,
                node_ids=node_ids,
                selection_set_id=selection_set_id,
                selection_set_name=selection_set_name,
            )
        # Fallback: engines may not yet be wired if this runs during
        # ``__init__`` bootstrap.
        return self._fetch_nodal_results_uncached(
            results_name=results_name,
            model_stage=model_stage,
            node_ids=node_ids,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
        )

    def _fetch_nodal_results_uncached(
        self,
        *,
        results_name: Union[str, Sequence[str], None] = None,
        model_stage: Union[str, Sequence[str], None] = None,
        node_ids: Union[int, Sequence[int], Sequence[Sequence[int]], np.ndarray, None] = None,
        selection_set_id: Union[int, Sequence[int], None] = None,
        selection_set_name: Union[str, Sequence[str], None] = None,
    ) -> "NodalResults":
        """Uncached read path — used by the engine and by the public
        ``get_nodal_results`` fallback when no engine is attached.
        """
        stages = self._normalize_stages(model_stage, self.dataset.model_stages)
        results = self._normalize_results(results_name)

        ids = self._resolve_node_ids(
            node_ids=node_ids,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
        )
        ids_sorted = np.sort(ids)

        fmap = self._node_file_map(ids_sorted)
        file_groups = {fid: g for fid, g in fmap.groupby("file_id")}

        coords_df = (
            self._ensure_node_index_df()
            .drop_duplicates("node_id")
            .set_index("node_id")
            .loc[ids_sorted, ["x", "y", "z"]]
        )

        # Multi-stage fetches use a *contiguous global* step axis: each
        # stage's local step 0..N-1 is shifted by the cumulative step
        # count of all earlier stages. ``stage_step_ranges`` records the
        # global ``(start, end)`` per stage for downstream filtering and
        # plot-boundary annotation.
        stage_frames: list[pd.DataFrame] = []
        stage_step_ranges: dict[str, tuple[int, int]] = {}
        stage_layouts: dict[str, tuple] = {}
        step_offset = 0

        for st in stages:
            file_frames = []
            for fid, grp in file_groups.items():
                with self.dataset._pool.with_partition(int(fid)) as h5:
                    file_frames.append(
                        self._read_multi_results_all_steps(
                            h5=h5,
                            stage=st,
                            results=results,
                            node_ids=grp["node_id"].to_numpy(),
                            local_idx=grp["index"].to_numpy(),
                        )
                    )

            df_stage = pd.concat(file_frames, ignore_index=True)
            n_steps_stage = (
                int(df_stage["step"].max()) + 1 if not df_stage.empty else 0
            )
            df_stage["step"] = df_stage["step"] + step_offset
            df_stage = df_stage.set_index(["node_id", "step"]).sort_index()

            stage_frames.append(df_stage)
            stage_step_ranges[st] = (step_offset, step_offset + n_steps_stage)
            stage_layouts[st] = tuple(map(tuple, df_stage.columns))
            step_offset += n_steps_stage

        # Cross-stage layout check: same column tuple per stage. Mirrors
        # the bucket-by-bucket detail in
        # :func:`MpcoFormatError` raises elsewhere (see
        # ``ElementManager._validate_homogeneous_layouts_across_stages``
        # for the equivalent on the element side).
        if len(stages) > 1:
            unique_layouts = {v for v in stage_layouts.values()}
            if len(unique_layouts) > 1:
                from ..io.meta_parser import MpcoFormatError

                raise MpcoFormatError(
                    f"Heterogeneous nodal column layouts across "
                    f"MODEL_STAGE groups for "
                    f"results={tuple(results)!r}: "
                    f"{len(unique_layouts)} distinct layouts across "
                    f"{len(stages)} stages. Restart-style models that "
                    f"change topology between stages must be queried "
                    f"one stage at a time. Layouts: "
                    f"{ {s: [list(c) for c in cols] for s, cols in stage_layouts.items()} }"
                )

        df = (
            stage_frames[0]
            if len(stage_frames) == 1
            else pd.concat(stage_frames).sort_index()
        )

        # Stitch time arrays end-to-end. Each stage's TIME records
        # absolute time, so the concatenation is naturally monotonic
        # for the gravity → pushover → dynamic workflow this targets.
        time_chunks: list[np.ndarray] = []
        for st in stages:
            try:
                t = self.dataset.time.loc[st]["TIME"].to_numpy()
            except (KeyError, AttributeError):
                start, end = stage_step_ranges[st]
                t = np.full(end - start, np.nan, dtype=np.float64)
            n = stage_step_ranges[st][1] - stage_step_ranges[st][0]
            time_chunks.append(np.asarray(t, dtype=np.float64)[:n])
        time = (
            np.concatenate(time_chunks) if time_chunks else np.array([])
        )

        component_names = tuple("|".join(map(str, c)) for c in df.columns)

        from ..results.nodal_results_dataclass import NodalResults
        return NodalResults(
            df=df,
            time=time,
            name=self.dataset.name,
            nodes_ids=tuple(ids_sorted),
            nodes_info=coords_df,
            results_components=component_names,
            model_stages=stages,
            stage_step_ranges=stage_step_ranges,
            plot_settings=self.dataset.plot_settings,
            selection_set=self.dataset.selection_set,
            analysis_time=self.dataset.info.analysis_time,
            size=self.dataset.info.size,
        )


# Back-compat alias — see class docstring. The legacy name ``Nodes`` is
# guaranteed importable from ``STKO_to_python.nodes.nodes`` and from the
# ``STKO_to_python.nodes`` package. Do not remove without a migration
# deprecation cycle.
Nodes = NodeManager
