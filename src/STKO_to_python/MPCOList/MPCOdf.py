from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, Literal, TYPE_CHECKING
from collections.abc import Callable, Iterable, Sequence

if TYPE_CHECKING:
    from .MPCOResults import MPCOResults


class MPCO_df:
    def __init__(self, results: "MPCOResults"):
        self.results = results

    # ------------------------------------------------------------------
    # MPCOResults passthrough (convenience)
    # ------------------------------------------------------------------

    def select(self, *args, **kwargs):
        return self.results.select(*args, **kwargs)

    def compute_table(self, *args, **kwargs):
        return self.results.compute_table(*args, **kwargs)

    def parse_tier_letter(self, *args, **kwargs):
        return self.results.parse_tier_letter(*args, **kwargs)

    def _label_for(self, *args, **kwargs):
        return self.results._label_for(*args, **kwargs)

    def _tag_from_key(self, *args, **kwargs):
        return self.results._tag_from_key(*args, **kwargs)

    def _group_key(self, *args, **kwargs):
        return self.results._group_key(*args, **kwargs)

    def _normalize_grouping_spec(self, *args, **kwargs):
        return self.results._normalize_grouping_spec(*args, **kwargs)

    # ------------------------------------------------------------------
    # MPCO-specific DataFrame extractors
    # ------------------------------------------------------------------

    def wide_to_long(
        self,
        df_wide: pd.DataFrame,
        *,
        id_cols: Sequence[str] = ("Tier", "Case", "sta", "rup"),
        value_col: str = "EDP",

        # run-level clustering key (default: station–rupture)
        runkey_col: str = "runkey",
        runkey_from: Sequence[str] = ("sta", "rup"),
        runkey_sep: str = ":",

        # metadata (constant over df)
        result_name: str | None = None,
        component: str | int | None = None,
        reduce_time: str | None = None,
        relative_drift: bool | None = None,
        op: str = "raw",   # metadata only
    ) -> pd.DataFrame:
        """
        Canonical WIDE -> LONG conversion.

        Input (WIDE)
            Tier | Case | sta | rup | EDP   (plus optional runkey)

        Output (LONG)
            Tier | Case | sta | rup | runkey | component | result_name |
            reduce_time | relative_drift | op | edp
        """
        if not isinstance(df_wide, pd.DataFrame):
            raise TypeError("wide_to_long: df_wide must be a pandas DataFrame.")

        id_cols = list(id_cols)
        required = list(id_cols) + [value_col]
        missing = [c for c in required if c not in df_wide.columns]
        if missing:
            raise ValueError(f"wide_to_long: missing columns={missing}. Got cols={list(df_wide.columns)}")

        if op not in ("raw", "log"):
            raise ValueError("wide_to_long: op must be 'raw' or 'log'.")

        # ----------------------------
        # core columns
        # ----------------------------
        df = df_wide.loc[:, id_cols + [value_col]].copy()
        df = df.rename(columns={value_col: "edp"})
        df["edp"] = pd.to_numeric(df["edp"], errors="coerce")

        # ----------------------------
        # runkey
        # ----------------------------
        if runkey_col in df_wide.columns:
            df[runkey_col] = df_wide[runkey_col].astype(str)
        else:
            runkey_from = list(runkey_from)
            rk_missing = [c for c in runkey_from if c not in df.columns]
            if rk_missing:
                raise ValueError(f"wide_to_long: cannot build runkey; missing columns={rk_missing}")

            rk = df[runkey_from[0]].astype(str)
            for c in runkey_from[1:]:
                rk = rk + runkey_sep + df[c].astype(str)
            df[runkey_col] = rk

        # ----------------------------
        # attach metadata (constant)
        # ----------------------------
        df["result_name"] = pd.NA if result_name is None else str(result_name)
        df["component"] = pd.NA if component is None else str(component)
        df["reduce_time"] = pd.NA if reduce_time is None else str(reduce_time)
        df["relative_drift"] = pd.NA if relative_drift is None else bool(relative_drift)
        df["op"] = str(op)

        # ----------------------------
        # dtypes (design factors)
        # ----------------------------
        if "Tier" in df.columns:
            df["Tier"] = pd.to_numeric(df["Tier"], errors="coerce").astype("Int64")
            df["Tier"] = df["Tier"].astype("category")

        for c in ("Case", "sta", "rup"):
            if c in df.columns:
                df[c] = df[c].astype(str).astype("category")

        df[runkey_col] = df[runkey_col].astype(str).astype("category")
        df["op"] = df["op"].astype("category")

        # keep these as categoricals even if currently all NA (this helps consistency across EDP tables)
        for c in ("result_name", "component", "reduce_time"):
            df[c] = df[c].astype("category")

        # nullable boolean
        df["relative_drift"] = df["relative_drift"].astype("boolean")

        # ----------------------------
        # output
        # ----------------------------
        out_cols = [
            *id_cols,
            runkey_col,
            "component",
            "result_name",
            "reduce_time",
            "relative_drift",
            "op",
            "edp",
        ]
        return df[out_cols]

    def drift_df(
        self,
        *,
        top: tuple[float, float, float],
        bottom: tuple[float, float, float],
        components: Sequence[int] = (1, 2),
        result_name: str = "DISPLACEMENT",
        relative_drift: bool = True,
        reduce_time: str = "abs_max",  # "abs_max" | "max" | "min" | "rms"
        stage: str | None = None,
        combine: str = "srss",  # "srss" | "maxabs" | "none" (none => single comp)
        op: str = "raw",  # "raw" | "log"
        eps_log: float = 1e-16,
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        order: Callable[[tuple[str, str, str], Any], Any] | None = None,
    ) -> pd.DataFrame:
        """
        Canonical WIDE:
            Tier | Case | sta | rup | EDP
        """
        if reduce_time not in ("abs_max", "max", "min", "rms"):
            raise ValueError("drift_df: reduce_time must be one of: 'abs_max', 'max', 'min', 'rms'.")
        if combine not in ("srss", "maxabs", "none"):
            raise ValueError("drift_df: combine must be one of: 'srss', 'maxabs', 'none'.")
        if op not in ("raw", "log"):
            raise ValueError("drift_df: op must be 'raw' or 'log'.")
        if op == "log" and eps_log <= 0:
            raise ValueError("drift_df: eps_log must be > 0 when op='log'.")

        comps = tuple(int(c) for c in components)
        if not comps:
            raise ValueError("drift_df: components must be non-empty.")
        if combine == "none" and len(comps) != 1:
            raise ValueError("drift_df: combine='none' requires a single component.")

        def _reduce(y: np.ndarray) -> float:
            y = np.asarray(y, dtype=float)
            y = y[np.isfinite(y)]
            if y.size == 0:
                return float("nan")
            if reduce_time == "abs_max":
                return float(np.nanmax(np.abs(y)))
            if reduce_time == "max":
                return float(np.nanmax(y))
            if reduce_time == "min":
                return float(np.nanmin(y))
            if reduce_time == "rms":
                return float(np.sqrt(np.nanmean(y * y)))
            raise RuntimeError("unreachable")

        def _apply_op(x: float) -> float:
            if not np.isfinite(x):
                return float("nan")
            if op == "raw":
                return float(x)
            # drift edp is nonnegative by construction (abs/rms typically), but guard anyway
            return float(np.log(np.maximum(float(x), eps_log)))

        def _select_stage(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
            """
            If results are multi-stage (stage, node_id, step), select:
            - the requested stage if provided and present
            - otherwise the last stage in sorted order (deterministic)
            If results are NOT staged (node_id, step), return unchanged.
            """
            if not isinstance(s, (pd.Series, pd.DataFrame)):
                return s

            idx = s.index
            if not isinstance(idx, pd.MultiIndex):
                return s

            # Only treat as "staged" if it is exactly (stage, node_id, step)
            if idx.nlevels != 3:
                return s

            # stage level is level 0 by convention in your NodalResults.fetch()
            stages = tuple(sorted({str(x) for x in idx.get_level_values(0)}))
            if not stages:
                return s.iloc[0:0]

            if stage is not None:
                st = str(stage)
                if st in stages:
                    return s.xs(st, level=0)
                # if user asked for a missing stage, fail loudly (better than silently wrong)
                raise ValueError(f"Requested stage={st!r} not found. Available stages={stages}")

            # default: last stage
            return s.xs(stages[-1], level=0)


        def _delta_u_series(nr: Any, *, comp: int, top_id: int, bot_id: int) -> pd.Series:
            s = nr.fetch(result_name=result_name, component=comp, node_ids=[top_id, bot_id])
            s = _select_stage(s)
            if not (isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 2):
                return pd.Series(dtype=float)
            u_top = s.xs(top_id, level=0).sort_index()
            u_bot = s.xs(bot_id, level=0).sort_index()
            u_top, u_bot = u_top.align(u_bot, join="inner")
            return (u_top - u_bot).astype(float)

        def _drift_series(nr: Any, *, comp: int) -> pd.Series:
            try:
                s = nr.drift(
                    top=top,
                    bottom=bottom,
                    component=comp,
                    result_name=result_name,
                    stage=stage,
                    reduce="series",
                )
            except TypeError:
                try:
                    s = nr.drift(
                        top=top,
                        bottom=bottom,
                        component=comp,
                        result_name=result_name,
                        reduce="series",
                    )
                except TypeError:
                    s = nr.drift(top=top, bottom=bottom, component=comp, reduce="series")

            s = _select_stage(s)
            if isinstance(s, pd.DataFrame):
                if s.shape[1] == 1:
                    s = s.iloc[:, 0]
                else:
                    return pd.Series(dtype=float)
            if not isinstance(s, pd.Series):
                return pd.Series(dtype=float)
            return s.sort_index().astype(float)

        def _series_for_component(nr: Any, *, comp: int, top_id: int, bot_id: int) -> pd.Series:
            return _drift_series(nr, comp=comp) if relative_drift else _delta_u_series(nr, comp=comp, top_id=top_id, bot_id=bot_id)

        def _combine_series(series_list: list[pd.Series]) -> pd.Series:
            series_list = [s for s in series_list if isinstance(s, pd.Series) and not s.empty]
            if not series_list:
                return pd.Series(dtype=float)
            if len(series_list) == 1:
                return series_list[0].sort_index()

            dfc = pd.concat(series_list, axis=1, join="inner").astype(float).dropna(how="any")
            if dfc.empty:
                return pd.Series(dtype=float)

            vals = dfc.to_numpy(dtype=float)
            if combine == "srss":
                r = np.sqrt(np.sum(vals * vals, axis=1))
            elif combine == "maxabs":
                r = np.max(np.abs(vals), axis=1)
            elif combine == "none":
                r = vals[:, 0]
            else:
                raise RuntimeError("unreachable")
            return pd.Series(r, index=dfc.index)

        pairs = self.select(model=model, station=station, rupture=rupture, order=order)
        out_cols = ["Tier", "Case", "sta", "rup", "EDP"]
        if not pairs:
            return pd.DataFrame(columns=out_cols)

        rows: list[dict[str, Any]] = []
        for (m, sta, rup), nr in pairs:
            tier, case = self.parse_tier_letter(m)

            top_id = int(nr.info.nearest_node_id([top], return_distance=False)[0])
            bot_id = int(nr.info.nearest_node_id([bottom], return_distance=False)[0])

            series_list: list[pd.Series] = []
            for comp in comps:
                s = _series_for_component(nr, comp=comp, top_id=top_id, bot_id=bot_id)
                if not s.empty:
                    series_list.append(s)

            r = _combine_series(series_list)
            val = _reduce(r.to_numpy(dtype=float)) if not r.empty else float("nan")
            val = _apply_op(val)

            rows.append(
                dict(
                    Tier=int(tier),
                    Case=str(case),
                    sta=str(sta),
                    rup=str(rup),
                    EDP=float(val) if np.isfinite(val) else float("nan"),
                )
            )

        df = pd.DataFrame(rows)
        df["Tier"] = df["Tier"].astype(int)
        df["Case"] = df["Case"].astype("category")
        df["sta"] = df["sta"].astype("category")
        df["rup"] = df["rup"].astype("category")
        return df[out_cols]

    def drift_df_long(
        self,
        *,
        top: tuple[float, float, float],
        bottom: tuple[float, float, float],
        components: Sequence[int] = (1, 2),
        result_name: str = "DISPLACEMENT",
        relative_drift: bool = True,
        reduce_time: str = "abs_max",
        stage: str | None = None,
        combine: str = "srss",
        op: str = "log",
        eps_log: float = 1e-16,
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        order: Callable[[tuple[str, str, str], Any], Any] | None = None,
    ) -> pd.DataFrame:
        df_wide = self.drift_df(
            top=top,
            bottom=bottom,
            components=components,
            result_name=result_name,
            relative_drift=relative_drift,
            reduce_time=reduce_time,
            stage=stage,
            combine=combine,
            op=op,
            eps_log=eps_log,
            model=model,
            station=station,
            rupture=rupture,
            order=order,
        )

        comp_meta: str | int | None
        if combine == "none":
            comp_meta = int(tuple(int(c) for c in components)[0])
        else:
            comp_meta = str(combine)

        return self.wide_to_long(
            df_wide,
            id_cols=("Tier", "Case", "sta", "rup"),
            value_col="EDP",
            result_name=result_name,
            component=comp_meta,
            reduce_time=reduce_time,
            relative_drift=relative_drift,
            op=op,
        )

    def pga_df(
        self,
        *,
        node: tuple[float, float, float],
        components: Sequence[int] = (1, 2),
        result_name: str = "ACCELERATION",
        stage: str | None = None,
        combine: str = "srss",  # "srss" | "maxabs" | "none"
        reduce_time: str = "abs_max",  # "abs_max" | "max" | "min" | "rms"
        op: str = "raw",  # "raw" | "log"
        eps_log: float = 1e-16,
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        order: Callable[[tuple[str, str, str], Any], Any] | None = None,
    ) -> pd.DataFrame:
        if reduce_time not in ("abs_max", "max", "min", "rms"):
            raise ValueError("pga_df: invalid reduce_time.")
        if combine not in ("srss", "maxabs", "none"):
            raise ValueError("pga_df: invalid combine.")
        if op not in ("raw", "log"):
            raise ValueError("pga_df: invalid op.")
        if op == "log" and eps_log <= 0:
            raise ValueError("pga_df: eps_log must be > 0.")

        comps = tuple(int(c) for c in components)
        if not comps:
            raise ValueError("pga_df: components must be non-empty.")
        if combine == "none" and len(comps) != 1:
            raise ValueError("pga_df: combine='none' requires a single component.")

        def _reduce(y: np.ndarray) -> float:
            y = np.asarray(y, dtype=float)
            y = y[np.isfinite(y)]
            if y.size == 0:
                return float("nan")
            if reduce_time == "abs_max":
                return float(np.max(np.abs(y)))
            if reduce_time == "max":
                return float(np.max(y))
            if reduce_time == "min":
                return float(np.min(y))
            if reduce_time == "rms":
                return float(np.sqrt(np.mean(y * y)))
            raise RuntimeError("unreachable")

        def _apply_op(x: float) -> float:
            if not np.isfinite(x):
                return float("nan")
            if op == "raw":
                return float(x)
            return float(np.log(np.maximum(float(x), eps_log)))

        def _select_stage(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
            """
            If results are multi-stage (stage, node_id, step), select:
            - the requested stage if provided and present
            - otherwise the last stage in sorted order (deterministic)
            If results are NOT staged (node_id, step), return unchanged.
            """
            if not isinstance(s, (pd.Series, pd.DataFrame)):
                return s

            idx = s.index
            if not isinstance(idx, pd.MultiIndex):
                return s

            # Only treat as "staged" if it is exactly (stage, node_id, step)
            if idx.nlevels != 3:
                return s

            # stage level is level 0 by convention in your NodalResults.fetch()
            stages = tuple(sorted({str(x) for x in idx.get_level_values(0)}))
            if not stages:
                return s.iloc[0:0]

            if stage is not None:
                st = str(stage)
                if st in stages:
                    return s.xs(st, level=0)
                # if user asked for a missing stage, fail loudly (better than silently wrong)
                raise ValueError(f"Requested stage={st!r} not found. Available stages={stages}")

            # default: last stage
            return s.xs(stages[-1], level=0)


        def _acc_series(nr: Any, *, comp: int, node_id: int) -> pd.Series:
            s = nr.fetch(result_name=result_name, component=comp, node_ids=[node_id])
            s = _select_stage(s)
            if not (isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 2):
                return pd.Series(dtype=float)
            a = s.xs(node_id, level=0).sort_index()
            return a.astype(float) if isinstance(a, pd.Series) else pd.Series(dtype=float)

        def _combine_series(series_list: list[pd.Series]) -> pd.Series:
            series_list = [s for s in series_list if isinstance(s, pd.Series) and not s.empty]
            if not series_list:
                return pd.Series(dtype=float)
            if len(series_list) == 1:
                return series_list[0]

            dfc = pd.concat(series_list, axis=1, join="inner").astype(float).dropna(how="any")
            if dfc.empty:
                return pd.Series(dtype=float)

            vals = dfc.to_numpy(dtype=float)
            if combine == "srss":
                r = np.sqrt(np.sum(vals * vals, axis=1))
            elif combine == "maxabs":
                r = np.max(np.abs(vals), axis=1)
            else:  # none
                r = vals[:, 0]
            return pd.Series(r, index=dfc.index)

        pairs = self.select(model=model, station=station, rupture=rupture, order=order)
        if not pairs:
            return pd.DataFrame(columns=["Tier", "Case", "sta", "rup", "EDP"])

        rows: list[dict[str, Any]] = []
        for (m, sta, rup), nr in pairs:
            tier, case = self.parse_tier_letter(m)

            node_id = int(nr.info.nearest_node_id([node], return_distance=False)[0])

            series_list: list[pd.Series] = []
            for c in comps:
                s = _acc_series(nr, comp=c, node_id=node_id)
                if not s.empty:
                    series_list.append(s)

            r = _combine_series(series_list)
            val = _reduce(r.to_numpy(dtype=float)) if not r.empty else float("nan")
            val = _apply_op(val)

            rows.append(
                dict(
                    Tier=int(tier),
                    Case=str(case),
                    sta=str(sta),
                    rup=str(rup),
                    EDP=float(val) if np.isfinite(val) else float("nan"),
                )
            )

        df = pd.DataFrame(rows)
        df["Tier"] = df["Tier"].astype(int)
        df["Case"] = df["Case"].astype("category")
        df["sta"] = df["sta"].astype("category")
        df["rup"] = df["rup"].astype("category")
        return df[["Tier", "Case", "sta", "rup", "EDP"]]

    def pga_df_long(
        self,
        *,
        node: tuple[float, float, float],
        components: Sequence[int] = (1, 2),
        result_name: str = "ACCELERATION",
        stage: str | None = None,
        combine: str = "srss",
        reduce_time: str = "abs_max",
        op: str = "log",
        eps_log: float = 1e-16,
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        order: Callable[[tuple[str, str, str], Any], Any] | None = None,
    ) -> pd.DataFrame:
        df_wide = self.pga_df(
            node=node,
            components=components,
            result_name=result_name,
            stage=stage,
            combine=combine,
            reduce_time=reduce_time,
            op=op,
            eps_log=eps_log,
            model=model,
            station=station,
            rupture=rupture,
            order=order,
        )

        comp_meta: str | int = int(tuple(components)[0]) if combine == "none" else str(combine)

        return self.wide_to_long(
            df_wide,
            value_col="EDP",
            result_name=result_name,
            component=comp_meta,
            reduce_time=reduce_time,
            relative_drift=None,
            op=op,
        )

    def pga_df_mod(
        self,
        *,
        node: int | tuple[float, float] | tuple[float, float, float],
        components: tuple[int, ...] = (1, 2),
        result_name: str = "ACCELERATION",
        combine: str = "srss",                 # "srss" | "maxabs" | "none"
        reduce_time: str = "abs_max",          # "abs_max" | "max" | "min"
        op: str | None = None,                 # None | "log"
        eps_log: float = 1e-16,
        stage: str | None = None,
        # ---- Case A correction ----
        fix_A_relative: bool = True,
        motions_root: str | Path | None = Path(r"C:\Users\nmb\Dropbox\UANDES EC\San Ramon v3\motions_reduced"),
        # ---- selection like the original pga_df_long ----
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        order: Callable[[tuple[str, str, str], Any], Any] | None = None,
    ) -> pd.DataFrame:
        """
        Compute PGA per (Tier, Case, sta, rup), with optional Case A correction.

        Robust to different record lengths (e.g., model 40s vs GM 60s):
        - alignment is always done by INNER join on the step index.
        """

        if fix_A_relative and motions_root is None:
            raise ValueError("motions_root must be provided when fix_A_relative=True.")
        motions_root = Path(motions_root) if motions_root is not None else None

        if combine not in ("srss", "maxabs", "none"):
            raise ValueError("combine must be 'srss', 'maxabs', or 'none'.")
        if reduce_time not in ("abs_max", "max", "min"):
            raise ValueError("reduce_time must be 'abs_max', 'max', or 'min'.")
        if op not in (None, "log"):
            raise ValueError("op must be None or 'log'.")
        if op == "log" and eps_log <= 0:
            raise ValueError("eps_log must be > 0 when op='log'.")
        if not components:
            raise ValueError("components must be non-empty.")
        if combine == "none" and len(components) != 1:
            raise ValueError("combine='none' requires a single component.")

        comps = tuple(int(c) for c in components)

        def _norm_sta_folder(sta: str) -> str:
            s = str(sta)
            return s if s.startswith("sta_") else f"sta_{s}"

        def _load_ground_acc_step(sta: str, rup: str, comp: int) -> pd.Series:
            """
            Reads acceleration.txt:
                time ax ay az   (m/s^2)

            Returns:
                Series indexed by integer step (0..N-1), values in m/s^2
            """
            sta_dir = _norm_sta_folder(sta)
            rup_s = str(rup)

            candidates: list[Path] = [
                motions_root / sta_dir / rup_s / "acceleration.txt",
            ]
            if not rup_s.startswith("rup_"):
                candidates.append(motions_root / sta_dir / f"rup_{rup_s}" / "acceleration.txt")

            last_err: Exception | None = None
            for f in candidates:
                try:
                    data = np.loadtxt(f, skiprows=1)
                    if data.ndim != 2 or data.shape[1] != 4:
                        raise ValueError(f"Expected (t,ax,ay,az) in {f}, got shape={data.shape}")
                    a = data[:, comp]  # comp=1,2,3
                    return pd.Series(
                        a.astype(float),
                        index=pd.RangeIndex(len(a), name="step"),
                        name=f"ag[{comp}]",
                    )
                except Exception as e:
                    last_err = e
                    continue

            raise FileNotFoundError(
                f"Could not load acceleration.txt for sta={sta!r}, rup={rup!r}. Tried: {candidates}"
            ) from last_err

        def _select_stage(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
            if not isinstance(s, (pd.Series, pd.DataFrame)):
                return s
            idx = s.index
            if not isinstance(idx, pd.MultiIndex) or idx.nlevels != 3:
                return s

            stages = tuple(sorted({str(x) for x in idx.get_level_values(0)}))
            if not stages:
                return s.iloc[0:0]

            if stage is not None:
                st = str(stage)
                if st in stages:
                    return s.xs(st, level=0)
                raise ValueError(f"Requested stage={st!r} not found. Available stages={stages}")

            return s.xs(stages[-1], level=0)

        def _reduce(arr: np.ndarray) -> float:
            if arr.size == 0:
                return float("nan")
            if reduce_time == "abs_max":
                return float(np.nanmax(np.abs(arr)))
            if reduce_time == "max":
                return float(np.nanmax(arr))
            if reduce_time == "min":
                return float(np.nanmin(arr))
            raise RuntimeError("unreachable")

        def _apply_op(x: float) -> float:
            if not np.isfinite(x):
                return float("nan")
            if op is None:
                return float(x)
            return float(np.log(np.maximum(float(x), float(eps_log))))

        pairs = self.select(model=model, station=station, rupture=rupture, order=order)
        rows: list[dict[str, Any]] = []

        for (model_name, sta, rup), nr in pairs:
            tier, case = self.parse_tier_letter(model_name)

            if isinstance(node, (int, np.integer)):
                node_id = int(node)
            else:
                node_id = int(nr.info.nearest_node_id([node], return_distance=False)[0])

            series_list: list[pd.Series] = []

            for comp in comps:
                try:
                    s = nr.fetch(result_name=result_name, component=comp, node_ids=[node_id])
                except Exception:
                    continue

                s = _select_stage(s)

                if not (isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 2):
                    continue

                s = s.xs(node_id, level=0).sort_index().astype(float)
                if s.empty:
                    continue

                # Case A correction: relative -> absolute
                if fix_A_relative and str(case) == "A":
                    ag = _load_ground_acc_step(str(sta), str(rup), int(comp))
                    # INNER alignment makes 60s vs 40s safe
                    s, ag = s.align(ag, join="inner")
                    if s.empty:
                        continue
                    s = s + 1000.0 * ag  # m/s^2 -> mm/s^2

                series_list.append(s)

            if not series_list:
                continue

            # Align components on common steps
            dfc = pd.concat(series_list, axis=1, join="inner").astype(float).dropna(how="any")
            if dfc.empty:
                continue

            vals = dfc.to_numpy(dtype=float)  # (nsteps, ncomp)

            if combine == "srss":
                combined = np.sqrt(np.nansum(vals * vals, axis=1))
            elif combine == "maxabs":
                combined = np.nanmax(np.abs(vals), axis=1)
            else:  # "none"
                combined = vals[:, 0]

            pga_val = _apply_op(_reduce(combined))

            rows.append(
                dict(
                    Tier=int(tier),
                    Case=str(case),
                    sta=str(sta),
                    rup=str(rup),
                    pga=float(pga_val) if np.isfinite(pga_val) else float("nan"),
                )
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["Tier"] = df["Tier"].astype(int)
        df["Case"] = df["Case"].astype("category")
        df["sta"] = df["sta"].astype("category")
        df["rup"] = df["rup"].astype("category")
        return df


    def pga_df_long_mod(
        self,
        *,
        node: int | tuple[float, float] | tuple[float, float, float],
        components: tuple[int, ...] = (1, 2),
        result_name: str = "ACCELERATION",
        stage: str | None = None,
        combine: str = "srss",                  # "srss" | "maxabs" | "none"
        reduce_time: str = "abs_max",           # "abs_max" | "max" | "min"
        op: str | None = "log",                 # None | "log"
        eps_log: float = 1e-16,
        fix_A_relative: bool = True,
        motions_root: str | Path | None = Path(r"C:\Users\nmb\Dropbox\UANDES EC\San Ramon v3\motions_reduced"),
        # keep same structure as pga_df_long:
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        order: Callable[[tuple[str, str, str], Any], Any] | None = None,
    ) -> pd.DataFrame:
        """
        LONG-format PGA table (BayesStatsModel compatible).

        Output columns (LONG):
            Tier | Case | sta | rup | runkey | component | result_name |
            reduce_time | relative_drift | op | edp
        """

        if op not in (None, "log"):
            raise ValueError("pga_df_long_mod: op must be None or 'log'.")
        if op == "log" and eps_log <= 0:
            raise ValueError("pga_df_long_mod: eps_log must be > 0 when op='log'.")

        df_wide = self.pga_df_mod(
            node=node,
            components=components,
            result_name=result_name,
            combine=combine,
            reduce_time=reduce_time,
            op=op,
            eps_log=eps_log,
            stage=stage,
            fix_A_relative=fix_A_relative,
            motions_root=motions_root,
            model=model,
            station=station,
            rupture=rupture,
            order=order,
        )

        # Always return a proper LONG table (even if empty)
        if df_wide.empty:
            return pd.DataFrame(
                columns=[
                    "Tier", "Case", "sta", "rup", "runkey",
                    "component", "result_name", "reduce_time",
                    "relative_drift", "op", "edp",
                ]
            )

        df_wide = df_wide.copy()
        df_wide["EDP"] = pd.to_numeric(df_wide["pga"], errors="coerce")

        # component metadata to match your LONG conventions
        comp_meta: str | int = int(components[0]) if combine == "none" else str(combine)

        return self.wide_to_long(
            df_wide,
            id_cols=("Tier", "Case", "sta", "rup"),
            value_col="EDP",
            result_name=result_name,
            component=comp_meta,
            reduce_time=reduce_time,
            relative_drift=None,
            op=("log" if op == "log" else "raw"),
        )





    # ------------------------------------------------------------------
    # TORSION (WIDE + LONG)
    # ------------------------------------------------------------------

    def torsion_df(
        self,
        *,
        z_coord: float,
        node_a_xy: tuple[float, float],
        node_b_xy: tuple[float, float],
        result_name: str = "DISPLACEMENT",
        stage: str | None = None,
        reduce_time: str = "abs_max",
        op: str = "raw",
        eps_log: float = 1e-16,
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        order: Callable[[tuple[str, str, str], Any], Any] | None = None,
    ) -> pd.DataFrame:
        if reduce_time not in ("abs_max", "max", "min", "rms"):
            raise ValueError("torsion_df: invalid reduce_time.")
        if op not in ("raw", "log"):
            raise ValueError("torsion_df: invalid op.")
        if op == "log" and eps_log <= 0:
            raise ValueError("torsion_df: eps_log must be > 0.")

        xa, ya = map(float, node_a_xy)
        xb, yb = map(float, node_b_xy)
        z = float(z_coord)

        dx = xb - xa
        dy = yb - ya
        L = float(np.hypot(dx, dy))
        if L <= 0.0:
            raise ValueError("torsion_df: node_a_xy and node_b_xy must be distinct.")

        nx = -dy / L
        ny = dx / L

        def _reduce(y: np.ndarray) -> float:
            y = np.asarray(y, dtype=float)
            y = y[np.isfinite(y)]
            if y.size == 0:
                return float("nan")
            if reduce_time == "abs_max":
                return float(np.max(np.abs(y)))
            if reduce_time == "max":
                return float(np.max(y))
            if reduce_time == "min":
                return float(np.min(y))
            if reduce_time == "rms":
                return float(np.sqrt(np.mean(y * y)))
            raise RuntimeError("unreachable")

        def _apply_op(x: float) -> float:
            if not np.isfinite(x):
                return float("nan")
            if op == "raw":
                return float(x)
            # torsion can be +/-; define log on magnitude
            return float(np.log(np.maximum(abs(float(x)), eps_log)))

        def _select_stage(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
            """
            If results are multi-stage (stage, node_id, step), select:
            - the requested stage if provided and present
            - otherwise the last stage in sorted order (deterministic)
            If results are NOT staged (node_id, step), return unchanged.
            """
            if not isinstance(s, (pd.Series, pd.DataFrame)):
                return s

            idx = s.index
            if not isinstance(idx, pd.MultiIndex):
                return s

            # Only treat as "staged" if it is exactly (stage, node_id, step)
            if idx.nlevels != 3:
                return s

            # stage level is level 0 by convention in your NodalResults.fetch()
            stages = tuple(sorted({str(x) for x in idx.get_level_values(0)}))
            if not stages:
                return s.iloc[0:0]

            if stage is not None:
                st = str(stage)
                if st in stages:
                    return s.xs(st, level=0)
                # if user asked for a missing stage, fail loudly (better than silently wrong)
                raise ValueError(f"Requested stage={st!r} not found. Available stages={stages}")

            # default: last stage
            return s.xs(stages[-1], level=0)


        def _disp_series(nr: Any, *, comp: int, node_id: int) -> pd.Series:
            s = nr.fetch(result_name=result_name, component=comp, node_ids=[node_id])
            s = _select_stage(s)
            if not (isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 2):
                return pd.Series(dtype=float)
            u = s.xs(node_id, level=0).sort_index()
            return u.astype(float) if isinstance(u, pd.Series) else pd.Series(dtype=float)

        pairs = self.select(model=model, station=station, rupture=rupture, order=order)
        if not pairs:
            return pd.DataFrame(columns=["Tier", "Case", "sta", "rup", "EDP"])

        rows: list[dict[str, Any]] = []
        for (m, sta, rup), nr in pairs:
            tier, case = self.parse_tier_letter(m)

            id_a = int(nr.info.nearest_node_id([(xa, ya, z)], return_distance=False)[0])
            id_b = int(nr.info.nearest_node_id([(xb, yb, z)], return_distance=False)[0])

            ux_a = _disp_series(nr, comp=1, node_id=id_a)
            uy_a = _disp_series(nr, comp=2, node_id=id_a)
            ux_b = _disp_series(nr, comp=1, node_id=id_b)
            uy_b = _disp_series(nr, comp=2, node_id=id_b)

            dfu = pd.concat([ux_a, uy_a, ux_b, uy_b], axis=1, join="inner").astype(float).dropna(how="any")
            if dfu.empty:
                theta = pd.Series(dtype=float)
            else:
                dux = dfu.iloc[:, 2] - dfu.iloc[:, 0]
                duy = dfu.iloc[:, 3] - dfu.iloc[:, 1]
                d_trans = dux * nx + duy * ny
                theta = d_trans / L

            val = _reduce(theta.to_numpy(dtype=float)) if not theta.empty else float("nan")
            val = _apply_op(val)

            rows.append(dict(Tier=int(tier), Case=str(case), sta=str(sta), rup=str(rup), EDP=float(val) if np.isfinite(val) else float("nan")))

        df = pd.DataFrame(rows)
        df["Tier"] = df["Tier"].astype(int)
        df["Case"] = df["Case"].astype("category")
        df["sta"] = df["sta"].astype("category")
        df["rup"] = df["rup"].astype("category")
        return df[["Tier", "Case", "sta", "rup", "EDP"]]

    def torsion_df_long(
        self,
        *,
        z_coord: float,
        node_a_xy: tuple[float, float],
        node_b_xy: tuple[float, float],
        result_name: str = "DISPLACEMENT",
        stage: str | None = None,
        reduce_time: str = "abs_max",
        op: str = "raw",
        eps_log: float = 1e-16,
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        order: Callable[[tuple[str, str, str], Any], Any] | None = None,
    ) -> pd.DataFrame:
        df_wide = self.torsion_df(
            z_coord=z_coord,
            node_a_xy=node_a_xy,
            node_b_xy=node_b_xy,
            result_name=result_name,
            stage=stage,
            reduce_time=reduce_time,
            op=op,
            eps_log=eps_log,
            model=model,
            station=station,
            rupture=rupture,
            order=order,
        )

        return self.wide_to_long(
            df_wide,
            value_col="EDP",
            result_name=result_name,
            component="torsion",
            reduce_time=reduce_time,
            relative_drift=None,
            op=op,
        )

    def base_rocking_df(
        self,
        *,
        z_coord: float,
        node_xy: Sequence[tuple[float, float]],
        result_name: str = "DISPLACEMENT",
        stage: str | None = None,
        metric: Literal["theta", "theta_x", "theta_y"] = "theta",
        reduce_time: str = "abs_max",
        op: str = "raw",
        eps_log: float = 1e-16,
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        order: Callable[[tuple[str, str, str], Any], Any] | None = None,
        colinear_tol: float = 1e-9,  # kept for API stability; rank check is primary
    ) -> pd.DataFrame:
        """
        Rocking from vertical displacement plane fit:
            w(x,y,t) = a(t) + b(t)x + c(t)y
            theta_x(t) =  c(t)
            theta_y(t) = -b(t)
            theta(t)   = sqrt(theta_x^2 + theta_y^2)

        If plan geometry is rank-deficient (colinear), rocking is set to 0.0.
        """
        if reduce_time not in ("abs_max", "max", "min", "rms"):
            raise ValueError("base_rocking_df: reduce_time must be one of: 'abs_max', 'max', 'min', 'rms'.")
        if op not in ("raw", "log"):
            raise ValueError("base_rocking_df: op must be 'raw' or 'log'.")
        if op == "log" and eps_log <= 0:
            raise ValueError("base_rocking_df: eps_log must be > 0 when op='log'.")
        if metric not in ("theta", "theta_x", "theta_y"):
            raise ValueError("base_rocking_df: metric must be 'theta', 'theta_x', or 'theta_y'.")
        if len(node_xy) < 3:
            raise ValueError("base_rocking_df: node_xy must have at least 3 points.")

        pts_xy = [(float(x), float(y)) for (x, y) in node_xy]
        z = float(z_coord)

        def _reduce(y: np.ndarray) -> float:
            y = np.asarray(y, dtype=float)
            y = y[np.isfinite(y)]
            if y.size == 0:
                return float("nan")
            if reduce_time == "abs_max":
                return float(np.nanmax(np.abs(y)))
            if reduce_time == "max":
                return float(np.nanmax(y))
            if reduce_time == "min":
                return float(np.nanmin(y))
            if reduce_time == "rms":
                return float(np.sqrt(np.nanmean(y * y)))
            raise RuntimeError("unreachable")

        def _apply_op(x: float) -> float:
            if not np.isfinite(x):
                return float("nan")
            if op == "raw":
                return float(x)
            # rocking components can be +/-; define log on magnitude
            return float(np.log(np.maximum(abs(float(x)), eps_log)))

        def _select_stage(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
            """
            If results are multi-stage (stage, node_id, step), select:
            - the requested stage if provided and present
            - otherwise the last stage in sorted order (deterministic)
            If results are NOT staged (node_id, step), return unchanged.
            """
            if not isinstance(s, (pd.Series, pd.DataFrame)):
                return s

            idx = s.index
            if not isinstance(idx, pd.MultiIndex):
                return s

            # Only treat as "staged" if it is exactly (stage, node_id, step)
            if idx.nlevels != 3:
                return s

            # stage level is level 0 by convention in your NodalResults.fetch()
            stages = tuple(sorted({str(x) for x in idx.get_level_values(0)}))
            if not stages:
                return s.iloc[0:0]

            if stage is not None:
                st = str(stage)
                if st in stages:
                    return s.xs(st, level=0)
                # if user asked for a missing stage, fail loudly (better than silently wrong)
                raise ValueError(f"Requested stage={st!r} not found. Available stages={stages}")

            # default: last stage
            return s.xs(stages[-1], level=0)


        def _w_series(nr: Any, *, node_id: int) -> pd.Series:
            s = nr.fetch(result_name=result_name, component=3, node_ids=[node_id])
            s = _select_stage(s)
            if not (isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 2):
                return pd.Series(dtype=float)
            w = s.xs(node_id, level=0).sort_index()
            return w.astype(float) if isinstance(w, pd.Series) else pd.Series(dtype=float)

        pairs = self.select(model=model, station=station, rupture=rupture, order=order)
        if not pairs:
            return pd.DataFrame(columns=["Tier", "Case", "sta", "rup", "EDP"])

        # constant plane-fit matrix
        A = np.column_stack(
            [
                np.ones(len(pts_xy)),
                [p[0] for p in pts_xy],
                [p[1] for p in pts_xy],
            ]
        ).astype(float)

        # rank-based degeneracy check (primary)
        geom_colinear = np.linalg.matrix_rank(A) < 3
        # keep tolerance hook (optional): if user wants stricter -> lower tol does nothing here,
        # but we can respect it by treating near-singular as colinear
        if not geom_colinear:
            svals = np.linalg.svd(A, compute_uv=False)
            if svals[-1] <= float(colinear_tol):
                geom_colinear = True

        pinvA = np.linalg.pinv(A)  # (3 x npts)

        rows: list[dict[str, Any]] = []
        for (m, sta, rup), nr in pairs:
            tier, case = self.parse_tier_letter(m)

            if geom_colinear:
                val = _apply_op(0.0)
                rows.append(dict(Tier=int(tier), Case=str(case), sta=str(sta), rup=str(rup), EDP=float(val)))
                continue

            node_ids = [int(nr.info.nearest_node_id([(x, y, z)], return_distance=False)[0]) for (x, y) in pts_xy]

            w_list = [_w_series(nr, node_id=nid) for nid in node_ids]
            if any((not isinstance(s, pd.Series) or s.empty) for s in w_list):
                rows.append(dict(Tier=int(tier), Case=str(case), sta=str(sta), rup=str(rup), EDP=float("nan")))
                continue

            dfw = pd.concat(w_list, axis=1, join="inner").astype(float).dropna(how="any")
            if dfw.empty:
                rows.append(dict(Tier=int(tier), Case=str(case), sta=str(sta), rup=str(rup), EDP=float("nan")))
                continue

            # W: (npts x nt)
            W = dfw.to_numpy(dtype=float).T
            # P: (3 x nt) = pinvA (3 x npts) @ W (npts x nt)
            P = pinvA @ W
            b = P[1, :]  # dw/dx
            c = P[2, :]  # dw/dy

            theta_x = c
            theta_y = -b

            if metric == "theta_x":
                y = theta_x
            elif metric == "theta_y":
                y = theta_y
            else:
                y = np.sqrt(theta_x * theta_x + theta_y * theta_y)

            val = _reduce(y)
            val = _apply_op(val)

            rows.append(
                dict(
                    Tier=int(tier),
                    Case=str(case),
                    sta=str(sta),
                    rup=str(rup),
                    EDP=float(val) if np.isfinite(val) else float("nan"),
                )
            )

        df = pd.DataFrame(rows)
        df["Tier"] = df["Tier"].astype(int)
        df["Case"] = df["Case"].astype("category")
        df["sta"] = df["sta"].astype("category")
        df["rup"] = df["rup"].astype("category")
        return df[["Tier", "Case", "sta", "rup", "EDP"]]

    def base_rocking_df_long(
        self,
        *,
        z_coord: float,
        node_xy: Sequence[tuple[float, float]],
        result_name: str = "DISPLACEMENT",
        stage: str | None = None,
        metric: Literal["theta", "theta_x", "theta_y"] = "theta",
        reduce_time: str = "abs_max",
        op: str = "log",
        eps_log: float = 1e-16,
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        order: Callable[[tuple[str, str, str], Any], Any] | None = None,
        colinear_tol: float = 1e-9,
    ) -> pd.DataFrame:
        df_wide = self.base_rocking_df(
            z_coord=z_coord,
            node_xy=node_xy,
            result_name=result_name,
            stage=stage,
            metric=metric,
            reduce_time=reduce_time,
            op=op,
            eps_log=eps_log,
            model=model,
            station=station,
            rupture=rupture,
            order=order,
            colinear_tol=colinear_tol,
        )

        comp_meta: str = f"base_rocking:{metric}"

        return self.wide_to_long(
            df_wide,
            value_col="EDP",
            result_name=result_name,
            component=comp_meta,
            reduce_time=reduce_time,
            relative_drift=None,
            op=op,
        )
