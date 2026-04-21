from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Literal
)

import fnmatch
import logging
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from .MPCOdf import MPCO_df

logger = logging.getLogger(__name__)

Key = Tuple[str, str, str]  # (model, station, rupture)
GroupKey: TypeAlias = tuple[str, ...]


class _LazyPickle:
    """Proxy that defers loading a pickle file until first attribute access."""

    def __init__(self, path: Path) -> None:
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_obj", None)

    def _load(self):
        obj = object.__getattribute__(self, "_obj")
        if obj is None:
            from ..results.nodal_results_dataclass import NodalResults
            path = object.__getattribute__(self, "_path")
            obj = NodalResults.load_pickle(path)
            object.__setattr__(self, "_obj", obj)
        return obj

    def __getattr__(self, name: str):
        return getattr(self._load(), name)

    def __repr__(self) -> str:
        obj = object.__getattribute__(self, "_obj")
        if obj is None:
            path = object.__getattribute__(self, "_path")
            return f"<_LazyPickle(not loaded) {path.name}>"
        return repr(obj)

class MPCOResults:
    """
    Orchestration wrapper around Dict[(model, station, rupture) -> NodalResults].

    - selection/filtering by key (glob/contains/OR)
    - labels and styles
    - plotting (overlay or per-record) with consistent auto-limits
    - computing tidy metric tables

    Assumes each value (nr) behaves like your NodalResults:
      - nr.time
      - nr.drift(...)
      - nr.interstory_drift_envelope(...)
      - nr.story_pga_envelope(...)
      - nr.roof_torsion(...)
      - nr.base_rocking(...)
      - nr.residual_interstory_drift_profile(...)
      - nr.info.<metric> (analysis_time, size, ...)
    """

    _FNAME_PATTERN = re.compile(
        r"^(?P<model>[^_]+(?:_[^_]+)*)__(?P<station>[^_]+(?:_[^_]+)*)__(?P<rupture>[^_]+(?:_[^_]+)*)\.pkl(\.gz)?$"
    )
    _DIMS_ALL = ("number", "letter", "sta", "rup")
    _PSTAT_RE = re.compile(r"^p(\d{1,2})$")

    def __init__(
        self,
        data: Dict[Key, Any],
        *,
        style: Optional[dict] = None,
        name: Optional[str] = None,
    ) -> None:
        self.data: Dict[Key, Any] = dict(data)
        self.style: Optional[dict] = style
        self.name: Optional[str] = name
        self._station_index_cache: Optional[dict[str, int]] = None

        # Single MPCO_df instance shared by both accessors below. Phase 4.5
        # introduced ``.df`` as the canonical accessor; ``.create_df`` is
        # retained for backward compatibility with existing call sites.
        self.create_df = MPCO_df(self)

    @property
    def df(self) -> "MPCO_df":
        """Canonical accessor for the MPCO-specific DataFrame extractors
        (``drift_df``, ``pga_df``, ``torsion_df``, ``base_rocking_df``,
        ``wide_to_long``, and their ``_long`` / ``_mod`` variants).

        Same instance as ``self.create_df`` — ``.df`` is the preferred
        spelling introduced in Phase 4.5 to match spec §8:
        "Collapse MPCO_df into MPCOResults as a .df accessor." The older
        ``.create_df`` attribute continues to work for back-compat.
        """
        return self.create_df

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
    @classmethod
    def load_dir(
        cls,
        *,
        out_dir: Path,
        style: Optional[dict] = None,
        name: Optional[str] = None,
        lazy: bool = True,
    ) -> "MPCOResults":
        out_dir = Path(out_dir)
        out: Dict[Key, Any] = {}

        from ..results.nodal_results_dataclass import NodalResults

        for p in out_dir.glob("*.pkl*"):
            m = cls._FNAME_PATTERN.match(p.name)
            if not m:
                continue
            key: Key = (m.group("model"), m.group("station"), m.group("rupture"))
            if key in out:
                raise ValueError(f"Duplicate key {key} from file {p.name!r}")
            if lazy:
                out[key] = _LazyPickle(p)
            else:
                out[key] = NodalResults.load_pickle(p)

        return cls(out, style=style, name=name)

    # ------------------------------------------------------------------
    # Collection protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[Key]:
        return iter(self.data)

    def __getitem__(self, key: Key) -> Any:
        return self.data[key]

    def keys(self) -> Iterable[Key]:
        return self.data.keys()

    def values(self) -> Iterable[Any]:
        return self.data.values()

    def items(self) -> Iterable[tuple[Key, Any]]:
        return self.data.items()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_matcher(x: Any) -> Callable[[str], bool] | None:
        if x is None:
            return None

        if isinstance(x, (str, bytes)):
            s = str(x).strip()
            if s.lower() == "all":
                return None
            s_low = s.lower()

            if s_low.startswith("*") and s_low.endswith("*"):
                key = s_low.strip("*")
                return lambda v: key in str(v).lower()

            if s_low.startswith("*"):
                key = s_low[1:]
                return lambda v: str(v).lower().endswith(key)

            if s_low.endswith("*"):
                key = s_low[:-1]
                return lambda v: str(v).lower().startswith(key)

            return lambda v: str(v).lower() == s_low

        if isinstance(x, Iterable):
            ms = [MPCOResults._make_matcher(v) for v in x]
            if any(m is None for m in ms):
                return None
            ms2 = [m for m in ms if m is not None]
            if not ms2:
                return None
            return lambda v: any(m(v) for m in ms2)

        return lambda v: v == x

    @staticmethod
    def _first_glob_match(value: str, glob_map: Mapping[str, Any]) -> Any | None:
        for pat, out in glob_map.items():  # order matters
            if fnmatch.fnmatchcase(str(value), pat):
                return out
        return None

    @staticmethod
    def _align_xy(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray, int, int]:
        x2 = np.asarray(x)
        y2 = np.asarray(y)
        nx, ny = x2.shape[0], y2.shape[0]
        n = min(nx, ny)
        if n == 0:
            return x2[:0], y2[:0], nx, ny
        return x2[:n], y2[:n], nx - n, ny - n

    @staticmethod
    def _running_envelope(y: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=float)
        if mode == "abs":
            env = np.maximum.accumulate(np.abs(y))
            return env, -env
        if mode == "signed":
            upper = np.maximum.accumulate(y)
            lower = np.minimum.accumulate(y)
            return upper, lower
        raise ValueError("mode must be 'abs' or 'signed'.")

    @staticmethod
    def _step_path(z_lower: np.ndarray, z_upper: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(z_upper)
        zL = z_lower[order]
        zU = z_upper[order]
        d = d[order]

        floors = zU
        n = len(floors)

        xs = [d[0]]
        ys = [floors[0]]

        for i in range(n - 1):
            xs.append(d[i])
            ys.append(floors[i + 1])
            xs.append(d[i + 1])
            ys.append(floors[i + 1])

        return np.array(xs), np.array(ys)

    @staticmethod
    def parse_tier_letter(model_label: str) -> tuple[int, str]:
        s = str(model_label).upper()
        m = re.search(r"(\d+)([A-Z])", s)
        if not m:
            raise ValueError(f"Could not parse tier/letter from model label: {model_label!r}")
        tier = int(m.group(1))
        letter = m.group(2)
        return tier, letter

    # ------------------------------------------------------------------
    # Grouping core (reusable)
    # ------------------------------------------------------------------
    def _normalize_grouping_spec(
        self,
        *,
        group_by: tuple[str, ...] | None,
        color_by: str | None,
        stat: str | None,
    ) -> tuple[tuple[str, ...] | None, str | None, str | None, bool]:
        """
        Returns: (group_by, color_by, stat, color_by_group)

        - stat: None | "mean" | "pXX" (lower-cased)
        - if stat is not None and group_by is None: defaults group_by=("number","letter")
        - if color_by is None and group_by is not None: defaults color_by="__group__" (group identity)
        """
        # normalize stat
        if stat is not None:
            s = str(stat).lower().strip()
            if s == "mean":
                stat = "mean"
            else:
                m = self._PSTAT_RE.match(s)
                if not m:
                    raise ValueError("stat must be None, 'mean', or 'pXX' (e.g. 'p84').")
                p = int(m.group(1))
                if not (1 <= p <= 99):
                    raise ValueError("pXX must be between p1 and p99.")
                stat = s

        # default group_by if asked for stats
        if stat is not None and group_by is None:
            group_by = ("number", "letter")

        # validate group_by
        if group_by is not None:
            gb = tuple(str(d).lower().strip() for d in group_by)
            bad = [d for d in gb if d not in self._DIMS_ALL]
            if bad:
                raise ValueError(f"group_by contains invalid dims {bad}. Valid: {self._DIMS_ALL}.")
            if len(set(gb)) != len(gb):
                raise ValueError(f"group_by has duplicates: {gb}")
            group_by = gb

        # normalize color_by
        cb = None if color_by is None else str(color_by).lower().strip()
        if cb is not None and cb not in self._DIMS_ALL and cb != "__group__":
            raise ValueError(f"color_by must be one of {self._DIMS_ALL}, '__group__', or None.")

        color_by_group = False
        if cb is None and group_by is not None:
            cb = "__group__"
            color_by_group = True
        elif cb == "__group__":
            color_by_group = True

        return group_by, cb, stat, color_by_group

    def _tag_from_key(self, k: Key) -> dict[str, str]:
        m, sta, rup = k
        mm = str(m).strip()
        mnum = re.search(r"(\d+)", mm)
        mlet = re.search(r"([A-Za-z])\s*$", mm)
        return {
            "number": mnum.group(1) if mnum else "",
            "letter": mlet.group(1).upper() if mlet else "",
            "sta": str(sta),
            "rup": str(rup),
        }

    def _group_key(self, k: Key, group_by: tuple[str, ...] | None) -> GroupKey:
        if group_by is None:
            return (f"{k[0]}|{k[1]}|{k[2]}",)
        tags = self._tag_from_key(k)
        return tuple(tags[d] for d in group_by)

    def _color_tag(self, *, gk: GroupKey, k: Key, color_by: str | None, color_by_group: bool) -> str:
        if color_by is None:
            return ""
        if color_by_group:
            return "|".join(gk)
        return self._tag_from_key(k)[color_by]

    def _build_groups(self, pairs: list[tuple[Key, Any]], group_by: tuple[str, ...] | None) -> dict[GroupKey, list[tuple[Key, Any]]]:
        groups: dict[GroupKey, list[tuple[Key, Any]]] = {}
        for k, nr in pairs:
            gk = self._group_key(k, group_by)
            groups.setdefault(gk, []).append((k, nr))
        return groups

    @staticmethod
    def _reduce_stack(A: np.ndarray, stat: str) -> np.ndarray:
        if stat == "mean":
            return np.nanmean(A, axis=0)
        p = float(stat[1:])
        return np.nanpercentile(A, p, axis=0)

    @staticmethod
    def _reduce_stack_signed(A: np.ndarray, stat: str, signed: bool) -> np.ndarray:
        if stat == "mean":
            return np.nanmean(A, axis=0)

        p = float(stat[1:])

        if not signed:
            return np.nanpercentile(np.abs(A), p, axis=0)

        mu = np.nanmean(A, axis=0)
        out = np.empty_like(mu)

        for j in range(mu.size):
            q = p if mu[j] >= 0.0 else (100.0 - p)
            out[j] = float(np.nanpercentile(A[:, j], q))
        return out

    @staticmethod
    def _mask_last_seconds(t: np.ndarray, tail_s: float | None) -> np.ndarray | slice:
        if tail_s is None:
            return slice(None)
        t = np.asarray(t, dtype=float)
        if t.size == 0:
            return slice(0, 0)
        t0 = float(t[-1]) - float(tail_s)
        return t >= t0

    @staticmethod
    def _legend_below(
        fig: plt.Figure,
        ax: plt.Axes,
        *,
        fontsize: float,
        ncol: int | None,
        frameon: bool,
        bottom: float = 0.22,
        y: float = -0.14,
    ) -> None:
        handles, labels = ax.get_legend_handles_labels()
        H, L = [], []
        for h, l in zip(handles, labels):
            if l and l != "_nolegend_":
                H.append(h)
                L.append(l)
        if not H:
            return

        n = len(H)
        ncol2 = ncol or min(6, n)
        fig.subplots_adjust(bottom=bottom)
        ax.legend(
            H,
            L,
            loc="upper center",
            bbox_to_anchor=(0.5, y),
            ncol=ncol2,
            frameon=frameon,
            fontsize=fontsize,
            handlelength=2.2,
            columnspacing=1.2,
            borderaxespad=0.0,
        )

    # ------------------------------------------------------------------
    # Internal primitives
    # ------------------------------------------------------------------
    def _station_index(self) -> dict[str, int]:
        if self._station_index_cache is None:
            stations_sorted = sorted({k[1] for k in self.data.keys()})
            self._station_index_cache = {s: i for i, s in enumerate(stations_sorted)}
        return self._station_index_cache

    def _style_for_key(self, key: Key) -> dict:
        if not self.style:
            return {}

        model, station, _ = key
        defaults = self.style.get("defaults", {})

        out = {
            "color": defaults.get("color", "black"),
            "linestyle": defaults.get("linestyle", "-"),
            "marker": defaults.get("marker", "o"),
            "linewidth": defaults.get("linewidth", 1.0),
            "markersize": defaults.get("markersize", 3),
            "alpha": defaults.get("alpha", 1.0),
        }

        sta = self.style.get("station", {})
        exp = sta.get("explicit_map", {}) or {}
        cyc = sta.get("cycle", []) or []
        if station in exp:
            out["color"] = exp[station]
        elif cyc:
            idx = self._station_index().get(station, 0)
            out["color"] = cyc[idx % len(cyc)]

        mnum = self.style.get("model_number", {}).get("map", {}) or {}
        mk = self._first_glob_match(model, mnum)
        if mk is not None:
            out["marker"] = mk

        mlet = self.style.get("model_letter", {}).get("map", {}) or {}
        ls = self._first_glob_match(model, mlet)
        if ls is not None:
            out["linestyle"] = ls

        return out

    def _label_for(self, key: Key, nr: Any) -> str:
        return getattr(nr, "name", None) or f"{key[0]} | {key[1]} | {key[2]}"

    def select(
        self,
        *,
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        order: Callable[[Key, Any], Any] | None = None,
    ) -> list[tuple[Key, Any]]:
        mm = self._make_matcher(model)
        sm = self._make_matcher(station)
        rm = self._make_matcher(rupture)

        pairs: list[tuple[Key, Any]] = []
        for k, obj in self.data.items():
            m, s, r = k
            if mm and not mm(m):
                continue
            if sm and not sm(s):
                continue
            if rm and not rm(r):
                continue
            pairs.append((k, obj))

        if not pairs:
            return []

        if order is None:
            pairs.sort(key=lambda kv: kv[0])
        else:
            pairs.sort(key=lambda kv: order(kv[0], kv[1]))

        return pairs

    @staticmethod
    def _bounds_update(bounds: dict[str, float], *, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
        bounds["xmin"] = min(bounds["xmin"], xmin)
        bounds["xmax"] = max(bounds["xmax"], xmax)
        bounds["ymin"] = min(bounds["ymin"], ymin)
        bounds["ymax"] = max(bounds["ymax"], ymax)

    @staticmethod
    def _apply_limits(
        ax: plt.Axes,
        *,
        bounds: dict[str, float] | None,
        xlim: tuple[float, float] | None,
        ylim: tuple[float, float] | None,
        sym_x: bool,
        sym_y: bool,
    ) -> None:
        if bounds is None:
            return

        if xlim is None:
            if sym_x:
                a = max(abs(bounds["xmin"]), abs(bounds["xmax"]))
                ax.set_xlim(-a, a)
            else:
                ax.set_xlim(bounds["xmin"], bounds["xmax"])
        else:
            ax.set_xlim(*xlim)

        if ylim is None:
            if sym_y:
                a = max(abs(bounds["ymin"]), abs(bounds["ymax"]))
                ax.set_ylim(-a, a)
            else:
                ax.set_ylim(bounds["ymin"], bounds["ymax"])
        else:
            ax.set_ylim(*ylim)

    def _plot_overlay_or_facets(
        self,
        *,
        pairs: list[tuple[Key, Any]],
        plot_one: Callable[[plt.Axes, Key, Any], tuple[float, float, float, float] | None],
        overlay: bool,
        figsize_overlay: tuple[float, float],
        figsize_single: tuple[float, float],
        title: str,
        xlabel: str,
        ylabel: str,
        xlim: tuple[float, float] | None,
        ylim: tuple[float, float] | None,
        sym_x: bool,
        sym_y: bool,
        vline0: bool = False,
        legend: bool = True,
        grid: bool = True,
    ):
        """
        plot_one returns bounds (xmin, xmax, ymin, ymax) or None if nothing plotted.

        If xlim/ylim is None, uses bounds to set limits:
          - symmetric if sym_x/sym_y True
          - minmax otherwise
        """
        if overlay:
            fig, ax = plt.subplots(figsize=figsize_overlay)
            if vline0:
                ax.axvline(0.0, linewidth=1)

            bounds = {"xmin": np.inf, "xmax": -np.inf, "ymin": np.inf, "ymax": -np.inf}
            any_plotted = False

            for k, nr in pairs:
                b = plot_one(ax, k, nr)
                if b is None:
                    continue
                any_plotted = True
                self._bounds_update(bounds, xmin=b[0], xmax=b[1], ymin=b[2], ymax=b[3])

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)

            if any_plotted:
                self._apply_limits(ax, bounds=bounds, xlim=xlim, ylim=ylim, sym_x=sym_x, sym_y=sym_y)

            if grid:
                ax.grid(True, alpha=0.35)
            if legend:
                ax.legend()

            plt.tight_layout()
            return fig, ax

        figs = []
        for k, nr in pairs:
            fig, ax = plt.subplots(figsize=figsize_single)
            if vline0:
                ax.axvline(0.0, linewidth=1)

            b = plot_one(ax, k, nr)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            label = self._label_for(k, nr)
            ax.set_title(f"{title} — {label}")

            if b is not None:
                bounds = {"xmin": b[0], "xmax": b[1], "ymin": b[2], "ymax": b[3]}
                self._apply_limits(ax, bounds=bounds, xlim=xlim, ylim=ylim, sym_x=sym_x, sym_y=sym_y)

            if grid:
                ax.grid(True, alpha=0.35)
            if legend:
                ax.legend()

            plt.tight_layout()
            figs.append((fig, ax))

        return figs

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------
    def plot_drift(
        self,
        *,
        top: tuple[float, float, float],
        bottom: tuple[float, float, float],
        component: int = 1,
        relative_drift: bool = True,  # True -> divide by height (drift ratio). False -> delta_u (no /dz)
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        overlay: bool = True,
        figsize: tuple[float, float] = (10, 6),
        group_by_color: str | None = None,   # kept for compatibility; style is handled by _style_for_key
        linewidth: float = 1.00,
        warn_mismatch: bool = True,
        running_envelope: str | None = None,  # None | "abs" | "signed"
        envelope_alpha: float = 0.35,
        envelope_linewidth: float | None = None,
        envelope_only: bool = False,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        legend_fontsize: float = 7,
        legend_ncol: int | None = None,
        legend_frameon: bool = False,
    ):
        """
        Plot drift histories over MPCO collection.

        Parameters
        ----------
        relative_drift
            - True:  drift(t) = (u_top - u_bottom) / (z_top - z_bottom)
                    (uses nr.drift)
            - False: delta_u(t) = (u_top - u_bottom) (no division by height)
                    (computed via nr.fetch + subtraction)
        """
        valid_groups = {None, "sta", "rup", "letter", "number"}
        if group_by_color not in valid_groups:
            raise ValueError(f"group_by_color must be one of {valid_groups}")

        pairs = self.select(model=model, station=station, rupture=rupture)
        if not pairs:
            raise ValueError("No matching results for the given selection.")
        pairs = sorted(pairs, key=lambda kv: kv[0])

        # ------------------------------------------------------------
        # Helper: compute delta_u (no /dz) from top/bottom coordinates
        # ------------------------------------------------------------
        def _delta_u_series(nr: Any) -> "pd.Series":
            top_id = int(nr.info.nearest_node_id([top], return_distance=False)[0])
            bot_id = int(nr.info.nearest_node_id([bottom], return_distance=False)[0])

            s = nr.fetch(result_name="DISPLACEMENT", component=component, node_ids=[top_id, bot_id])

            # If multi-stage, pick last stage deterministically (MPCO.plot_drift has no stage arg)
            if isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 3:
                stages = tuple(sorted({str(x) for x in s.index.get_level_values(0)}))
                if not stages:
                    raise ValueError("plot_drift(relative_drift=False): no stages found.")
                s = s.xs(stages[-1], level=0)

            if not (isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 2):
                raise ValueError("plot_drift(relative_drift=False): expected index (node_id, step).")

            u_top = s.xs(top_id, level=0).sort_index()
            u_bot = s.xs(bot_id, level=0).sort_index()
            u_top, u_bot = u_top.align(u_bot, join="inner")

            du = u_top - u_bot
            du.name = f"delta_u(component={component})"
            return du

        # ------------------------------------------------------------
        # Plot per run
        # ------------------------------------------------------------
        def plot_one(ax: plt.Axes, k: Key, nr: Any):
            if relative_drift:
                series = nr.drift(top=top, bottom=bottom, component=component)
            else:
                series = _delta_u_series(nr)

            y = np.asarray(series.values, float)
            t = np.asarray(nr.time, float)

            t2, y2, t_trim, y_trim = self._align_xy(t, y)
            if warn_mismatch and (t_trim or y_trim):
                logger.info(
                    "[plot_drift] mismatch %s: time=%d y=%d -> %d",
                    k, len(t), len(y), len(t2),
                )

            if t2.size == 0:
                return None

            label = self._label_for(k, nr)

            kw = self._style_for_key(k)  # <-- style owns colors; no palette logic here
            kw["linewidth"] = linewidth

            if running_envelope is None:
                ax.plot(t2, y2, label=label, **kw)
                return (float(t2.min()), float(t2.max()), float(y2.min()), float(y2.max()))

            up, lo = self._running_envelope(y2, running_envelope)
            env_kw = dict(kw)
            env_kw.pop("label", None)
            env_kw["alpha"] = envelope_alpha
            env_kw["linewidth"] = envelope_linewidth or linewidth

            if envelope_only:
                ax.plot(t2, up, label=label, **env_kw)
                ax.plot(t2, lo, **env_kw)
            else:
                ax.plot(t2, y2, label=label, **kw)
                ax.plot(t2, up, **env_kw)
                ax.plot(t2, lo, **env_kw)

            yy = np.r_[up, lo]
            return (float(t2.min()), float(t2.max()), float(yy.min()), float(yy.max()))

        title = "Drift histories" if relative_drift else "Relative displacement histories (Δu)"
        if running_envelope:
            title += f" + running envelope ({running_envelope})"

        out = self._plot_overlay_or_facets(
            pairs=pairs,
            plot_one=plot_one,
            overlay=overlay,
            figsize_overlay=figsize,
            figsize_single=(7, 4),
            title=title,
            xlabel="Time (s)",
            ylabel="Drift" if relative_drift else "Δu (disp. units)",
            xlim=xlim,
            ylim=ylim,
            sym_x=False,
            sym_y=True,
            vline0=False,
            legend=False,
            grid=True,
        )

        def _legend(fig: plt.Figure, ax: plt.Axes):
            self._legend_below(fig, ax, fontsize=legend_fontsize, ncol=legend_ncol, frameon=legend_frameon)

        if overlay:
            fig, ax = out
            _legend(fig, ax)
            plt.tight_layout()
            return fig, ax

        figs = out
        for fig, ax in figs:
            _legend(fig, ax)
            plt.tight_layout()
        return figs

    def plot_drift_envelope(
        self,
        *,
        component: int = 1,
        selection_set_name: str = "CenterPoints",
        selection_set_id: Any = None,
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        group_by: tuple[str, ...] | None = None,     # ("number","letter"), etc.
        reduce_by: tuple[str, ...] | None = None,    # kept for API symmetry; not used explicitly
        stat: str | None = None,                     # None | "mean" | "pXX"
        color_by: str | None = None,                 # None | "number" | "letter" | "sta" | "rup"
        show_individual: bool = True,
        individual_alpha: float = 0.15,
        linewidth: float = 1.2,
        group_linewidth: float = 2.6,
        show_min: bool = True,
        show_max: bool = True,
        overlay: bool = True,
        figsize: tuple[float, float] = (15, 10),
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        legend_fontsize: float = 9,
        legend_ncol: int | None = None,
        legend_frameon: bool = False,
    ):
        group_by, color_by, stat, color_by_group = self._normalize_grouping_spec(group_by=group_by, color_by=color_by, stat=stat)

        pairs = self.select(model=model, station=station, rupture=rupture, order=None)
        if not pairs:
            raise ValueError("No matching results.")
        pairs = sorted(pairs, key=lambda kv: kv[0])

        groups = self._build_groups(pairs, group_by)

        # Color cache: tag -> matplotlib color
        tag_color: dict[str, Any] = {}

        def _get_or_set_color(tag: str, line) -> Any:
            if not tag:
                return None
            if tag not in tag_color:
                tag_color[tag] = line.get_color()
            return tag_color[tag]

        def _reduce_max(A: np.ndarray) -> np.ndarray:
            if stat is None:
                raise RuntimeError("stat=None")
            return self._reduce_stack(A, stat)

        def _reduce_min(A: np.ndarray) -> np.ndarray:
            if stat is None:
                raise RuntimeError("stat=None")
            if stat == "mean":
                return np.nanmean(A, axis=0)
            p = float(stat[1:])
            return np.nanpercentile(A, 100.0 - p, axis=0)

        def _plot(ax: plt.Axes) -> tuple[float, float, float, float] | None:
            xmin, xmax = np.inf, -np.inf
            ymin, ymax = np.inf, -np.inf

            for gk in sorted(groups.keys()):
                items = groups[gk]
                if not items:
                    continue

                # env base grid from first run
                env0 = items[0][1].interstory_drift_envelope(
                    component=component,
                    selection_set_name=selection_set_name,
                    selection_set_id=selection_set_id,
                )
                zL = env0["z_lower"].to_numpy(float)
                zU = env0["z_upper"].to_numpy(float)

                ymin = min(ymin, float(np.nanmin(zL)))
                ymax = max(ymax, float(np.nanmax(zU)))

                # individuals
                if stat is None or show_individual:
                    for k, nr in items:
                        env = nr.interstory_drift_envelope(
                            component=component,
                            selection_set_name=selection_set_name,
                            selection_set_id=selection_set_id,
                        )
                        dmax = env["max_drift"].to_numpy(float)
                        dmin = env["min_drift"].to_numpy(float)

                        alpha = individual_alpha if stat is not None else 1.0

                        ct = self._color_tag(gk=gk, k=k, color_by=color_by, color_by_group=color_by_group)
                        if show_max:
                            x, y = self._step_path(zL, zU, dmax)
                            if color_by is None:
                                line, = ax.plot(x, y, alpha=alpha, linewidth=linewidth, label="_nolegend_")
                                _get_or_set_color(ct, line)
                            else:
                                col = tag_color.get(ct)
                                if col is None:
                                    line, = ax.plot(x, y, alpha=alpha, linewidth=linewidth, label="_nolegend_")
                                    _get_or_set_color(ct, line)
                                else:
                                    ax.plot(x, y, color=col, alpha=alpha, linewidth=linewidth, label="_nolegend_")
                            xmin, xmax = min(xmin, float(np.nanmin(x))), max(xmax, float(np.nanmax(x)))

                        if show_min:
                            x, y = self._step_path(zL, zU, dmin)
                            if color_by is None:
                                line, = ax.plot(x, y, alpha=alpha, linewidth=linewidth, label="_nolegend_")
                                _get_or_set_color(ct, line)
                            else:
                                col = tag_color.get(ct)
                                if col is None:
                                    line, = ax.plot(x, y, alpha=alpha, linewidth=linewidth, label="_nolegend_")
                                    _get_or_set_color(ct, line)
                                else:
                                    ax.plot(x, y, color=col, alpha=alpha, linewidth=linewidth, label="_nolegend_")
                            xmin, xmax = min(xmin, float(np.nanmin(x))), max(xmax, float(np.nanmax(x)))

                # group stat
                if stat is not None:
                    DMAX, DMIN = [], []
                    for _, nr in items:
                        env = nr.interstory_drift_envelope(
                            component=component,
                            selection_set_name=selection_set_name,
                            selection_set_id=selection_set_id,
                        )
                        DMAX.append(env["max_drift"].to_numpy(float))
                        DMIN.append(env["min_drift"].to_numpy(float))

                    dmax_g = _reduce_max(np.vstack(DMAX))
                    dmin_g = _reduce_min(np.vstack(DMIN))

                    label = " | ".join(gk) + f" ({stat})"
                    ct0 = self._color_tag(gk=gk, k=items[0][0], color_by=color_by, color_by_group=color_by_group)

                    if show_max:
                        x, y = self._step_path(zL, zU, dmax_g)
                        if color_by is None:
                            line, = ax.plot(x, y, linewidth=group_linewidth, label=label)
                            _get_or_set_color(ct0, line)
                        else:
                            col = tag_color.get(ct0)
                            if col is None:
                                line, = ax.plot(x, y, linewidth=group_linewidth, label=label)
                                _get_or_set_color(ct0, line)
                            else:
                                ax.plot(x, y, color=col, linewidth=group_linewidth, label=label)
                        xmin, xmax = min(xmin, float(np.nanmin(x))), max(xmax, float(np.nanmax(x)))

                    if show_min:
                        x, y = self._step_path(zL, zU, dmin_g)
                        if color_by is None:
                            line, = ax.plot(x, y, linewidth=group_linewidth, label="_nolegend_")
                            _get_or_set_color(ct0, line)
                        else:
                            col = tag_color.get(ct0)
                            if col is None:
                                line, = ax.plot(x, y, linewidth=group_linewidth, label="_nolegend_")
                                _get_or_set_color(ct0, line)
                            else:
                                ax.plot(x, y, color=col, linewidth=group_linewidth, label="_nolegend_")
                        xmin, xmax = min(xmin, float(np.nanmin(x))), max(xmax, float(np.nanmax(x)))

            if not np.isfinite(xmin):
                return None
            return float(xmin), float(xmax), float(ymin), float(ymax)

        fig, ax = plt.subplots(figsize=figsize)
        ax.axvline(0.0, lw=1)
        bounds = _plot(ax)

        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            if xlim is None:
                a = max(abs(xmin), abs(xmax))
                ax.set_xlim(-a, a)
            else:
                ax.set_xlim(*xlim)

            if ylim is None:
                ax.set_ylim(ymin, ymax)
            else:
                ax.set_ylim(*ylim)

        ax.set_xlabel("Interstory drift")
        ax.set_ylabel("z")
        ax.set_title("Interstory drift envelope (tied)")
        ax.grid(True, alpha=0.35)

        self._legend_below(fig, ax, fontsize=legend_fontsize, ncol=legend_ncol, frameon=legend_frameon)
        plt.tight_layout()
        return fig, ax

    def plot_residual_drift_profile(
        self,
        *,
        component: int = 1,
        selection_set_name: str = "CenterPoints",
        selection_set_id: Any = None,
        tail_seconds: float | None = 10.0,
        agg: str = "mean",          # "mean" | "median"
        signed: bool = True,
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        linewidth: float = 1.0,
        overlay: bool = True,
        figsize: tuple[float, float] = (15, 10),
        group_by: tuple[str, ...] | None = None,     # e.g. ("number","letter")
        reduce_by: tuple[str, ...] | None = None,    # kept for API symmetry; not used explicitly
        stat: str | None = None,                     # None | "mean" | "pXX"
        color_by: str | None = None,                 # None | "number" | "letter" | "sta" | "rup"
        show_individual: bool = True,
        individual_alpha: float = 0.15,
        group_linewidth: float = 2.6,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        legend_fontsize: float = 7,
        legend_ncol: int | None = None,
        legend_frameon: bool = False,
    ):
        if agg not in ("mean", "median"):
            raise ValueError("agg must be 'mean' or 'median'.")

        group_by, color_by, stat, color_by_group = self._normalize_grouping_spec(group_by=group_by, color_by=color_by, stat=stat)

        pairs = self.select(model=model, station=station, rupture=rupture, order=None)
        if not pairs:
            raise ValueError("No matching results for the given selection.")
        pairs = sorted(pairs, key=lambda kv: kv[0])

        groups = self._build_groups(pairs, group_by)

        tag_color: dict[str, Any] = {}

        def _reduce_window(x: np.ndarray) -> float:
            if x.size == 0:
                return float("nan")
            if agg == "mean":
                return float(np.nanmean(x))
            return float(np.nanmedian(x))

        def _residual_profile_for_run(k: Key, nr: Any) -> pd.DataFrame:
            prof = nr.residual_interstory_drift_profile(
                component=component,
                selection_set_name=selection_set_name,
                selection_set_id=selection_set_id,
                tail=1,          # ignored here
                agg="mean",      # ignored here
                signed=True,
            )

            z_lower = prof["z_lower"].to_numpy(dtype=float)
            z_upper = prof["z_upper"].to_numpy(dtype=float)
            lower_nodes = prof["lower_node"].to_numpy(dtype=int)
            upper_nodes = prof["upper_node"].to_numpy(dtype=int)
            dz = prof["dz"].to_numpy(dtype=float)

            t = np.asarray(nr.time, dtype=float)
            msk = self._mask_last_seconds(t, tail_seconds)

            res_vals: list[float] = []
            for n_lo, n_up, dz_i in zip(lower_nodes, upper_nodes, dz):
                dr = nr.drift(
                    top=int(n_up),
                    bottom=int(n_lo),
                    component=component,
                    result_name="DISPLACEMENT",
                    signed=True,
                    reduce="series",
                )
                y = np.asarray(dr.to_numpy(dtype=float), dtype=float)
                t2, y2, *_ = self._align_xy(t, y)

                if isinstance(msk, slice):
                    yw = y2[msk]
                else:
                    msk2 = np.asarray(msk, dtype=bool)[: t2.size]
                    yw = y2[msk2]

                v = _reduce_window(yw)
                if not signed:
                    v = abs(v)
                res_vals.append(v)

            return pd.DataFrame(
                {
                    "z_lower": z_lower,
                    "z_upper": z_upper,
                    "residual_drift": np.asarray(res_vals, dtype=float),
                }
            )

        def _plot_axes(ax: plt.Axes, *, only_group: GroupKey | None = None) -> tuple[float, float, float, float] | None:
            xmin, xmax = np.inf, -np.inf
            ymin, ymax = np.inf, -np.inf

            for gk in sorted(groups.keys()):
                if only_group is not None and gk != only_group:
                    continue

                items = groups[gk]
                if not items:
                    continue

                run_keys: list[Key] = []
                run_dfs: list[pd.DataFrame] = []

                for k, nr in items:
                    df = _residual_profile_for_run(k, nr)
                    run_keys.append(k)
                    run_dfs.append(df)

                if not run_dfs:
                    continue

                base = run_dfs[0]
                zL = base["z_lower"].to_numpy(dtype=float)
                zU = base["z_upper"].to_numpy(dtype=float)

                ymin = min(ymin, float(np.nanmin(zL)))
                ymax = max(ymax, float(np.nanmax(zU)))

                # individuals
                if stat is None or show_individual:
                    for k, df in zip(run_keys, run_dfs):
                        d = df["residual_drift"].to_numpy(dtype=float)
                        x, y = self._step_path(zL, zU, d)

                        ct = self._color_tag(gk=gk, k=k, color_by=color_by, color_by_group=color_by_group)
                        alpha = individual_alpha if stat is not None else 1.0

                        if color_by is None:
                            label = self._label_for(k, self.data[k]) if stat is None else "_nolegend_"
                            line, = ax.plot(x, y, linewidth=linewidth, alpha=alpha, label=label)
                            if ct and ct not in tag_color:
                                tag_color[ct] = line.get_color()
                        else:
                            col = tag_color.get(ct)
                            if col is None:
                                line, = ax.plot(x, y, linewidth=linewidth, alpha=alpha, label="_nolegend_")
                                if ct:
                                    tag_color[ct] = line.get_color()
                            else:
                                ax.plot(x, y, color=col, linewidth=linewidth, alpha=alpha, label="_nolegend_")

                        xmin = min(xmin, float(np.nanmin(x)))
                        xmax = max(xmax, float(np.nanmax(x)))

                # group stat
                if stat is not None:
                    A = np.vstack([df["residual_drift"].to_numpy(dtype=float) for df in run_dfs])
                    d_g = self._reduce_stack_signed(A, stat=stat, signed=signed)

                    ct0 = self._color_tag(gk=gk, k=run_keys[0], color_by=color_by, color_by_group=color_by_group)
                    label = " | ".join(gk) + f" ({stat})"
                    xg, yg = self._step_path(zL, zU, d_g)

                    if color_by is None:
                        line, = ax.plot(xg, yg, linewidth=group_linewidth, label=label)
                        if ct0 and ct0 not in tag_color:
                            tag_color[ct0] = line.get_color()
                    else:
                        col = tag_color.get(ct0)
                        if col is None:
                            line, = ax.plot(xg, yg, linewidth=group_linewidth, label=label)
                            if ct0:
                                tag_color[ct0] = line.get_color()
                        else:
                            ax.plot(xg, yg, color=col, linewidth=group_linewidth, label=label)

                    xmin = min(xmin, float(np.nanmin(xg)))
                    xmax = max(xmax, float(np.nanmax(xg)))

            if not np.isfinite(xmin):
                return None
            return float(xmin), float(xmax), float(ymin), float(ymax)

        def _apply_limits(ax: plt.Axes, bounds: tuple[float, float, float, float] | None):
            if bounds is None:
                return
            xmin, xmax, ymin, ymax = bounds

            if xlim is None:
                a = max(abs(xmin), abs(xmax))
                ax.set_xlim(-a, a)
            else:
                ax.set_xlim(*xlim)

            if ylim is None:
                ax.set_ylim(ymin, ymax)
            else:
                ax.set_ylim(*ylim)

        if overlay:
            fig, ax = plt.subplots(figsize=figsize)
            ax.axvline(0.0, linewidth=1)
            bounds = _plot_axes(ax)
            ax.set_xlabel("Residual interstory drift")
            ax.set_ylabel("z")
            ttl = "Residual interstory drift profile (tied)"
            if tail_seconds is not None:
                ttl += f" — last {tail_seconds:g}s"
            ax.set_title(ttl)
            ax.grid(True, alpha=0.35)
            _apply_limits(ax, bounds)
            self._legend_below(fig, ax, fontsize=legend_fontsize, ncol=legend_ncol, frameon=legend_frameon)
            plt.tight_layout()
            return fig, ax

        figs = []
        for gk in sorted(groups.keys()):
            fig, ax = plt.subplots(figsize=figsize)
            ax.axvline(0.0, linewidth=1)
            bounds = _plot_axes(ax, only_group=gk)
            ax.set_xlabel("Residual interstory drift")
            ax.set_ylabel("z")
            ax.set_title(f"Residual interstory drift profile (tied) — {' | '.join(gk)}")
            ax.grid(True, alpha=0.35)
            _apply_limits(ax, bounds)
            self._legend_below(fig, ax, fontsize=legend_fontsize, ncol=legend_ncol, frameon=legend_frameon)
            plt.tight_layout()
            figs.append((fig, ax))
        return figs

    def plot_pga_envelope(
        self,
        *,
        component: int = 1,
        selection_set_name: str = "CenterPoints",
        in_g: bool = True,
        g_value: float = 9810.0,
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        overlay: bool = True,
        figsize: tuple[float, float] = (6, 4),
        title: str | None = None,
        group_by: tuple[str, ...] | None = None,
        reduce_by: tuple[str, ...] | None = None,    # kept for API symmetry; not used explicitly
        stat: str | None = None,                     # None | "mean" | "pXX"
        color_by: str | None = None,                 # None | "number" | "letter" | "sta" | "rup"
        show_individual: bool = True,
        individual_alpha: float = 0.15,
        linewidth: float = 1.0,
        group_linewidth: float = 2.0,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        legend_fontsize: float = 7,
        legend_ncol: int | None = None,
        legend_frameon: bool = False,
    ):
        group_by, color_by, stat, color_by_group = self._normalize_grouping_spec(group_by=group_by, color_by=color_by, stat=stat)

        pairs = self.select(model=model, station=station, rupture=rupture, order=None)
        if not pairs:
            raise ValueError("No matching results for the given selection.")
        pairs = sorted(pairs, key=lambda kv: kv[0])

        groups = self._build_groups(pairs, group_by)

        tag_color: dict[str, Any] = {}

        def _pga_series_for_run(nr: Any) -> pd.Series:
            df = nr.story_pga_envelope(
                component=component,
                selection_set_name=selection_set_name,
                g_value=g_value,
                to_g=in_g,
            )
            s = pd.Series(df["pga"].to_numpy(dtype=float), index=df.index.to_numpy(dtype=float))
            s.name = "pga"
            return s

        def _plot_axes(ax: plt.Axes, *, only_group: GroupKey | None = None) -> tuple[float, float, float, float] | None:
            xmin, xmax = np.inf, -np.inf
            ymin, ymax = np.inf, -np.inf

            for gk in sorted(groups.keys()):
                if only_group is not None and gk != only_group:
                    continue

                items = groups[gk]
                if not items:
                    continue

                run_keys: list[Key] = []
                run_s: list[pd.Series] = []

                for k, nr in items:
                    s = _pga_series_for_run(nr)
                    if s.empty:
                        continue
                    run_keys.append(k)
                    run_s.append(s)

                if not run_s:
                    continue

                z_union = np.unique(np.concatenate([s.index.to_numpy(dtype=float) for s in run_s]))
                z_union = np.sort(z_union)

                A = []
                for s in run_s:
                    A.append(s.reindex(z_union).to_numpy(dtype=float))
                A = np.vstack(A)

                ymin = min(ymin, float(np.nanmin(z_union)))
                ymax = max(ymax, float(np.nanmax(z_union)))

                # individuals
                if stat is None or show_individual:
                    for k, row in zip(run_keys, A):
                        z = z_union
                        pga = row

                        ct = self._color_tag(gk=gk, k=k, color_by=color_by, color_by_group=color_by_group)
                        alpha = individual_alpha if stat is not None else 1.0

                        if color_by is None:
                            label = self._label_for(k, self.data[k]) if stat is None else "_nolegend_"
                            line, = ax.plot(pga, z, linewidth=linewidth, alpha=alpha, label=label)
                            if ct and ct not in tag_color:
                                tag_color[ct] = line.get_color()
                        else:
                            col = tag_color.get(ct)
                            if col is None:
                                line, = ax.plot(pga, z, linewidth=linewidth, alpha=alpha, label="_nolegend_")
                                if ct:
                                    tag_color[ct] = line.get_color()
                            else:
                                ax.plot(pga, z, color=col, linewidth=linewidth, alpha=alpha, label="_nolegend_")

                        if np.isfinite(pga).any():
                            xmin = min(xmin, float(np.nanmin(pga)))
                            xmax = max(xmax, float(np.nanmax(pga)))

                # group stat
                if stat is not None:
                    pga_g = self._reduce_stack(A, stat)

                    ct0 = self._color_tag(gk=gk, k=run_keys[0], color_by=color_by, color_by_group=color_by_group)
                    label = " | ".join(gk) + f" ({stat})"

                    if color_by is None:
                        line, = ax.plot(pga_g, z_union, linewidth=group_linewidth, label=label)
                        if ct0 and ct0 not in tag_color:
                            tag_color[ct0] = line.get_color()
                    else:
                        col = tag_color.get(ct0)
                        if col is None:
                            line, = ax.plot(pga_g, z_union, linewidth=group_linewidth, label=label)
                            if ct0:
                                tag_color[ct0] = line.get_color()
                        else:
                            ax.plot(pga_g, z_union, color=col, linewidth=group_linewidth, label=label)

                    if np.isfinite(pga_g).any():
                        xmin = min(xmin, float(np.nanmin(pga_g)))
                        xmax = max(xmax, float(np.nanmax(pga_g)))

            if not np.isfinite(xmin):
                return None
            return float(xmin), float(xmax), float(ymin), float(ymax)

        def _apply_limits(ax: plt.Axes, bounds: tuple[float, float, float, float] | None):
            if bounds is None:
                return
            xmin, xmax, ymin, ymax = bounds

            if xlim is None:
                ax.set_xlim(xmin, xmax)
            else:
                ax.set_xlim(*xlim)

            if ylim is None:
                ax.set_ylim(ymin, ymax)
            else:
                ax.set_ylim(*ylim)

        if overlay:
            fig, ax = plt.subplots(figsize=figsize)
            bounds = _plot_axes(ax)
            ax.set_xlabel("PGA (g)" if in_g else "PGA")
            ax.set_ylabel("z")
            ax.set_title(title or "PGA envelope")
            ax.grid(True, alpha=0.35)
            _apply_limits(ax, bounds)
            self._legend_below(fig, ax, fontsize=legend_fontsize, ncol=legend_ncol, frameon=legend_frameon)
            plt.tight_layout()
            return fig, ax

        figs = []
        for gk in sorted(groups.keys()):
            fig, ax = plt.subplots(figsize=figsize)
            bounds = _plot_axes(ax, only_group=gk)
            ax.set_xlabel("PGA (g)" if in_g else "PGA")
            ax.set_ylabel("z")
            ax.set_title((title or "PGA envelope") + f" — {' | '.join(gk)}")
            ax.grid(True, alpha=0.35)
            _apply_limits(ax, bounds)
            self._legend_below(fig, ax, fontsize=legend_fontsize, ncol=legend_ncol, frameon=legend_frameon)
            plt.tight_layout()
            figs.append((fig, ax))
        return figs

    def plot_torsion(
        self,
        *,
        z_coord: float,
        node_a_xy: tuple[float, float] = (5750, 5750),
        node_b_xy: tuple[float, float] = (38250, 25250),
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        overlay: bool = True,
        figsize: tuple[float, float] = (6, 4),
        title: str | None = None,
        order: Callable[[Key, Any], Any] | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        plot_residual: Literal[None, "norm", "rigidity"] = None,
        # --- updates (match plot_drift "labels & so on") ---
        group_by_color: str | None = None,   # None | "sta" | "rup" | "letter" | "number"
        linewidth: float = 1.0,
        legend_fontsize: float = 7,
        legend_ncol: int | None = None,
        legend_frameon: bool = False,
    ):
        """
        Plot torsion θ(t) computed from two reference roof points A,B.

        If plot_residual is provided:
        - "norm"     plots ||r(t)|| (displacement units)
        - "rigidity" plots ||r||/||Δu|| (dimensionless)

        Requires NodalResults.torsion(..., return_residual=True, return_quality=True).
        """
        valid_groups = {None, "sta", "rup", "letter", "number"}
        if group_by_color not in valid_groups:
            raise ValueError(f"group_by_color must be one of {valid_groups}")

        pairs = self.select(model=model, station=station, rupture=rupture, order=order)
        if not pairs:
            raise ValueError("No matching results for the given selection.")

        pairs = sorted(pairs, key=lambda kv: kv[0])
        palette = list(plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"]))

        def _group_value(k: Key) -> str:
            m, s, r = k
            if group_by_color is None:
                return ""
            if group_by_color == "sta":
                return str(s)
            if group_by_color == "rup":
                return str(r)
            if group_by_color == "letter":
                mm = str(m)
                m2 = re.search(r"([A-Za-z])\s*$", mm)
                return m2.group(1).upper() if m2 else ""
            if group_by_color == "number":
                mm = str(m)
                m2 = re.search(r"(\d+)", mm)
                return m2.group(1) if m2 else ""
            raise RuntimeError("unreachable")

        # color map
        color_map: dict[Any, str] = {}
        if group_by_color is None:
            for i, (k, _) in enumerate(pairs):
                color_map[k] = palette[i % len(palette)]
        else:
            groups_seen: list[str] = []
            for k, _ in pairs:
                g = _group_value(k)
                if g not in groups_seen:
                    groups_seen.append(g)
            for i, g in enumerate(sorted(groups_seen)):
                color_map[g] = palette[i % len(palette)]

        def plot_one(ax: plt.Axes, k: Key, nr: Any):
            # --- compute y series ---
            if plot_residual is None:
                tors = nr.torsion(
                    node_a_coord=(*node_a_xy, float(z_coord)),
                    node_b_coord=(*node_b_xy, float(z_coord)),
                    reduce="series",
                )
                y = np.asarray(tors.values, dtype=float)
            else:
                tors, dbg = nr.torsion(
                    node_a_coord=(*node_a_xy, float(z_coord)),
                    node_b_coord=(*node_b_xy, float(z_coord)),
                    reduce="series",
                    return_residual=True,
                    return_quality=True,
                )

                if plot_residual == "norm":
                    y = dbg["res_norm"].to_numpy(dtype=float)
                elif plot_residual == "rigidity":
                    y = dbg["rigidity_ratio"].to_numpy(dtype=float)
                else:
                    raise ValueError(f"Unknown plot_residual={plot_residual!r}")

            # --- align x/y ---
            t = np.asarray(nr.time, dtype=float)
            t2, y2, *_ = self._align_xy(t, y)
            if t2.size == 0:
                return None

            label = self._label_for(k, nr)
            kw = self._style_for_key(k)
            kw["linewidth"] = linewidth

            if group_by_color is None:
                kw["color"] = color_map[k]
            else:
                kw["color"] = color_map[_group_value(k)]

            ax.plot(t2, y2, label=label, **kw)

            xmin, xmax = float(np.nanmin(t2)), float(np.nanmax(t2))
            ymin, ymax = float(np.nanmin(y2)), float(np.nanmax(y2))
            return (xmin, xmax, ymin, ymax)

        ylabel = {
            None: "Torsion (rad)",
            "norm": "Residual magnitude ‖r(t)‖ (disp. units)",
            "rigidity": "Rigidity ratio ‖r‖ / ‖Δu‖",
        }[plot_residual]

        out = self._plot_overlay_or_facets(
            pairs=pairs,
            plot_one=plot_one,
            overlay=overlay,
            figsize_overlay=figsize,
            figsize_single=figsize,
            title=title
            or ("Roof torsion" if plot_residual is None else f"Roof torsion residual ({plot_residual})"),
            xlabel="Time (s)",
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            sym_x=False,
            sym_y=(plot_residual is None),  # torsion symmetric; residuals are nonnegative
            vline0=False,
            legend=False,  # <- we place legend below, like drift
            grid=True,
        )

        def _legend(fig: plt.Figure, ax: plt.Axes) -> None:
            self._legend_below(
                fig,
                ax,
                fontsize=legend_fontsize,
                ncol=legend_ncol,
                frameon=legend_frameon,
            )

        if overlay:
            fig, ax = out
            _legend(fig, ax)
            plt.tight_layout()
            return fig, ax

        figs = out
        for fig, ax in figs:
            _legend(fig, ax)
            plt.tight_layout()
        return figs

    def plot_base_rocking(
        self,
        *,
        z_coord: float,
        node_xy: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
            (5750, 5750),
            (38250, 5750),
            (5750, 25250),
        ),
        component: str = "theta_mag_rad",   # <- NEW DEFAULT (norm)
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        overlay: bool = True,
        figsize: tuple[float, float] = (6, 4),
        title: str | None = None,
        order: Callable[[Key, Any], Any] | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        # styling like your other plots
        group_by_color: str | None = None,   # None | "sta" | "rup" | "letter" | "number"
        linewidth: float = 1.0,
        legend_fontsize: float = 7,
        legend_ncol: int | None = None,
        legend_frameon: bool = False,
    ):
        """
        Plot base rocking time history.

        component:
        - "theta_mag_rad" (default)  -> sqrt(theta_x^2 + theta_y^2)
        - "theta_x_rad"
        - "theta_y_rad"
        - "w0"
        """
        valid_groups = {None, "sta", "rup", "letter", "number"}
        if group_by_color not in valid_groups:
            raise ValueError(f"group_by_color must be one of {valid_groups}")

        pairs = self.select(model=model, station=station, rupture=rupture, order=order)
        if not pairs:
            raise ValueError("No matching results for the given selection.")
        pairs = sorted(pairs, key=lambda kv: kv[0])

        palette = list(plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"]))

        def _group_value(k: Key) -> str:
            m, s, r = k
            if group_by_color is None:
                return ""
            if group_by_color == "sta":
                return str(s)
            if group_by_color == "rup":
                return str(r)
            if group_by_color == "letter":
                mm = str(m)
                m2 = re.search(r"([A-Za-z])\s*$", mm)
                return m2.group(1).upper() if m2 else ""
            if group_by_color == "number":
                mm = str(m)
                m2 = re.search(r"(\d+)", mm)
                return m2.group(1) if m2 else ""
            raise RuntimeError("unreachable")

        # color map
        color_map: dict[Any, str] = {}
        if group_by_color is None:
            for i, (k, _) in enumerate(pairs):
                color_map[k] = palette[i % len(palette)]
        else:
            groups_seen: list[str] = []
            for k, _ in pairs:
                g = _group_value(k)
                if g not in groups_seen:
                    groups_seen.append(g)
            for i, g in enumerate(sorted(groups_seen)):
                color_map[g] = palette[i % len(palette)]

        def plot_one(ax: plt.Axes, k: Key, nr: Any) -> tuple[float, float, float, float] | None:
            df = nr.base_rocking(
                node_coords_xy=node_xy,
                z_coord=float(z_coord),
                result_name="DISPLACEMENT",
                uz_component=3,
                reduce="series",
            )

            if component not in df.columns:
                raise ValueError(
                    f"component={component!r} not in base_rocking output columns {tuple(df.columns)}. "
                    "Use 'theta_mag_rad', 'theta_x_rad', 'theta_y_rad', or 'w0'."
                )

            y = np.asarray(df[component].to_numpy(), dtype=float)

            t = getattr(nr, "time", None)
            if t is not None and len(t) == len(y):
                x = np.asarray(t, dtype=float)
            else:
                x = np.asarray(df.index, dtype=float)

            x2, y2, *_ = self._align_xy(x, y)
            if x2.size == 0:
                return None

            label = self._label_for(k, nr)
            kw = self._style_for_key(k)
            kw["linewidth"] = linewidth

            if group_by_color is None:
                kw["color"] = color_map[k]
            else:
                kw["color"] = color_map[_group_value(k)]

            ax.plot(x2, y2, label=label, **kw)

            xmin, xmax = float(np.nanmin(x2)), float(np.nanmax(x2))
            ymin, ymax = float(np.nanmin(y2)), float(np.nanmax(y2))
            return (xmin, xmax, ymin, ymax)

        ylabel = {
            "theta_mag_rad": "Base rocking |θ| (rad)",
            "theta_x_rad": "Base rocking θx (rad)",
            "theta_y_rad": "Base rocking θy (rad)",
            "w0": "Base vertical translation w0 (disp. units)",
        }.get(component, f"Base rocking ({component})")

        out = self._plot_overlay_or_facets(
            pairs=pairs,
            plot_one=plot_one,
            overlay=overlay,
            figsize_overlay=figsize,
            figsize_single=figsize,
            title=title or f"Base rocking — {component}",
            xlabel="Time (s)",
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            sym_x=False,
            sym_y=(component != "theta_mag_rad"),  # |θ| is nonnegative; others symmetric-ish
            vline0=False,
            legend=False,
            grid=True,
        )

        def _legend(fig: plt.Figure, ax: plt.Axes) -> None:
            self._legend_below(
                fig,
                ax,
                fontsize=legend_fontsize,
                ncol=legend_ncol,
                frameon=legend_frameon,
            )

        if overlay:
            fig, ax = out
            _legend(fig, ax)
            plt.tight_layout()
            return fig, ax

        figs = out
        for fig, ax in figs:
            _legend(fig, ax)
            plt.tight_layout()
        return figs

    def plot_roof_torsion(
        self,
        *,
        z_coord: float,
        node_a_xy: tuple[float, float],
        node_b_xy: tuple[float, float],
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        plot_residual: str | None = None,   # None | 'rigidity'
        overlay: bool = True,
        figsize: tuple[float, float] = (10, 5),
        title: str | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        group_by_color: str | None = None,  # None | "sta" | "rup" | "letter" | "number"
        linewidth: float = 1.0,
        legend_fontsize: float = 7,
        legend_ncol: int | None = None,
        legend_frameon: bool = False,
    ):
        """
        Plot roof torsion (rotation about z) time history.

        Parameters
        ----------
        z_coord : float
            Z-coordinate of the roof level (model units).
        node_a_xy, node_b_xy : (float, float)
            XY coordinates of the two reference nodes at the roof level.
            Full 3-D coords are built as (*node_a_xy, z_coord).
        plot_residual : None | 'rigidity'
            None   -> plot theta_rad time series (default).
            'rigidity' -> plot the rigidity_ratio quality indicator instead
                          (requires return_quality=True on NodalResults.roof_torsion).
        """
        valid_residuals = {None, "rigidity"}
        if plot_residual not in valid_residuals:
            raise ValueError(f"plot_residual must be one of {valid_residuals}")

        valid_groups = {None, "sta", "rup", "letter", "number"}
        if group_by_color not in valid_groups:
            raise ValueError(f"group_by_color must be one of {valid_groups}")

        node_a_coord = (*node_a_xy, float(z_coord))
        node_b_coord = (*node_b_xy, float(z_coord))
        return_quality = plot_residual == "rigidity"

        pairs = self.select(model=model, station=station, rupture=rupture)
        if not pairs:
            raise ValueError("No matching results for the given selection.")
        pairs = sorted(pairs, key=lambda kv: kv[0])

        palette = list(plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"]))

        def _group_value(k: Key) -> str:
            m, s, r = k
            if group_by_color is None:
                return ""
            if group_by_color == "sta":
                return str(s)
            if group_by_color == "rup":
                return str(r)
            if group_by_color == "letter":
                mm = str(m)
                m2 = re.search(r"([A-Za-z])\s*$", mm)
                return m2.group(1).upper() if m2 else ""
            if group_by_color == "number":
                mm = str(m)
                m2 = re.search(r"(\d+)", mm)
                return m2.group(1) if m2 else ""
            raise RuntimeError("unreachable")

        color_map: dict[Any, str] = {}
        if group_by_color is None:
            for i, (k, _) in enumerate(pairs):
                color_map[k] = palette[i % len(palette)]
        else:
            groups_seen: list[str] = []
            for k, _ in pairs:
                g = _group_value(k)
                if g not in groups_seen:
                    groups_seen.append(g)
            for i, g in enumerate(sorted(groups_seen)):
                color_map[g] = palette[i % len(palette)]

        def plot_one(ax: plt.Axes, k: Key, nr: Any) -> tuple[float, float, float, float] | None:
            result = nr.roof_torsion(
                node_a_coord=node_a_coord,
                node_b_coord=node_b_coord,
                result_name="DISPLACEMENT",
                ux_component=1,
                uy_component=2,
                return_quality=return_quality,
            )

            if return_quality:
                # result is (theta_series, debug_df)
                theta_series, debug_df = result
                if plot_residual == "rigidity":
                    y_series = debug_df["rigidity_ratio"]
                else:
                    y_series = theta_series
            else:
                y_series = result

            y = np.asarray(y_series.to_numpy(), dtype=float)

            t = getattr(nr, "time", None)
            if t is not None and len(t) == len(y):
                x = np.asarray(t, dtype=float)
            else:
                x = np.asarray(y_series.index, dtype=float)

            x2, y2, *_ = self._align_xy(x, y)
            if x2.size == 0:
                return None

            label = self._label_for(k, nr)
            kw = self._style_for_key(k)
            kw["linewidth"] = linewidth

            if group_by_color is None:
                kw["color"] = color_map[k]
            else:
                kw["color"] = color_map[_group_value(k)]

            ax.plot(x2, y2, label=label, **kw)

            xmin, xmax = float(np.nanmin(x2)), float(np.nanmax(x2))
            ymin, ymax = float(np.nanmin(y2)), float(np.nanmax(y2))
            return (xmin, xmax, ymin, ymax)

        if plot_residual == "rigidity":
            ylabel = "Rigidity ratio (–)"
            default_title = "Roof torsion — rigidity ratio"
        else:
            ylabel = "Roof rotation θz (rad)"
            default_title = "Roof torsion θz"

        out = self._plot_overlay_or_facets(
            pairs=pairs,
            plot_one=plot_one,
            overlay=overlay,
            figsize_overlay=figsize,
            figsize_single=figsize,
            title=title or default_title,
            xlabel="Time (s)",
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            sym_x=False,
            sym_y=(plot_residual is None),
            vline0=False,
            legend=False,
            grid=True,
        )

        def _legend(fig: plt.Figure, ax: plt.Axes) -> None:
            self._legend_below(
                fig,
                ax,
                fontsize=legend_fontsize,
                ncol=legend_ncol,
                frameon=legend_frameon,
            )

        if overlay:
            fig, ax = out
            _legend(fig, ax)
            plt.tight_layout()
            return fig, ax

        figs = out
        for fig, ax in figs:
            _legend(fig, ax)
            plt.tight_layout()
        return figs

    def plot_orbit(
        self,
        *,
        top: tuple[float, float, float],
        bottom: tuple[float, float, float],
        x_component: int = 1,
        y_component: int = 2,
        relative_drift: bool = False,          # True -> divide by height (drift ratio). False -> Δu (no /dz)
        result_name: str = "DISPLACEMENT",

        # run selection
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,

        # plot layout
        overlay: bool = True,
        figsize: tuple[float, float] = (6, 6),
        title: str | None = None,
        equal_aspect: bool = True,
        square_axes: bool = True,             # <-- NEW: force same numeric span on x and y (square window)
        grid: bool = True,
        legend_fontsize: float = 7,
        legend_ncol: int | None = None,
        legend_frameon: bool = False,

        # kept for compatibility; styling is handled by _style_for_key
        group_by_color: str | None = None,

        # style
        linewidth: float = 1.0,
        alpha: float = 1.0,

        # axis limits
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,

        # markers
        show_start_end: bool = True,
        start_end_size: float = 20.0,
    ):
        """
        Orbit plot of relative motion between top and bottom, using the same aligned samples:

        - relative_drift=True:
                dx(t) = (u_top_x - u_bot_x) / dz
                dy(t) = (u_top_y - u_bot_y) / dz

        - relative_drift=False:
                dx(t) = (u_top_x - u_bot_x)
                dy(t) = (u_top_y - u_bot_y)

        Plots dy vs dx (parametric).

        square_axes=True forces the *numeric* axis ranges to match (square window),
        in addition to equal aspect.
        """
        valid_groups = {None, "sta", "rup", "letter", "number"}
        if group_by_color not in valid_groups:
            raise ValueError(f"group_by_color must be one of {valid_groups}")

        pairs = self.select(model=model, station=station, rupture=rupture)
        if not pairs:
            raise ValueError("No matching results for the given selection.")
        pairs = sorted(pairs, key=lambda kv: kv[0])

        # ------------------------------------------------------------
        # Helper: delta_u (no /dz) for one component, matching drift() logic
        # ------------------------------------------------------------
        def _delta_u_series(nr: Any, *, comp: int) -> "pd.Series":
            top_id = int(nr.info.nearest_node_id([top], return_distance=False)[0])
            bot_id = int(nr.info.nearest_node_id([bottom], return_distance=False)[0])

            s = nr.fetch(result_name=result_name, component=comp, node_ids=[top_id, bot_id])

            # If multi-stage exists, pick last stage deterministically (MPCO plot has no stage arg)
            if isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 3:
                stages = tuple(sorted({str(x) for x in s.index.get_level_values(0)}))
                if not stages:
                    raise ValueError("plot_orbit(relative_drift=False): no stages found.")
                s = s.xs(stages[-1], level=0)

            if not (isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 2):
                raise ValueError("plot_orbit(relative_drift=False): expected index (node_id, step).")

            u_top = s.xs(top_id, level=0).sort_index()
            u_bot = s.xs(bot_id, level=0).sort_index()
            u_top, u_bot = u_top.align(u_bot, join="inner")

            du = u_top - u_bot
            du.name = f"delta_u({result_name}:{comp})"
            return du

        # ------------------------------------------------------------
        # Helper: force square numeric limits
        # ------------------------------------------------------------
        def _square_axis_limits(ax: plt.Axes, *, symmetric: bool) -> None:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()

            if symmetric:
                a = max(abs(x0), abs(x1), abs(y0), abs(y1))
                if not np.isfinite(a) or a == 0.0:
                    a = 1.0
                ax.set_xlim(-a, a)
                ax.set_ylim(-a, a)
                return

            lo = min(x0, y0)
            hi = max(x1, y1)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
                lo, hi = -1.0, 1.0
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        # ------------------------------------------------------------
        # Plot per run
        # ------------------------------------------------------------
        def plot_one(ax: plt.Axes, k: Key, nr: Any):
            if relative_drift:
                dx = nr.drift(top=top, bottom=bottom, component=x_component, result_name=result_name, reduce="series")
                dy = nr.drift(top=top, bottom=bottom, component=y_component, result_name=result_name, reduce="series")
            else:
                dx = _delta_u_series(nr, comp=x_component)
                dy = _delta_u_series(nr, comp=y_component)

            # enforce same samples
            dx, dy = dx.align(dy, join="inner")
            if dx.size == 0:
                return None

            x = np.asarray(dx.to_numpy(dtype=float), dtype=float)
            y = np.asarray(dy.to_numpy(dtype=float), dtype=float)

            n = min(x.size, y.size)
            if n == 0:
                return None
            x = x[:n]
            y = y[:n]

            label = self._label_for(k, nr)

            kw = self._style_for_key(k)      # <-- style owns color/ls/marker
            kw["linewidth"] = linewidth
            kw["alpha"] = alpha

            ax.plot(x, y, label=label, **kw)

            if show_start_end and n > 0:
                ax.scatter([x[0]], [y[0]], s=start_end_size, alpha=alpha)
                ax.scatter([x[-1]], [y[-1]], s=start_end_size, alpha=alpha)

            if not np.isfinite(x).any() or not np.isfinite(y).any():
                return None

            return (
                float(np.nanmin(x)),
                float(np.nanmax(x)),
                float(np.nanmin(y)),
                float(np.nanmax(y)),
            )

        # labels / title
        if relative_drift:
            xlabel = f"Drift [{x_component}]"
            ylabel = f"Drift [{y_component}]"
            ttl = title or f"Drift orbit — {result_name}[{x_component}] vs [{y_component}]"
            symx = True
            symy = True
        else:
            xlabel = f"Δu [{x_component}]"
            ylabel = f"Δu [{y_component}]"
            ttl = title or f"Δu orbit — {result_name}[{x_component}] vs [{y_component}]"
            symx = False
            symy = False

        out = self._plot_overlay_or_facets(
            pairs=pairs,
            plot_one=plot_one,
            overlay=overlay,
            figsize_overlay=figsize,
            figsize_single=figsize,
            title=ttl,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            sym_x=symx,
            sym_y=symy,
            vline0=False,
            legend=False,
            grid=grid,
        )

        def _legend(fig: plt.Figure, ax: plt.Axes):
            self._legend_below(fig, ax, fontsize=legend_fontsize, ncol=legend_ncol, frameon=legend_frameon)

        if overlay:
            fig, ax = out

            if equal_aspect:
                ax.set_aspect("equal", adjustable="box")

            # Force square numeric window ONLY when user didn't force limits
            if square_axes and (xlim is None) and (ylim is None):
                _square_axis_limits(ax, symmetric=relative_drift)

            _legend(fig, ax)
            plt.tight_layout()
            return fig, ax

        figs = out
        for fig, ax in figs:
            if equal_aspect:
                ax.set_aspect("equal", adjustable="box")

            if square_axes and (xlim is None) and (ylim is None):
                _square_axis_limits(ax, symmetric=relative_drift)

            _legend(fig, ax)
            plt.tight_layout()
        return figs

    # ------------------------------------------------------------------
    # Compute table
    # ------------------------------------------------------------------
    def compute_table(
        self,
        *,
        metrics: Mapping[str, Callable[[Key, Any], Any]] | Sequence[str],
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        order: Callable[[Key, Any], Any] | None = None,
        include_label: bool = True,
        drop_na_rows: bool = False,
    ) -> pd.DataFrame:
        pairs = self.select(model=model, station=station, rupture=rupture, order=order)
        if not pairs:
            raise ValueError("No matching results for the given selection.")

        fns: dict[str, Callable[[Key, Any], Any]] = {}

        if isinstance(metrics, Mapping):
            for name, fn in metrics.items():
                if not callable(fn):
                    raise TypeError(f"metrics[{name!r}] is not callable.")
                fns[str(name)] = fn
        else:
            for name in metrics:
                nm = str(name)

                def _make_info_getter(attr: str) -> Callable[[Key, Any], Any]:
                    return lambda _k, nr: getattr(nr.info, attr, None)

                fns[nm] = _make_info_getter(nm)

        rows: list[dict[str, Any]] = []

        for k, nr in pairs:
            m, s, r = k
            row: dict[str, Any] = {"model": m, "station": s, "rupture": r}
            if include_label:
                row["label"] = self._label_for(k, nr)

            for name, fn in fns.items():
                val = fn(k, nr)

                if isinstance(val, (int, float, np.integer, np.floating)) or val is None:
                    row[name] = None if val is None else float(val)

                elif isinstance(val, dict):
                    for kk, vv in val.items():
                        col = f"{name}.{kk}"
                        row[col] = None if vv is None else float(vv)

                else:
                    try:
                        row[name] = float(val)  # type: ignore[arg-type]
                    except Exception:
                        row[name] = val

            rows.append(row)

        df = pd.DataFrame(rows)

        if drop_na_rows:
            metric_cols = [c for c in df.columns if c not in ("model", "station", "rupture", "label")]
            if metric_cols:
                df = df.dropna(subset=metric_cols, how="all")

        return df

    # ------------------------------------------------------------------
    # Metric matrix and plots
    # ------------------------------------------------------------------
    def metric_matrix(
        self,
        *,
        metric: str = "analysis_time",
        agg: str = "mean",
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
    ) -> pd.DataFrame:
        reducers = {"mean", "median", "max", "min", "sum"}
        if agg not in reducers:
            raise ValueError(f"Unknown agg='{agg}'. Use one of {sorted(reducers)}.")

        T = self.compute_table(metrics=[metric], model=model, station=station, rupture=rupture, include_label=False)
        T = T.dropna(subset=[metric])
        if T.empty:
            raise ValueError(f"No values found for metric={metric!r}.")

        tiers: list[int] = []
        cases: list[str] = []
        for m in T["model"].astype(str).tolist():
            t, c = self.parse_tier_letter(m)
            tiers.append(t)
            cases.append(c)

        T = T.copy()
        T["Tier"] = tiers
        T["Case"] = cases

        cases_sorted = sorted(T["Case"].unique())
        tiers_sorted = sorted(T["Tier"].unique())

        mat = (
            T.groupby(["Case", "Tier"])[metric]
            .agg(agg)
            .unstack("Tier")
            .reindex(index=cases_sorted, columns=tiers_sorted)
        )
        return mat

    def plot_metric_heatmap(
        self,
        *,
        metric: str = "analysis_time",
        agg: str = "mean",
        title: str | None = None,
        cmap: str = "viridis",
        figsize: tuple[float, float] = (7, 4.5),
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        use_seaborn: bool = True,
        show_std: bool = True,
        std_k: float = 1.0,
        fmt_mean: str = ".2f",
        fmt_std: str = ".2f",
        annot: bool = True,
        center: float | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        ax: plt.Axes | None = None,
        fontsize: float = 8,
    ):
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
            
        reducers = {"mean", "median", "max", "min", "sum"}
        if agg not in reducers:
            raise ValueError(f"Unknown agg='{agg}'. Use one of {sorted(reducers)}.")

        T = self.compute_table(
            metrics=[metric],
            model=model,
            station=station,
            rupture=rupture,
            include_label=False,
        ).dropna(subset=[metric])

        if T.empty:
            raise ValueError(f"No values found for metric={metric!r}.")

        tiers, cases = [], []
        for m in T["model"].astype(str).tolist():
            t, c = self.parse_tier_letter(m)
            tiers.append(t)
            cases.append(c)

        T = T.copy()
        T["Tier"] = tiers
        T["Case"] = cases

        cases_sorted = sorted(T["Case"].unique())
        tiers_sorted = sorted(T["Tier"].unique())

        gb = T.groupby(["Case", "Tier"])[metric]

        if agg == "mean":
            mat_mean = gb.mean().unstack("Tier").reindex(index=cases_sorted, columns=tiers_sorted)
            mat_std = gb.std(ddof=1).unstack("Tier").reindex(index=cases_sorted, columns=tiers_sorted)
            Z = mat_mean.to_numpy(dtype=float)
        elif agg == "median":
            mat_mean = gb.median().unstack("Tier").reindex(index=cases_sorted, columns=tiers_sorted)
            mat_std = None
            Z = mat_mean.to_numpy(dtype=float)
        elif agg == "max":
            mat_mean = gb.max().unstack("Tier").reindex(index=cases_sorted, columns=tiers_sorted)
            mat_std = None
            Z = mat_mean.to_numpy(dtype=float)
        elif agg == "min":
            mat_mean = gb.min().unstack("Tier").reindex(index=cases_sorted, columns=tiers_sorted)
            mat_std = None
            Z = mat_mean.to_numpy(dtype=float)
        elif agg == "sum":
            mat_mean = gb.sum().unstack("Tier").reindex(index=cases_sorted, columns=tiers_sorted)
            mat_std = None
            Z = mat_mean.to_numpy(dtype=float)
        else:
            raise RuntimeError("unreachable")

        annot_data = None
        if annot:
            annot_data = np.full(Z.shape, "", dtype=object)
            for i, case in enumerate(mat_mean.index):
                for j, tier in enumerate(mat_mean.columns):
                    v = mat_mean.loc[case, tier]
                    if pd.isna(v):
                        continue

                    if agg == "mean" and show_std and mat_std is not None:
                        s = mat_std.loc[case, tier]
                        if pd.isna(s):
                            annot_data[i, j] = format(float(v), fmt_mean)
                        else:
                            annot_data[i, j] = f"{format(float(v), fmt_mean)}\n±{format(float(std_k*s), fmt_std)}"
                    else:
                        annot_data[i, j] = format(float(v), fmt_mean)

        if use_seaborn:
            try:
                import seaborn as sns

                sns.set_theme(style="white")
                sns.heatmap(
                    mat_mean,
                    ax=ax,
                    cmap=cmap,
                    annot=annot_data if annot else False,
                    fmt="",
                    annot_kws={"fontsize": fontsize},
                    linewidths=0.5,
                    linecolor="white",
                    cbar_kws={"label": metric},
                    center=center,
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_xlabel("Tier", fontsize=fontsize)
                ax.set_ylabel("Case", fontsize=fontsize)
                ax.set_xticklabels([f"Tier {c}" for c in mat_mean.columns.tolist()], fontsize=fontsize-1, rotation=45)
                ax.set_yticklabels(mat_mean.index.tolist(), rotation=0, fontsize=fontsize-1)

            except Exception:
                use_seaborn = False

        if not use_seaborn:
            im = ax.imshow(Z, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(Z.shape[1]), fontsize=fontsize-1)
            ax.set_xticklabels([f"Tier {c}" for c in mat_mean.columns.tolist()], fontsize=fontsize-1, rotation=45)
            ax.set_yticks(np.arange(Z.shape[0]), fontsize=fontsize-1)
            ax.set_yticklabels(mat_mean.index.tolist(), fontsize=fontsize)

            if annot and annot_data is not None:
                for i in range(Z.shape[0]):
                    for j in range(Z.shape[1]):
                        s = annot_data[i, j]
                        if s:
                            ax.text(j, i, s, ha="center", va="center", fontsize=fontsize)

            cbar = fig.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label(metric)

            ax.set_xlabel("Tier", fontsize=fontsize)
            ax.set_ylabel("Case", fontsize=fontsize)

        if title is None:
            if agg == "mean" and show_std:
                title = f"{metric} matrix (mean ± {std_k:g}σ)"
            else:
                title = f"{metric} matrix ({agg})"
        ax.set_title(title)

        plt.tight_layout()
        return fig, ax

    def plot_metric_barh(
        self,
        *,
        metric: str = "analysis_time",
        agg: str = "mean",
        sort: bool = True,
        title: str | None = None,
        figsize: tuple[float, float] = (7, 5),
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        show_std_errorbar: bool = True,
        show_std_text: bool = False,
        std_k: float = 1.0,
        text_anchor: str = "error_end",   # "value" | "error_end"
        text_pad_frac: float = 0.01,
        err_elinewidth: float = 1.0,
        err_capthick: float = 1.0,
        err_capsize: float = 3.0,
        right_margin_frac: float = 0.08,
        left_margin_frac: float = 0.02,
        ax: plt.Axes | None = None,
        fontsize: float = 8,
    ):
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        reducers = {"mean", "median", "max", "min", "sum"}
        if agg not in reducers:
            raise ValueError(f"Unknown agg='{agg}'. Use one of {sorted(reducers)}.")
        if text_anchor not in ("value", "error_end"):
            raise ValueError("text_anchor must be 'value' or 'error_end'.")

        T = self.compute_table(
            metrics=[metric],
            model=model,
            station=station,
            rupture=rupture,
            include_label=False,
        ).dropna(subset=[metric])

        if T.empty:
            raise ValueError(f"No values found for metric={metric!r}.")

        tiers, cases = [], []
        for m in T["model"].astype(str):
            t, c = self.parse_tier_letter(m)
            tiers.append(t)
            cases.append(c)

        T = T.copy()
        T["Tier"] = tiers
        T["Case"] = cases
        T["Label"] = T["Tier"].astype(str) + T["Case"]

        gb = T.groupby(["Tier", "Case"])[metric]

        if agg == "mean":
            A = gb.mean().reset_index(name="value")
            A["std"] = gb.std(ddof=1).reset_index(drop=True)
            A["err"] = std_k * A["std"]
        else:
            A = getattr(gb, agg)().reset_index(name="value")
            A["std"] = np.nan
            A["err"] = np.nan

        A["Label"] = A["Tier"].astype(str) + A["Case"]

        if sort:
            A = A.sort_values("value", ascending=True).reset_index(drop=True)
        else:
            A = A.reset_index(drop=True)

        y = np.arange(len(A))
        vals = A["value"].to_numpy(dtype=float)
        stds = A["std"].to_numpy(dtype=float)
        errs = A["err"].to_numpy(dtype=float)

        ax.barh(y, vals, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(A["Label"], fontsize=fontsize)

        if agg == "mean" and show_std_errorbar:
            ax.errorbar(
                vals,
                y,
                xerr=np.nan_to_num(errs, nan=0.0),
                fmt="none",
                ecolor="black",
                elinewidth=err_elinewidth,
                capsize=err_capsize,
                capthick=err_capthick,
                zorder=3,
            )

        err_safe = np.nan_to_num(errs, nan=0.0)

        x_data_min = float(np.nanmin(vals - err_safe))
        x_data_max = float(np.nanmax(vals + err_safe))
        span_for_pad = max(x_data_max - min(0.0, x_data_min), 1.0)
        pad = text_pad_frac * span_for_pad

        for i, (v, s, e) in enumerate(zip(vals, stds, err_safe)):
            if agg == "mean" and show_std_text and np.isfinite(s):
                txt = f"{v:.0f} ± {std_k*s:.0f}"
            else:
                txt = f"{v:.0f}"

            if text_anchor == "error_end" and agg == "mean":
                x_text = v + e + pad
            else:
                x_text = v + pad

            ax.text(x_text, i, txt, va="center", ha="left", fontsize=fontsize, zorder=4)

        ax.set_xlabel(metric, fontsize=fontsize)
        ax.set_ylabel("Tier–Case", fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize)

        if title is None:
            if agg == "mean" and (show_std_errorbar or show_std_text):
                title = f"{metric} per Tier–Case (mean ± {std_k:g}σ)"
            else:
                title = f"{metric} per Tier–Case ({agg})"
        ax.set_title(title, fontsize=fontsize)

        ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=1)

        if show_std_text:
            if text_anchor == "error_end" and agg == "mean":
                x_text_max = float(np.nanmax(vals + err_safe + pad))
            else:
                x_text_max = float(np.nanmax(vals + pad))
        else:
            x_text_max = x_data_max

        x_right = max(x_data_max, x_text_max)
        span = max(x_right - min(0.0, x_data_min), 1.0)

        x_left = min(0.0, x_data_min) - left_margin_frac * span
        x_right = x_right + right_margin_frac * span
        ax.set_xlim(x_left, x_right)

        plt.tight_layout()
        return fig, ax

    def plot_metric_3dbar(
        self,
        *,
        metric: str = "analysis_time",
        agg: str = "mean",
        logz: bool = False,
        title: str | None = None,
        figsize: tuple[float, float] = (9.5, 7),
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        cmap_name: str = "seismic",
        zlim: tuple[float, float] | None = None,
        z_margin_frac: float = 0.08,
        show_values: bool = True,
        value_fmt: str = ".0f",
        value_fontsize: float = 8,
        show_std: bool = False,
        std_k: float = 1.0,
        std_color: str = "black",
        std_linewidth: float = 1.0,
    ):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        reducers = {"mean", "median", "max", "min", "sum"}
        if agg not in reducers:
            raise ValueError(f"Unknown agg='{agg}'. Use one of {sorted(reducers)}.")

        mat = self.metric_matrix(metric=metric, agg=agg, model=model, station=station, rupture=rupture)
        Z = mat.to_numpy(dtype=float)

        cases = list(mat.index)
        tiers = list(mat.columns)

        x = np.arange(len(tiers))
        y = np.arange(len(cases))
        xpos, ypos = np.meshgrid(x, y)
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros_like(xpos, dtype=float)
        dz = Z.ravel()

        mask = ~np.isnan(dz)
        xpos, ypos, dz = xpos[mask], ypos[mask], dz[mask]
        zpos = zpos[mask]

        dx = 0.55
        dy = 0.55

        if dz.size == 0:
            raise ValueError("No values to plot (all NaN).")

        norm = Normalize(vmin=float(np.nanmin(dz)), vmax=float(np.nanmax(dz)))
        cmap = cm.get_cmap(cmap_name)
        colors = cmap(norm(dz))

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        ax.bar3d(
            xpos, ypos, zpos,
            dx, dy, dz,
            color=colors,
            shade=True,
            edgecolor=(0, 0, 0, 0.15),
            linewidth=0.3,
        )

        ax.view_init(elev=22, azim=-55)

        ax.set_xticks(x + dx / 2)
        ax.set_xticklabels([f"Tier {t}" for t in tiers])
        ax.set_yticks(y + dy / 2)
        ax.set_yticklabels(cases)

        ax.set_xlabel("Tier", labelpad=10)
        ax.set_ylabel("Case", labelpad=10)
        ax.set_zlabel(metric, labelpad=10)

        if logz:
            ax.set_zscale("log")

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_edgecolor((1, 1, 1, 0))
            axis.pane.set_facecolor((0.98, 0.98, 0.98, 0.9))

        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)

        std_map: dict[tuple[int, str], float] = {}
        if show_std:
            if agg != "mean":
                raise ValueError("show_std=True only supported when agg='mean'.")

            T = self.compute_table(
                metrics=[metric],
                model=model,
                station=station,
                rupture=rupture,
                include_label=False,
            ).dropna(subset=[metric])

            if not T.empty:
                tiers2, cases2 = [], []
                for m in T["model"].astype(str):
                    t, c = self.parse_tier_letter(m)
                    tiers2.append(t)
                    cases2.append(c)
                T = T.copy()
                T["Tier"] = tiers2
                T["Case"] = cases2

                gb = T.groupby(["Tier", "Case"])[metric].std(ddof=1)
                for (t, c), s in gb.items():
                    if pd.notna(s):
                        std_map[(int(t), str(c))] = float(s)

            for xi, yi, hi in zip(xpos, ypos, dz):
                tier = int(tiers[int(round(xi))]) if isinstance(tiers[0], (int, np.integer)) else int(xi) + 1
                case = str(cases[int(round(yi))])
                s = std_map.get((tier, case))
                if s is None:
                    continue
                e = std_k * s
                xmid = xi + dx / 2
                ymid = yi + dy / 2
                ax.plot(
                    [xmid, xmid],
                    [ymid, ymid],
                    [hi - e, hi + e],
                    color=std_color,
                    linewidth=std_linewidth,
                )

        if show_values:
            for xi, yi, hi in zip(xpos, ypos, dz):
                ax.text(
                    xi + dx / 2,
                    yi + dy / 2,
                    hi,
                    format(float(hi), value_fmt),
                    ha="center",
                    va="bottom",
                    fontsize=value_fontsize,
                )

        if zlim is None:
            dz_max = float(np.nanmax(dz))
            dz_min = float(np.nanmin(dz))

            if show_std and std_map:
                max_std = float(max(std_map.values()))
                dz_max = dz_max + std_k * max_std

            span = max(dz_max - max(0.0, dz_min), 1e-12)
            z0 = 0.0 if not logz else max(dz_min, 1e-12)
            z1 = dz_max + z_margin_frac * span
            ax.set_zlim(z0, z1)
        else:
            ax.set_zlim(*zlim)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.08, shrink=0.7)
        cbar.set_label(metric)

        ax.set_title(title or f"{metric} per Tier–Case ({agg})", pad=14)
        plt.tight_layout()
        plt.show()

    def plot_asce_torsional_irregularity_heatmap(
        self,
        *,
        component: int = 1,
        side_a_top: tuple[float, float, float],
        side_a_bottom: tuple[float, float, float],
        side_b_top: tuple[float, float, float],
        side_b_bottom: tuple[float, float, float],
        result_name: str = "DISPLACEMENT",
        stage: str | None = None,
        reduce_time: str = "abs_max",          # "abs_max" | "max" | "min"
        definition: str = "max_over_avg",      # "max_over_avg" | "max_over_min"
        eps: float = 1e-16,
        # selection
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        # plot options (mirrors plot_metric_heatmap)
        title: str | None = None,
        cmap: str = "viridis",
        figsize: tuple[float, float] = (7, 4.5),
        use_seaborn: bool = True,
        show_std: bool = True,
        std_k: float = 1.0,
        fmt_mean: str = ".2f",
        fmt_std: str = ".2f",
        annot: bool = True,
        center: float | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        metric_name = "asce_torsion_ratio"

        metrics = {
            metric_name: lambda _k, nr: nr.asce_torsional_irregularity(
                component=component,
                side_a_top=side_a_top,
                side_a_bottom=side_a_bottom,
                side_b_top=side_b_top,
                side_b_bottom=side_b_bottom,
                result_name=result_name,
                stage=stage,
                reduce_time=reduce_time,
                definition=definition,
                eps=eps,
            )["ratio"]
        }

        T = self.compute_table(
            metrics=metrics,
            model=model,
            station=station,
            rupture=rupture,
            include_label=False,
        ).dropna(subset=[metric_name])

        if T.empty:
            raise ValueError("No values found for ASCE torsional irregularity ratio for the given selection.")

        tiers, cases = [], []
        for m in T["model"].astype(str).tolist():
            t, c = self.parse_tier_letter(m)
            tiers.append(t)
            cases.append(c)

        T = T.copy()
        T["Tier"] = tiers
        T["Case"] = cases

        cases_sorted = sorted(T["Case"].unique())
        tiers_sorted = sorted(T["Tier"].unique())

        gb = T.groupby(["Case", "Tier"])[metric_name]

        if show_std:
            mat_mean = gb.mean().unstack("Tier").reindex(index=cases_sorted, columns=tiers_sorted)
            mat_std = gb.std(ddof=1).unstack("Tier").reindex(index=cases_sorted, columns=tiers_sorted)
        else:
            mat_mean = gb.mean().unstack("Tier").reindex(index=cases_sorted, columns=tiers_sorted)
            mat_std = None

        Z = mat_mean.to_numpy(dtype=float)

        annot_data = None
        if annot:
            annot_data = np.full(Z.shape, "", dtype=object)
            for i, case in enumerate(mat_mean.index):
                for j, tier in enumerate(mat_mean.columns):
                    v = mat_mean.loc[case, tier]
                    if pd.isna(v):
                        continue
                    if show_std and mat_std is not None:
                        s = mat_std.loc[case, tier]
                        if pd.isna(s):
                            annot_data[i, j] = format(float(v), fmt_mean)
                        else:
                            annot_data[i, j] = f"{format(float(v), fmt_mean)}\n±{format(float(std_k*s), fmt_std)}"
                    else:
                        annot_data[i, j] = format(float(v), fmt_mean)

        fig, ax = plt.subplots(figsize=figsize)

        if use_seaborn:
            try:
                import seaborn as sns
                sns.set_theme(style="white")
                sns.heatmap(
                    mat_mean,
                    ax=ax,
                    cmap=cmap,
                    annot=annot_data if annot else False,
                    fmt="",
                    linewidths=0.5,
                    linecolor="white",
                    cbar_kws={"label": metric_name},
                    center=center,
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_xlabel("Tier")
                ax.set_ylabel("Case")
                ax.set_xticklabels([f"Tier {c}" for c in mat_mean.columns.tolist()])
                ax.set_yticklabels(mat_mean.index.tolist(), rotation=0)
            except Exception:
                use_seaborn = False

        if not use_seaborn:
            im = ax.imshow(Z, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(Z.shape[1]))
            ax.set_xticklabels([f"Tier {c}" for c in mat_mean.columns.tolist()])
            ax.set_yticks(np.arange(Z.shape[0]))
            ax.set_yticklabels(mat_mean.index.tolist())

            if annot and annot_data is not None:
                for i in range(Z.shape[0]):
                    for j in range(Z.shape[1]):
                        s = annot_data[i, j]
                        if s:
                            ax.text(j, i, s, ha="center", va="center", fontsize=9)

            cbar = fig.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label(metric_name)

            ax.set_xlabel("Tier")
            ax.set_ylabel("Case")

        if title is None:
            title = f"ASCE torsional irregularity ratio ({definition}, {reduce_time})"
            if show_std:
                title += f" — mean ± {std_k:g}σ"
        ax.set_title(title)

        plt.tight_layout()
        return fig, ax

    def plot_asce_torsional_irregularity_barh(
        self,
        *,
        component: int = 1,
        side_a_top: tuple[float, float, float],
        side_a_bottom: tuple[float, float, float],
        side_b_top: tuple[float, float, float],
        side_b_bottom: tuple[float, float, float],
        result_name: str = "DISPLACEMENT",
        stage: str | None = None,
        reduce_time: str = "abs_max",
        definition: str = "max_over_avg",
        eps: float = 1e-16,
        # selection
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        # plot
        sort: bool = True,
        figsize: tuple[float, float] = (8, 7),
        title: str | None = None,
    ):
        metric_name = "asce_torsion_ratio"

        metrics = {
            metric_name: lambda _k, nr: nr.asce_torsional_irregularity(
                component=component,
                side_a_top=side_a_top,
                side_a_bottom=side_a_bottom,
                side_b_top=side_b_top,
                side_b_bottom=side_b_bottom,
                result_name=result_name,
                stage=stage,
                reduce_time=reduce_time,
                definition=definition,
                eps=eps,
            )["ratio"]
        }

        T = self.compute_table(
            metrics=metrics,
            model=model,
            station=station,
            rupture=rupture,
            include_label=True,
        ).dropna(subset=[metric_name])

        if T.empty:
            raise ValueError("No values found for ASCE torsional irregularity ratio for the given selection.")

        # label as TierCase + station + rupture (handy)
        tiers, cases = [], []
        for m in T["model"].astype(str).tolist():
            t, c = self.parse_tier_letter(m)
            tiers.append(t)
            cases.append(c)

        T = T.copy()
        T["Tier"] = tiers
        T["Case"] = cases
        T["TierCase"] = T["Tier"].astype(str) + T["Case"].astype(str)
        T["RowLabel"] = T["TierCase"] + " | " + T["station"].astype(str) + " | " + T["rupture"].astype(str)

        if sort:
            T = T.sort_values(metric_name, ascending=True).reset_index(drop=True)
        else:
            T = T.reset_index(drop=True)

        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(len(T))
        vals = T[metric_name].to_numpy(dtype=float)

        ax.barh(y, vals, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(T["RowLabel"].tolist(), fontsize=8)

        ax.set_xlabel(metric_name)
        if title is None:
            title = f"ASCE torsional irregularity ratio ({definition}, {reduce_time})"
        ax.set_title(title)

        ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=1)
        plt.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Dataframe extraction
    # ------------------------------------------------------------------

    def collect_interstory_drift_envelope_pd(
        self,
        *,
        component: object,
        selection_set_name: str | None = "CenterPoints",
        selection_set_id: int | Sequence[int] | None = None,
        node_ids: Sequence[int] | None = None,
        coordinates: Sequence[Sequence[float]] | None = None,
        result_name: str = "DISPLACEMENT",
        stage: str | None = None,
        dz_tol: float = 1e-3,
        representative: str = "max_abs",
        # selection of runs
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        order: Callable[[Key, Any], Any] | None = None,
        # grouping tags (added as columns)
        group_by: tuple[str, ...] | None = ("number", "letter"),
    ) -> pd.DataFrame:
        """
        Collect per-run interstory drift envelope tables into one tidy DataFrame.

        This method ONLY orchestrates:
        - selects runs
        - calls nr.interstory_drift_envelope_pd(...)
        - appends metadata columns (model/station/rupture + tags + group key)

        Returns
        -------
        Tidy DataFrame with columns like:
        model, station, rupture, run_label,
        number, letter, sta, rup,
        group,
        z_lower, z_upper, dz,
        max_drift, min_drift, max_abs_drift, representative_drift,
        lower_node, upper_node
        """
        # validate representative
        if representative not in ("max_abs", "max", "min"):
            raise ValueError("representative must be 'max_abs', 'max', or 'min'.")

        # group spec normalize (we only want group_by normalization here)
        group_by, _cb, _st, _cbg = self._normalize_grouping_spec(group_by=group_by, color_by=None, stat=None)

        # select runs
        pairs = self.select(model=model, station=station, rupture=rupture, order=order)
        if not pairs:
            raise ValueError("No matching results for the given selection.")
        pairs = sorted(pairs, key=lambda kv: kv[0])

        # IMPORTANT: NodalResults requires exactly one of the node selectors
        provided = sum(x is not None for x in (selection_set_id, selection_set_name, node_ids, coordinates))
        if provided != 1:
            raise ValueError(
                "Provide exactly ONE of: selection_set_id, selection_set_name, node_ids, coordinates."
            )

        frames: list[pd.DataFrame] = []

        for k, nr in pairs:
            env = nr.interstory_drift_envelope_pd(
                component=component,
                selection_set_name=selection_set_name,
                selection_set_id=selection_set_id,
                node_ids=node_ids,
                coordinates=coordinates,
                result_name=result_name,
                stage=stage,
                dz_tol=dz_tol,
                representative=representative,
            ).copy()

            # metadata
            m, s, r = k
            tags = self._tag_from_key(k)
            gk = self._group_key(k, group_by)
            group = "|".join(gk) if group_by is not None else "ALL"

            env["model"] = m
            env["station"] = s
            env["rupture"] = r
            env["run_label"] = self._label_for(k, nr)

            for kk, vv in tags.items():
                env[kk] = vv

            env["group"] = group

            # handy derived columns
            env["story_z_mid"] = 0.5 * (env["z_lower"].astype(float) + env["z_upper"].astype(float))

            frames.append(env)

        if not frames:
            raise ValueError("No drift envelope tables were produced.")

        out = pd.concat(frames, ignore_index=True)
        # nice ordering
        cols_first = ["group", "model", "station", "rupture", "run_label", "number", "letter", "sta", "rup"]
        cols_first = [c for c in cols_first if c in out.columns]
        cols_rest = [c for c in out.columns if c not in cols_first]
        out = out[cols_first + cols_rest]
        return out

    def collect_roof_drift_df(
        self,
        *,
        top: tuple[float, float, float],
        bottom: tuple[float, float, float],
        components: tuple[int, int] = (1, 2),          # (X, Y)
        result_name: str = "DISPLACEMENT",
        relative_drift: bool = True,                  # True -> drift ratio ( /dz ) via nr.drift
        reduce_time: str = "abs_max",                 # "abs_max" | "max" | "min" | "rms"
        stage: str | None = None,                     # optional (if your nr.drift supports it)
        # run selection
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        order: Callable[[Key, Any], Any] | None = None,
        # output options
        add_log: bool = True,
        eps_log: float = 1e-16,
        include_label: bool = False,
    ) -> pd.DataFrame:
        """
        Build a tidy DataFrame for statistical modeling of roof drift.

        One row per (run, direction):
            model, station, rupture, Tier, Case, direction, roof_drift

        Notes
        -----
        - If relative_drift=True: uses nr.drift(...) (expected to return drift ratio if your NodalResults does that).
        - If relative_drift=False: uses nr.fetch(...) and computes Δu_top - Δu_bottom (no /dz).
        - reduce_time controls how the time series is collapsed to a scalar.
        """

        if reduce_time not in ("abs_max", "max", "min", "rms"):
            raise ValueError("reduce_time must be one of: 'abs_max', 'max', 'min', 'rms'.")

        cx, cy = int(components[0]), int(components[1])

        # -------------------------
        # Helpers
        # -------------------------
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

        def _delta_u_series(nr: Any, *, comp: int) -> pd.Series:
            top_id = int(nr.info.nearest_node_id([top], return_distance=False)[0])
            bot_id = int(nr.info.nearest_node_id([bottom], return_distance=False)[0])

            s = nr.fetch(result_name=result_name, component=comp, node_ids=[top_id, bot_id])

            # multi-stage handling (match your plotting logic)
            if isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 3:
                stages = tuple(sorted({str(x) for x in s.index.get_level_values(0)}))
                if not stages:
                    return pd.Series(dtype=float)
                # if user provided stage and it's present, use it; else last stage
                if stage is not None and stage in stages:
                    s = s.xs(stage, level=0)
                else:
                    s = s.xs(stages[-1], level=0)

            if not (isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 2):
                return pd.Series(dtype=float)

            u_top = s.xs(top_id, level=0).sort_index()
            u_bot = s.xs(bot_id, level=0).sort_index()
            u_top, u_bot = u_top.align(u_bot, join="inner")
            du = u_top - u_bot
            du.name = f"delta_u({result_name}:{comp})"
            return du

        def _roof_scalar(nr: Any, *, comp: int) -> float:
            if relative_drift:
                # try to pass stage/result_name if your nr.drift supports it; otherwise fallback
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
                    # older signature
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
                y = np.asarray(getattr(s, "to_numpy", lambda **_: np.asarray(s)) (dtype=float), dtype=float)
                return _reduce(y)

            # relative_drift=False: scalar from Δu series
            du = _delta_u_series(nr, comp=comp)
            if du.empty:
                return float("nan")
            return _reduce(du.to_numpy(dtype=float))

        # -------------------------
        # Build table via compute_table
        # -------------------------
        metrics = {
            "roof_drift_X": (lambda _k, nr: _roof_scalar(nr, comp=cx)),
            "roof_drift_Y": (lambda _k, nr: _roof_scalar(nr, comp=cy)),
        }

        T = self.compute_table(
            metrics=metrics,
            model=model,
            station=station,
            rupture=rupture,
            order=order,
            include_label=include_label,
            drop_na_rows=False,
        )

        if T.empty:
            return T

        # Tier/Case
        tiers, cases = [], []
        for m in T["model"].astype(str).tolist():
            t, c = self.parse_tier_letter(m)
            tiers.append(t)
            cases.append(c)

        T = T.copy()
        T["Tier"] = np.asarray(tiers, dtype=int)
        T["Case"] = pd.Series(cases, dtype="category")
        T["TierCase"] = T["Tier"].astype(str) + T["Case"].astype(str)

        # Long format: direction factor
        id_vars = ["model", "station", "rupture", "Tier", "Case", "TierCase"]
        if include_label and "label" in T.columns:
            id_vars.append("label")

        L = T.melt(
            id_vars=id_vars,
            value_vars=["roof_drift_X", "roof_drift_Y"],
            var_name="direction",
            value_name="roof_drift",
        )

        L["direction"] = L["direction"].map({"roof_drift_X": "X", "roof_drift_Y": "Y"}).astype("category")
        L["station"] = L["station"].astype("category")
        L["rupture"] = L["rupture"].astype("category")

        # Handy run id (ignores direction)
        L["run_key"] = (
            L["model"].astype(str) + "|" + L["station"].astype(str) + "|" + L["rupture"].astype(str)
        )

        # Optional log response (common for drift-like metrics)
        if add_log:
            y = L["roof_drift"].to_numpy(dtype=float)
            L["log_roof_drift"] = np.log(np.maximum(np.abs(y), float(eps_log)))

        # Nice column order
        first = ["run_key", "model", "Tier", "Case", "TierCase", "station", "rupture", "direction", "roof_drift"]
        if add_log:
            first.append("log_roof_drift")
        if include_label and "label" in L.columns:
            first.insert(1, "label")

        cols = first + [c for c in L.columns if c not in first]
        return L[cols]

    def drift_df(
        self,
        *,
        top: tuple[float, float, float],
        bottom: tuple[float, float, float],
        components: Sequence[int] = (1, 2),
        result_name: str = "DISPLACEMENT",
        relative_drift: bool = True,          # True -> drift (/dz), False -> Δu
        reduce_time: str = "abs_max",         # "abs_max" | "max" | "min" | "rms"
        stage: str | None = None,

        # selection
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        order: Callable[[tuple[str, str, str], Any], Any] | None = None,
    ) -> pd.DataFrame:
        """
        Minimal wide DataFrame:

            Tier | Case | sta | rup | {result_name}_c{comp} ...

        One row per (model, station, rupture).
        """

        if reduce_time not in ("abs_max", "max", "min", "rms"):
            raise ValueError("reduce_time must be one of: 'abs_max', 'max', 'min', 'rms'.")

        comps = tuple(int(c) for c in components)
        if not comps:
            raise ValueError("components must be non-empty.")

        # -------------------------------------------------
        # helpers
        # -------------------------------------------------
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

        def _select_stage(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
            if isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 3:
                stages = tuple(sorted({str(x) for x in s.index.get_level_values(0)}))
                if not stages:
                    return s.iloc[0:0]
                if stage is not None and stage in stages:
                    return s.xs(stage, level=0)
                return s.xs(stages[-1], level=0)
            return s

        def _delta_u_series(nr: Any, *, comp: int, top_id: int, bot_id: int) -> pd.Series:
            s = nr.fetch(result_name=result_name, component=comp, node_ids=[top_id, bot_id])
            s = _select_stage(s)

            if not (isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 2):
                return pd.Series(dtype=float)

            u_top = s.xs(top_id, level=0).sort_index()
            u_bot = s.xs(bot_id, level=0).sort_index()
            u_top, u_bot = u_top.align(u_bot, join="inner")
            return u_top - u_bot

        def _pair_scalar(nr: Any, *, comp: int, top_id: int, bot_id: int) -> float:
            if relative_drift:
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

                y = np.asarray(
                    getattr(s, "to_numpy", lambda **_: np.asarray(s))(dtype=float),
                    dtype=float,
                )
                return _reduce(y)

            du = _delta_u_series(nr, comp=comp, top_id=top_id, bot_id=bot_id)
            if du.empty:
                return float("nan")
            return _reduce(du.to_numpy(dtype=float))

        # -------------------------------------------------
        # collect
        # -------------------------------------------------
        pairs = self.select(model=model, station=station, rupture=rupture, order=order)

        cols = ["Tier", "Case", "sta", "rup"] + [f"{result_name}_c{c}" for c in comps]
        if not pairs:
            return pd.DataFrame(columns=cols)

        rows: list[dict[str, Any]] = []

        for k, nr in pairs:
            m, sta, rup = k
            tier, case = self.parse_tier_letter(m)

            row: dict[str, Any] = {
                "Tier": int(tier),
                "Case": str(case),
                "sta": str(sta),
                "rup": str(rup),
            }

            top_id = int(nr.info.nearest_node_id([top], return_distance=False)[0])
            bot_id = int(nr.info.nearest_node_id([bottom], return_distance=False)[0])

            for comp in comps:
                row[f"{result_name}_c{comp}"] = _pair_scalar(
                    nr, comp=comp, top_id=top_id, bot_id=bot_id
                )

            rows.append(row)

        df = pd.DataFrame(rows)

        df["Tier"] = df["Tier"].astype(int)
        df["Case"] = df["Case"].astype("category")
        df["sta"] = df["sta"].astype("category")
        df["rup"] = df["rup"].astype("category")

        return df[cols]

    def plot_interstory_drift_histograms(
        self,
        *,
        component: object,
        selection_set_name: str | None = "CenterPoints",
        selection_set_id: int | Sequence[int] | None = None,
        node_ids: Sequence[int] | None = None,
        coordinates: Sequence[Sequence[float]] | None = None,
        result_name: str = "DISPLACEMENT",
        stage: str | None = None,
        dz_tol: float = 1e-3,
        representative: str = "max_abs",
        # story selection for histogram
        story_index: int | None = None,      # 0-based within each run (sorted by z_mid)
        story_z_mid: float | None = None,    # match by tolerance across all runs
        z_tol: float = 1e-6,
        # which metric to histogram
        metric: str = "max_abs_drift",
        # grouping
        group_by: tuple[str, ...] | None = ("number", "letter"),
        # run selection
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        # histogram plot options
        bins: int | Sequence[float] = 30,
        density: bool = True,
        overlay: bool = True,
        figsize: tuple[float, float] = (8, 4.5),
        alpha: float = 0.35,
        linewidth: float = 1.5,
        title: str | None = None,
        legend_fontsize: float = 8,
        legend_ncol: int | None = None,
        legend_frameon: bool = False,
    ):
        """
        Plot histograms of a drift envelope metric by group, at a selected story.
        """
        valid_metric = {"max_abs_drift", "max_drift", "min_drift", "representative_drift"}
        if metric not in valid_metric:
            raise ValueError(f"metric must be one of {sorted(valid_metric)}")

        if (story_index is None) == (story_z_mid is None):
            raise ValueError("Provide exactly one of: story_index or story_z_mid.")

        T = self.collect_interstory_drift_envelope_pd(
            component=component,
            selection_set_name=selection_set_name,
            selection_set_id=selection_set_id,
            node_ids=node_ids,
            coordinates=coordinates,
            result_name=result_name,
            stage=stage,
            dz_tol=dz_tol,
            representative=representative,
            model=model,
            station=station,
            rupture=rupture,
            group_by=group_by,
        )

        if metric not in T.columns:
            raise ValueError(f"metric={metric!r} not found in collected table columns: {list(T.columns)}")

        # --- choose story rows ---
        if story_index is not None:
            si = int(story_index)
            if si < 0:
                raise ValueError("story_index must be >= 0.")

            # assign a per-run story index by z_mid order
            T = T.sort_values(["model", "station", "rupture", "story_z_mid"]).copy()
            T["_story_idx"] = T.groupby(["model", "station", "rupture"])["story_z_mid"].cumcount()

            S = T.loc[T["_story_idx"] == si].copy()
            if S.empty:
                mx = int(T["_story_idx"].max())
                raise ValueError(f"No rows for story_index={si}. Available 0..{mx}.")
            story_label = f"story_index={si}"

        else:
            z0 = float(story_z_mid)
            tol = float(z_tol)
            S = T.loc[np.abs(T["story_z_mid"].to_numpy(dtype=float) - z0) <= tol].copy()
            if S.empty:
                zvals = np.sort(np.unique(T["story_z_mid"].to_numpy(dtype=float)))
                near = zvals[np.argsort(np.abs(zvals - z0))[:5]]
                raise ValueError(
                    f"No rows for story_z_mid≈{z0} (tol={tol}). Nearest: {near.tolist()}"
                )
            story_label = f"story_z_mid≈{z0:g}"

        groups = {g: df for g, df in S.groupby("group", sort=True)}

        # --- plot ---
        if overlay:
            fig, ax = plt.subplots(figsize=figsize)

            for g in sorted(groups.keys()):
                x = groups[g][metric].to_numpy(dtype=float)
                x = x[np.isfinite(x)]
                if x.size == 0:
                    continue

                ax.hist(
                    x,
                    bins=bins,
                    density=density,
                    histtype="step",
                    linewidth=linewidth,
                    label=f"{g} (n={x.size})",
                )
                ax.hist(
                    x,
                    bins=bins,
                    density=density,
                    histtype="stepfilled",
                    alpha=alpha,
                    label="_nolegend_",
                )

            ax.set_xlabel(metric)
            ax.set_ylabel("Density" if density else "Count")
            ax.set_title(title or f"Histogram of {metric} — {story_label}")
            ax.grid(True, alpha=0.35)

            self._legend_below(fig, ax, fontsize=legend_fontsize, ncol=legend_ncol, frameon=legend_frameon)
            plt.tight_layout()
            return fig, ax

        figs: list[tuple[plt.Figure, plt.Axes]] = []
        for g in sorted(groups.keys()):
            x = groups[g][metric].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue

            fig, ax = plt.subplots(figsize=figsize)
            ax.hist(x, bins=bins, density=density, histtype="stepfilled", alpha=alpha, label=f"{g} (n={x.size})")
            ax.hist(x, bins=bins, density=density, histtype="step", linewidth=linewidth, label="_nolegend_")

            ax.set_xlabel(metric)
            ax.set_ylabel("Density" if density else "Count")
            ax.set_title(title or f"{metric} — {g} — {story_label}")
            ax.grid(True, alpha=0.35)
            ax.legend(frameon=legend_frameon, fontsize=legend_fontsize)

            plt.tight_layout()
            figs.append((fig, ax))

        if not figs:
            raise ValueError("No groups produced valid histogram data.")

        return figs

