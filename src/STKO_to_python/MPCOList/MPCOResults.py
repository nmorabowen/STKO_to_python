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
)

import fnmatch
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

Key = Tuple[str, str, str]  # (model, station, rupture)
GroupKey: TypeAlias = tuple[str, ...]


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

    _FNAME_PATTERN = re.compile(r"^(?P<model>.+?)__(?P<station>.+?)__(?P<rupture>.+?)\.pkl(\.gz)?$")
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
    ) -> "MPCOResults":
        out_dir = Path(out_dir)
        out: Dict[Key, Any] = {}

        # local import to avoid circular deps
        from ..results.nodal_results_dataclass import NodalResults  # adjust to your package path

        for p in out_dir.glob("*.pkl*"):
            m = cls._FNAME_PATTERN.match(p.name)
            if not m:
                continue
            key: Key = (m.group("model"), m.group("station"), m.group("rupture"))
            if key in out:
                raise ValueError(f"Duplicate key {key} from file {p.name!r}")
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
        m1 = re.search(r"([1-4])", s)
        m2 = re.search(r"([A-D])", s)
        if not m1 or not m2:
            raise ValueError(f"Could not parse tier/letter from model label: {model_label!r}")
        tier = int(m1.group(1))
        letter = m2.group(1)
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
        mnum = re.search(r"([1-4])", mm)
        mlet = re.search(r"([A-Da-d])\s*$", mm)
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
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        overlay: bool = True,
        figsize: tuple[float, float] = (10, 6),
        group_by_color: str | None = None,   # None | "sta" | "rup" | "letter" | "number"
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
        valid_groups = {None, "sta", "rup", "letter", "number"}
        if group_by_color not in valid_groups:
            raise ValueError(f"group_by_color must be one of {valid_groups}")

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
                m2 = re.search(r"([A-Da-d])\s*$", mm)
                return m2.group(1).upper() if m2 else ""
            if group_by_color == "number":
                mm = str(m)
                m2 = re.search(r"([1-4])", mm)
                return m2.group(1) if m2 else ""
            raise RuntimeError("unreachable")

        color_map: dict[Any, str] = {}
        if group_by_color is None:
            for i, (k, _) in enumerate(pairs):
                color_map[k] = palette[i % len(palette)]
        else:
            groups = []
            for k, _ in pairs:
                g = _group_value(k)
                if g not in groups:
                    groups.append(g)
            for i, g in enumerate(sorted(groups)):
                color_map[g] = palette[i % len(palette)]

        def plot_one(ax: plt.Axes, k: Key, nr: Any):
            drift = nr.drift(top=top, bottom=bottom, component=component)
            y = np.asarray(drift.values, float)
            t = np.asarray(nr.time, float)

            t2, y2, t_trim, y_trim = self._align_xy(t, y)
            if warn_mismatch and (t_trim or y_trim):
                print(f"[plot_drift] mismatch {k}: time={len(t)} drift={len(y)} → {len(t2)}")

            if t2.size == 0:
                return None

            label = self._label_for(k, nr)
            kw = self._style_for_key(k)
            kw["linewidth"] = linewidth

            if group_by_color is None:
                kw["color"] = color_map[k]
            else:
                kw["color"] = color_map[_group_value(k)]

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

        title = "Drift histories"
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
            ylabel="Drift",
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

                v = _reduce_window(yw / float(dz_i))
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

    def plot_roof_torsion(
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
    ):
        pairs = self.select(model=model, station=station, rupture=rupture, order=order)
        if not pairs:
            raise ValueError("No matching results for the given selection.")

        def plot_one(ax: plt.Axes, k: Key, nr: Any) -> tuple[float, float, float, float] | None:
            tors = nr.roof_torsion(
                node_a_coord=(*node_a_xy, float(z_coord)),
                node_b_coord=(*node_b_xy, float(z_coord)),
            )
            t = np.asarray(nr.time, dtype=float)
            y = np.asarray(tors.values, dtype=float)
            t2, y2, *_ = self._align_xy(t, y)
            if t2.size == 0:
                return None

            label = self._label_for(k, nr)
            kw = self._style_for_key(k)
            ax.plot(t2, y2, label=label, **kw)

            xmin, xmax = float(np.nanmin(t2)), float(np.nanmax(t2))
            ymin, ymax = float(np.nanmin(y2)), float(np.nanmax(y2))
            return (xmin, xmax, ymin, ymax)

        return self._plot_overlay_or_facets(
            pairs=pairs,
            plot_one=plot_one,
            overlay=overlay,
            figsize_overlay=figsize,
            figsize_single=figsize,
            title=title or "Roof torsion",
            xlabel="Time (s)",
            ylabel="Roof torsion (rad)",
            xlim=xlim,
            ylim=ylim,
            sym_x=False,
            sym_y=True,
            vline0=False,
        )

    def plot_base_rocking(
        self,
        *,
        z_coord: float,
        node_xy: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
            (5750, 5750),
            (38250, 5750),
            (5750, 25250),
        ),
        component: str = "theta_x_rad",
        model: str | Iterable[str] | None = None,
        station: str | Iterable[str] | None = None,
        rupture: str | Iterable[str] | None = None,
        overlay: bool = True,
        figsize: tuple[float, float] = (6, 4),
        title: str | None = None,
        order: Callable[[Key, Any], Any] | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ):
        pairs = self.select(model=model, station=station, rupture=rupture, order=order)
        if not pairs:
            raise ValueError("No matching results for the given selection.")

        def plot_one(ax: plt.Axes, k: Key, nr: Any) -> tuple[float, float, float, float] | None:
            df = nr.base_rocking(
                node_coords_xy=node_xy,
                z_coord=float(z_coord),
                result_name="DISPLACEMENT",
                uz_component=3,
                reduce="series",
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
            ax.plot(x2, y2, label=label, **kw)

            xmin, xmax = float(np.nanmin(x2)), float(np.nanmax(x2))
            ymin, ymax = float(np.nanmin(y2)), float(np.nanmax(y2))
            return (xmin, xmax, ymin, ymax)

        return self._plot_overlay_or_facets(
            pairs=pairs,
            plot_one=plot_one,
            overlay=overlay,
            figsize_overlay=figsize,
            figsize_single=figsize,
            title=title or f"Base rocking — {component}",
            xlabel="Time (s)",
            ylabel=f"Base rocking ({component}) [rad]",
            xlim=xlim,
            ylim=ylim,
            sym_x=False,
            sym_y=True,
            vline0=False,
        )

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

        mat = (
            T.groupby(["Case", "Tier"])[metric]
            .agg(agg)
            .unstack("Tier")
            .reindex(index=list("ABCD"), columns=[1, 2, 3, 4])
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
    ):
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

        gb = T.groupby(["Case", "Tier"])[metric]

        if agg == "mean":
            mat_mean = gb.mean().unstack("Tier").reindex(index=list("ABCD"), columns=[1, 2, 3, 4])
            mat_std = gb.std(ddof=1).unstack("Tier").reindex(index=list("ABCD"), columns=[1, 2, 3, 4])
            Z = mat_mean.to_numpy(dtype=float)
        elif agg == "median":
            mat_mean = gb.median().unstack("Tier").reindex(index=list("ABCD"), columns=[1, 2, 3, 4])
            mat_std = None
            Z = mat_mean.to_numpy(dtype=float)
        elif agg == "max":
            mat_mean = gb.max().unstack("Tier").reindex(index=list("ABCD"), columns=[1, 2, 3, 4])
            mat_std = None
            Z = mat_mean.to_numpy(dtype=float)
        elif agg == "min":
            mat_mean = gb.min().unstack("Tier").reindex(index=list("ABCD"), columns=[1, 2, 3, 4])
            mat_std = None
            Z = mat_mean.to_numpy(dtype=float)
        elif agg == "sum":
            mat_mean = gb.sum().unstack("Tier").reindex(index=list("ABCD"), columns=[1, 2, 3, 4])
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
                    cbar_kws={"label": metric},
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
            cbar.set_label(metric)

            ax.set_xlabel("Tier")
            ax.set_ylabel("Case")

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
    ):
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

        fig, ax = plt.subplots(figsize=figsize)

        y = np.arange(len(A))
        vals = A["value"].to_numpy(dtype=float)
        stds = A["std"].to_numpy(dtype=float)
        errs = A["err"].to_numpy(dtype=float)

        ax.barh(y, vals, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(A["Label"])

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

            ax.text(x_text, i, txt, va="center", ha="left", fontsize=9, zorder=4)

        ax.set_xlabel(metric)
        ax.set_ylabel("Tier–Case")

        if title is None:
            if agg == "mean" and (show_std_errorbar or show_std_text):
                title = f"{metric} per Tier–Case (mean ± {std_k:g}σ)"
            else:
                title = f"{metric} per Tier–Case ({agg})"
        ax.set_title(title)

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
