from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.element_selection import ElementSelection
from ..core.selection import SelectionBox


class ElementResultsInfo:
    __slots__ = (
        "elements_ids",
        "elements_info",
        "model_stages",
        "results_components",
        "selection_set",
        "analysis_time",
        "size",
        "name",
    )

    def __init__(
        self,
        *,
        elements_ids: Optional[tuple[int, ...]] = None,
        elements_info: Optional[pd.DataFrame] = None,
        model_stages: Optional[tuple[str, ...]] = None,
        results_components: Optional[tuple[str, ...]] = None,
        selection_set: Optional[dict] = None,
        analysis_time: Optional[float] = None,
        size: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if elements_ids is not None:
            elements_ids = tuple(int(i) for i in elements_ids)
        if model_stages is not None:
            model_stages = tuple(str(s) for s in model_stages)
        if results_components is not None:
            results_components = tuple(str(c) for c in results_components)
        if elements_info is not None and not isinstance(elements_info, pd.DataFrame):
            raise TypeError(
                "elements_info must be a pandas DataFrame "
                f"(got {type(elements_info)!r})."
            )
        if isinstance(elements_info, pd.DataFrame) and elements_info.index.name is None:
            elements_info = elements_info.rename_axis("element_id")

        self.elements_ids = elements_ids
        self.elements_info = elements_info
        self.model_stages = model_stages
        self.results_components = results_components
        self.selection_set = selection_set
        self.analysis_time = analysis_time
        self.size = size
        self.name = name

    def available_element_ids(self) -> np.ndarray:
        if self.elements_ids is not None:
            out = np.asarray(self.elements_ids, dtype=np.int64)
            if out.size:
                return np.unique(out)
        if self.elements_info is None:
            return np.empty((0,), dtype=np.int64)
        return np.unique(self._element_ids())

    def resolve_selection(
        self,
        selection: ElementSelection | None = None,
        *,
        only_available: bool = True,
    ) -> list[int]:
        selection = selection or ElementSelection()
        if not isinstance(selection, ElementSelection):
            raise TypeError(
                f"selection must be an ElementSelection instance or None. Got {type(selection)!r}."
            )

        sources: list[np.ndarray] = []
        if selection.ids is not None:
            sources.append(np.asarray(selection.ids, dtype=np.int64))
        if selection.selection_set_id is not None:
            sources.append(
                np.asarray(
                    self.selection_set_element_ids(
                        selection.selection_set_id,
                        only_available=only_available,
                    ),
                    dtype=np.int64,
                )
            )
        if selection.selection_set_name is not None:
            sources.append(
                np.asarray(
                    self.selection_set_element_ids_by_name(
                        selection.selection_set_name,
                        only_available=only_available,
                    ),
                    dtype=np.int64,
                )
            )
        if selection.box is not None:
            sources.append(
                np.asarray(
                    self.element_ids_in_box(selection.box, file_id=selection.file_id),
                    dtype=np.int64,
                )
            )

        if not sources:
            out = self.available_element_ids()
        elif selection.combine == "union":
            out = np.unique(np.concatenate([np.unique(src) for src in sources]))
        else:
            out = np.unique(sources[0])
            for src in sources[1:]:
                out = np.intersect1d(out, np.unique(src), assume_unique=False)

        if only_available and self.elements_ids is not None:
            out = np.intersect1d(
                out,
                np.asarray(self.elements_ids, dtype=np.int64),
                assume_unique=False,
            )

        if selection.file_id is not None:
            out = self._filter_ids_by_file_id(out, selection.file_id)
        if selection.element_type is not None:
            out = self._filter_ids_by_element_type(out, selection.element_type)

        if out.size == 0:
            raise ValueError("Resolved element set is empty.")
        return [int(v) for v in out.tolist()]

    def element_ids_in_box(
        self,
        box: SelectionBox,
        *,
        file_id: Optional[int] = None,
    ) -> list[int]:
        df = self._require_elements_info()
        dims = box.ndim
        cols = ["centroid_x", "centroid_y"] if dims == 2 else ["centroid_x", "centroid_y", "centroid_z"]
        for col in cols:
            self._resolve_column(df, col, required=True)

        sub = df
        if file_id is not None:
            file_col = self._file_column(df)
            sub = sub.loc[sub[file_col].to_numpy(dtype=np.int64, copy=False) == int(file_id)]
            if sub.empty:
                raise ValueError(f"No elements found for file_id={int(file_id)}.")

        lo = np.asarray(box.min_corner, dtype=float)
        hi = np.asarray(box.max_corner, dtype=float)
        coords = sub[cols].to_numpy(dtype=float, copy=False)
        if box.inclusive:
            mask = np.all((coords >= lo) & (coords <= hi), axis=1)
        else:
            mask = np.all((coords > lo) & (coords < hi), axis=1)

        out = sub.loc[mask, self._element_id_column(sub)].to_numpy(dtype=np.int64, copy=False)
        if out.size == 0:
            raise ValueError(
                f"No elements found inside selection box {box.min_corner} -> {box.max_corner}."
            )
        return [int(v) for v in np.unique(out).tolist()]

    @staticmethod
    def _normalize_selection_names(
        selection_set_name: str | Sequence[str] | None,
    ) -> Tuple[str, ...]:
        if selection_set_name is None:
            return ()
        if isinstance(selection_set_name, str):
            s = selection_set_name.strip()
            if "," in s:
                parts = [p.strip() for p in s.split(",") if p.strip()]
                return tuple(parts)
            return (s,)
        out: list[str] = []
        for x in selection_set_name:
            if x is None:
                continue
            sx = str(x).strip()
            if sx:
                out.append(sx)
        return tuple(out)

    def selection_set_ids_from_names(
        self,
        selection_set_name: str | Sequence[str],
    ) -> Tuple[int, ...]:
        if self.selection_set is None:
            raise ValueError("selection_set is None. No selection sets available.")
        names = self._normalize_selection_names(selection_set_name)
        if not names:
            raise ValueError("selection_set_name is empty.")

        resolved: list[int] = []
        for raw in names:
            sid_i, _ = self._resolve_selection_set_name(raw)
            resolved.append(sid_i)
        return tuple(resolved)

    def selection_set_element_ids_by_name(
        self,
        selection_set_name: str | Sequence[str],
        *,
        only_available: bool = True,
    ) -> list[int]:
        sids = self.selection_set_ids_from_names(selection_set_name)
        return self.selection_set_element_ids(sids, only_available=only_available)

    def selection_set_element_ids(
        self,
        selection_set_id: int | Sequence[int],
        *,
        only_available: bool = True,
    ) -> list[int]:
        if self.selection_set is None:
            raise ValueError("selection_set is None. No selection sets available.")

        ids = [selection_set_id] if isinstance(selection_set_id, int) else list(selection_set_id)
        if len(ids) == 0:
            raise ValueError("selection_set_id is empty.")

        gathered: list[np.ndarray] = []
        missing: list[int] = []
        for sid in ids:
            sid_i = int(sid)
            if sid_i not in self.selection_set:
                missing.append(sid_i)
                continue

            entry = self.selection_set.get(sid_i) or {}
            elements = entry.get("ELEMENTS")
            set_name = self._selection_set_name_for(sid_i) or str(sid_i)
            if elements is None or len(elements) == 0:
                raise ValueError(
                    f"Selection set '{set_name}' (id={sid_i}) contains 0 elements "
                    f"in the source .cdata."
                )
            gathered.append(np.asarray(elements, dtype=np.int64))

        if missing:
            raise ValueError(
                f"Selection set id(s) not found: {missing}. "
                f"Available ids: {sorted(map(int, self.selection_set.keys()))[:50]}"
            )

        out = np.unique(np.concatenate(gathered)).astype(np.int64, copy=False)
        if only_available and self.elements_ids is not None:
            out = np.intersect1d(
                out,
                np.asarray(self.elements_ids, dtype=np.int64),
                assume_unique=False,
            )
        if out.size == 0:
            raise ValueError(
                "Resolved selection set element ids are empty "
                "(possibly due to only_available=True filtering)."
            )
        return [int(v) for v in out.tolist()]

    def _selection_set_name_for(self, sid: int) -> str:
        if self.selection_set is None:
            return ""
        d = self.selection_set.get(int(sid), {})
        if not isinstance(d, dict):
            return ""
        for k in ("SET_NAME", "set_name", "NAME", "name", "Name"):
            v = d.get(k)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return ""

    def _resolve_selection_set_name(self, raw_name: str) -> tuple[int, str]:
        if self.selection_set is None:
            raise ValueError("selection_set is None. No selection sets available.")

        query = str(raw_name).strip()
        if not query:
            raise ValueError("selection_set_name is empty.")

        exact_hits: list[tuple[int, str]] = []
        folded_hits: list[tuple[int, str]] = []
        available: set[str] = set()

        for sid in self.selection_set.keys():
            try:
                sid_i = int(sid)
            except Exception:
                continue
            name = self._selection_set_name_for(sid_i)
            if not name:
                continue
            available.add(name)
            if name == query:
                exact_hits.append((sid_i, name))
            if name.lower() == query.lower():
                folded_hits.append((sid_i, name))

        if len(exact_hits) == 1:
            return exact_hits[0]
        if len(exact_hits) > 1:
            raise ValueError(
                f"Ambiguous selection set name {raw_name!r}: exact matches IDs "
                f"{sorted(sid for sid, _ in exact_hits)}. Use selection_set_id instead."
            )
        if len(folded_hits) == 1:
            return folded_hits[0]
        if len(folded_hits) > 1:
            raise ValueError(
                f"Ambiguous selection set name {raw_name!r}: case-insensitive matches IDs "
                f"{sorted(sid for sid, _ in folded_hits)}. Use selection_set_id instead."
            )
        preview = ", ".join(sorted(available)[:50]) + (" ..." if len(available) > 50 else "")
        raise ValueError(
            f"Selection set name not found: {raw_name!r}. "
            f"Available names include: {preview}"
        )

    @staticmethod
    def _norm_col(name: object) -> str:
        s = str(name).strip()
        if s.startswith("#"):
            s = s[1:].strip()
        return s.lower()

    def _normalized_columns(self, df: pd.DataFrame) -> dict[str, str]:
        return {self._norm_col(c): str(c) for c in df.columns}

    def _resolve_column(
        self,
        df: pd.DataFrame,
        key: str,
        *,
        required: bool = True,
    ) -> Optional[str]:
        cols = self._normalized_columns(df)
        k = key.lower()
        if k in cols:
            return cols[k]
        if required:
            raise ValueError(
                f"elements_info is missing required column '{key}'. "
                f"Available columns (normalized): {sorted(cols.keys())}"
            )
        return None

    def _require_elements_info(self) -> pd.DataFrame:
        if self.elements_info is None:
            raise ValueError("elements_info is None. Cannot resolve element selection.")
        return self.elements_info

    def _element_id_column(self, df: pd.DataFrame) -> str:
        col = self._resolve_column(df, "element_id", required=False)
        return col or str(df.index.name or "element_id")

    def _element_ids(self) -> np.ndarray:
        df = self._require_elements_info()
        col = self._resolve_column(df, "element_id", required=False)
        if col is not None:
            return df[col].to_numpy(dtype=np.int64, copy=False)
        return df.index.to_numpy(dtype=np.int64, copy=False)

    def _file_column(self, df: pd.DataFrame) -> str:
        file_col = self._resolve_column(df, "file_name", required=False)
        if file_col is not None:
            return file_col
        file_col = self._resolve_column(df, "file_id", required=False)
        if file_col is not None:
            return file_col
        raise ValueError("elements_info does not contain 'file_name'/'file_id'.")

    def _filter_ids_by_file_id(self, ids: np.ndarray, file_id: int) -> np.ndarray:
        df = self._require_elements_info()
        file_col = self._file_column(df)
        id_col = self._resolve_column(df, "element_id", required=False)
        sub = df.loc[df[file_col].to_numpy(dtype=np.int64, copy=False) == int(file_id)]
        if sub.empty:
            raise ValueError(f"No elements found for file_id={int(file_id)}.")
        available = (
            sub[id_col].to_numpy(dtype=np.int64, copy=False)
            if id_col is not None
            else sub.index.to_numpy(dtype=np.int64, copy=False)
        )
        return np.intersect1d(ids, np.unique(available), assume_unique=False)

    def _filter_ids_by_element_type(
        self,
        ids: np.ndarray,
        element_types: tuple[str, ...],
    ) -> np.ndarray:
        df = self._require_elements_info()
        type_col = self._resolve_column(df, "element_type", required=True)
        id_col = self._resolve_column(df, "element_id", required=False)

        requested = tuple(str(t).strip() for t in element_types if str(t).strip())
        if not requested:
            raise ValueError("element_type is empty.")

        def _matches(available: str) -> bool:
            available_s = str(available).strip()
            available_base = available_s.split("[")[0]
            return any(
                available_s == req
                or available_base == req
                or available_s.startswith(req + "[")
                for req in requested
            )

        mask = df[type_col].map(_matches).to_numpy(dtype=bool, copy=False)
        if not np.any(mask):
            raise ValueError(
                f"element_type not found: {requested}. "
                f"Available element types include: "
                f"{sorted(map(str, df[type_col].dropna().unique().tolist()))[:50]}"
            )

        matched = df.loc[mask]
        available = (
            matched[id_col].to_numpy(dtype=np.int64, copy=False)
            if id_col is not None
            else matched.index.to_numpy(dtype=np.int64, copy=False)
        )
        return np.intersect1d(ids, np.unique(available), assume_unique=False)

    def __repr__(self) -> str:
        n_elements = len(self.elements_ids) if self.elements_ids is not None else None
        stages = self.model_stages or ()
        comps = self.results_components or ()
        info_type = type(self.elements_info).__name__ if self.elements_info is not None else None
        return (
            "ElementResultsInfo("
            f"n_elements={n_elements}, "
            f"elements_info={info_type}, "
            f"model_stages={stages}, "
            f"results_components={comps}"
            ")"
        )
