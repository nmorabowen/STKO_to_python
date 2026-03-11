from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.element_selection import ElementSelection
from ..core.selection import SelectionBox
from .element_results_info import ElementResultsInfo


class ElementResults:
    def __init__(
        self,
        df: pd.DataFrame,
        time: Any,
        *,
        name: Optional[str],
        elements_ids: Optional[Tuple[int, ...]] = None,
        elements_info: Optional[pd.DataFrame] = None,
        results_components: Optional[Tuple[str, ...]] = None,
        model_stages: Optional[Tuple[str, ...]] = None,
        selection_set: Optional[dict] = None,
        analysis_time: Optional[float] = None,
        size: Optional[int] = None,
        plot_settings: Any = None,
    ) -> None:
        self.df = df
        self.time = time
        self.name = name
        self.info = ElementResultsInfo(
            elements_ids=elements_ids,
            elements_info=elements_info,
            model_stages=model_stages,
            results_components=results_components,
            selection_set=selection_set,
            analysis_time=analysis_time,
            size=size,
            name=name,
        )
        self.plot_settings = plot_settings

    def save_pickle(
        self,
        path: str | Path,
        *,
        compress: bool | None = None,
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> Path:
        p = Path(path)
        if compress is None:
            compress = p.suffix.lower() == ".gz"
        if compress:
            with gzip.open(p, "wb") as f:
                pickle.dump(self, f, protocol=protocol)
        else:
            with open(p, "wb") as f:
                pickle.dump(self, f, protocol=protocol)
        return p

    @classmethod
    def load_pickle(
        cls,
        path: str | Path,
        *,
        compress: bool | None = None,
    ) -> "ElementResults":
        p = Path(path)
        if compress is None:
            compress = p.suffix.lower() == ".gz"
        if compress:
            with gzip.open(p, "rb") as f:
                obj = pickle.load(f)
        else:
            with open(p, "rb") as f:
                obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Pickle at {p} is {type(obj)!r}, expected {cls.__name__}.")
        return obj

    def list_results(self) -> Tuple[str, ...]:
        cols = self.df.columns
        if isinstance(cols, pd.MultiIndex):
            names = sorted({str(level0) for (level0, _) in cols})
        else:
            names = sorted({str(c) for c in cols})
        return tuple(names)

    def list_components(self, result_name: Optional[str] = None) -> Tuple[str, ...]:
        cols = self.df.columns
        if isinstance(cols, pd.MultiIndex):
            if result_name is None:
                return tuple(sorted({str(c1) for (_, c1) in cols}))
            comps = {str(c1) for (c0, c1) in cols if str(c0) == str(result_name)}
            if not comps:
                raise ValueError(
                    f"Result '{result_name}' not found.\n"
                    f"Available result types: {self.list_results()}"
                )
            return tuple(sorted(comps))

        if result_name is not None:
            raise ValueError(
                "Single-level columns: do not pass result_name.\n"
                f"Available components: {tuple(map(str, cols))}"
            )
        return tuple(map(str, cols))

    def _normalize_selection(
        self,
        *,
        selection: ElementSelection | None = None,
        element_ids: int | Sequence[int] | None = None,
        selection_set_id: int | Sequence[int] | None = None,
        selection_set_name: str | Sequence[str] | None = None,
        selection_box: SelectionBox | Sequence[Sequence[float]] | None = None,
        element_type: str | Sequence[str] | None = None,
    ) -> ElementSelection:
        return ElementSelection.from_element_filters(
            selection=selection,
            element_ids=element_ids,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            selection_box=selection_box,
            element_type=element_type,
        )

    def _element_level(self, idx: pd.MultiIndex) -> int:
        names = list(idx.names) if idx.names is not None else [None] * idx.nlevels
        if "element_id" in names:
            return names.index("element_id")
        if idx.nlevels == 2:
            return 0
        if idx.nlevels == 3:
            return 1
        raise ValueError(
            "[fetch] Cannot infer element_id level. "
            f"Index nlevels={idx.nlevels}, names={names}."
        )

    def _filter_df_by_element_ids(
        self,
        df: pd.DataFrame,
        resolved_element_ids: Sequence[int],
    ) -> pd.DataFrame:
        idx = df.index
        if not isinstance(idx, pd.MultiIndex):
            raise ValueError(
                "[fetch] Expected a MultiIndex containing element_id. "
                f"Got index type={type(idx).__name__}."
            )
        element_ids_arr = np.asarray(resolved_element_ids, dtype=np.int64)
        lvl = idx.get_level_values(self._element_level(idx))
        out = df.loc[lvl.isin(element_ids_arr)]
        if out.empty:
            raise ValueError(
                f"[fetch] None of the requested element_ids are present. "
                f"Requested (sample): {element_ids_arr[:10].tolist()}"
            )
        return out

    def _subset_elements_info(
        self,
        resolved_element_ids: Sequence[int],
    ) -> Optional[pd.DataFrame]:
        ei = self.info.elements_info
        if ei is None:
            return None
        element_ids_arr = np.asarray(resolved_element_ids, dtype=np.int64)
        if "element_id" in ei.columns:
            out = ei.loc[ei["element_id"].isin(element_ids_arr)]
        else:
            out = ei.loc[ei.index.isin(element_ids_arr)]
        if out.empty:
            raise ValueError("Resolved element ids are not present in elements_info.")
        return out.copy()

    def fetch(
        self,
        result_name: Optional[str] = None,
        component: Optional[object] = None,
        *,
        element_ids: int | Sequence[int] | None = None,
        selection_set_id: int | Sequence[int] | None = None,
        selection_set_name: str | Sequence[str] | None = None,
        selection_box: SelectionBox | Sequence[Sequence[float]] | None = None,
        selection: ElementSelection | None = None,
        element_type: str | Sequence[str] | None = None,
        only_available: bool = True,
        return_elements: bool = False,
    ) -> pd.Series | pd.DataFrame | tuple[pd.Series | pd.DataFrame, list[int]]:
        df = self.df
        resolved_element_ids: list[int] = []

        selection_obj = self._normalize_selection(
            selection=selection,
            element_ids=element_ids,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            selection_box=selection_box,
            element_type=element_type,
        )
        if not selection_obj.is_empty():
            resolved_element_ids = self.info.resolve_selection(
                selection_obj,
                only_available=only_available,
            )
            df = self._filter_df_by_element_ids(df, resolved_element_ids)

        cols = df.columns

        def _ret(out: pd.Series | pd.DataFrame):
            if return_elements:
                return out, resolved_element_ids
            return out

        if isinstance(cols, pd.MultiIndex):
            if result_name is None:
                raise ValueError(
                    "result_name must be provided.\n"
                    f"Available results: {self.list_results()}"
                )
            if component is None:
                sub_cols = [c for c in cols if str(c[0]) == str(result_name)]
                if not sub_cols:
                    raise ValueError(f"No components found for result '{result_name}'.")
                return _ret(df.loc[:, sub_cols])

            for c0, c1 in cols:
                if str(c0) == str(result_name) and str(c1) == str(component):
                    return _ret(df.loc[:, (c0, c1)])
            raise ValueError(
                f"Component '{component}' not found for result '{result_name}'.\n"
                f"Available components: {self.list_components(result_name)}"
            )

        if result_name is not None:
            raise ValueError("Single-level columns: use fetch(component=...) only.")
        if component is None:
            return _ret(df)
        if component in cols:
            return _ret(df[component])
        comp_str = str(component)
        if comp_str in cols:
            return _ret(df[comp_str])
        raise ValueError(
            f"Component '{component}' not found.\n"
            f"Available components: {tuple(map(str, cols))}"
        )

    def resolve_element_ids(
        self,
        *,
        selection: ElementSelection | None = None,
        element_ids: int | Sequence[int] | None = None,
        selection_set_id: int | Sequence[int] | None = None,
        selection_set_name: str | Sequence[str] | None = None,
        selection_box: SelectionBox | Sequence[Sequence[float]] | None = None,
        element_type: str | Sequence[str] | None = None,
        only_available: bool = True,
    ) -> list[int]:
        selection_obj = self._normalize_selection(
            selection=selection,
            element_ids=element_ids,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            selection_box=selection_box,
            element_type=element_type,
        )
        return self.info.resolve_selection(selection_obj, only_available=only_available)

    def select(
        self,
        *,
        selection: ElementSelection | None = None,
        element_ids: int | Sequence[int] | None = None,
        selection_set_id: int | Sequence[int] | None = None,
        selection_set_name: str | Sequence[str] | None = None,
        selection_box: SelectionBox | Sequence[Sequence[float]] | None = None,
        element_type: str | Sequence[str] | None = None,
        only_available: bool = True,
    ) -> "ElementResults":
        resolved_element_ids = self.resolve_element_ids(
            selection=selection,
            element_ids=element_ids,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            selection_box=selection_box,
            element_type=element_type,
            only_available=only_available,
        )

        if (
            selection is None
            and element_ids is None
            and selection_set_id is None
            and selection_set_name is None
            and selection_box is None
            and element_type is None
        ):
            resolved_element_ids = (
                list(self.info.elements_ids)
                if self.info.elements_ids is not None
                else []
            )
            df = self.df
            elements_info = self.info.elements_info
        else:
            df = self._filter_df_by_element_ids(self.df, resolved_element_ids)
            elements_info = self._subset_elements_info(resolved_element_ids)

        return ElementResults(
            df=df.copy(),
            time=self.time,
            name=self.name,
            elements_ids=tuple(int(v) for v in resolved_element_ids) if resolved_element_ids else self.info.elements_ids,
            elements_info=elements_info.copy() if isinstance(elements_info, pd.DataFrame) else elements_info,
            results_components=self.info.results_components,
            model_stages=self.info.model_stages,
            selection_set=self.info.selection_set,
            analysis_time=self.info.analysis_time,
            size=self.info.size,
            plot_settings=self.plot_settings,
        )

    def __repr__(self) -> str:
        results = self.list_results()
        first = results[0] if results else None
        comps = self.list_components(first) if first is not None else ()
        stages = self.info.model_stages or ()
        return (
            f"ElementResults(name={self.name!r}, "
            f"results={results}, "
            f"components={comps}, "
            f"stages={stages})"
        )
