from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Sequence

import numpy as np


def _is_scalar_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _normalize_int_tuple(
    value: int | Sequence[int] | Sequence[Sequence[int]] | np.ndarray | None,
    *,
    field_name: str,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return (int(value),)

    arr = np.asarray(value, dtype=object)
    flat: list[int] = []

    if arr.ndim == 0:
        return (int(arr.item()),)

    for item in arr.reshape(-1).tolist():
        if isinstance(item, (list, tuple, np.ndarray)):
            sub = np.asarray(item, dtype=np.int64).reshape(-1)
            flat.extend(int(x) for x in sub.tolist())
        else:
            flat.append(int(item))

    if not flat:
        raise ValueError(f"{field_name} is empty.")
    return tuple(sorted(set(flat)))


def _normalize_name_tuple(
    value: str | Sequence[str] | None,
    *,
    field_name: str,
) -> tuple[str, ...] | None:
    if value is None:
        return None

    items: list[str] = []
    if isinstance(value, str):
        raw_items = [part.strip() for part in value.split(",")]
    else:
        raw_items = [str(part).strip() for part in value if part is not None]

    for item in raw_items:
        if item:
            items.append(item)

    if not items:
        raise ValueError(f"{field_name} is empty.")
    return tuple(items)


def _normalize_points(
    value: Sequence[Sequence[float]] | Sequence[float] | np.ndarray | None,
    *,
    field_name: str,
) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            if arr.size not in (2, 3):
                raise TypeError(f"{field_name} points must have length 2 or 3.")
            points = (tuple(float(x) for x in arr.tolist()),)
        elif arr.ndim == 2:
            if arr.shape[1] not in (2, 3):
                raise TypeError(f"{field_name} points must have length 2 or 3.")
            points = tuple(tuple(float(x) for x in row.tolist()) for row in arr)
        else:
            raise TypeError(f"{field_name} must be a sequence of (x,y) or (x,y,z) coordinates.")
    else:
        seq = list(value)
        if not seq:
            raise ValueError(f"{field_name} is empty.")

        first = seq[0]
        if _is_scalar_number(first):
            point = tuple(float(x) for x in seq)
            if len(point) not in (2, 3):
                raise TypeError(f"{field_name} points must have length 2 or 3.")
            points = (point,)
        else:
            normalized: list[tuple[float, ...]] = []
            for i, point in enumerate(seq):
                if not isinstance(point, (list, tuple, np.ndarray)):
                    raise TypeError(
                        f"{field_name}[{i}] must be a sequence of coordinates with length 2 or 3."
                    )
                coords = tuple(float(x) for x in point)
                if len(coords) not in (2, 3):
                    raise TypeError(
                        f"{field_name}[{i}] must have length 2 or 3. Got {len(coords)}."
                    )
                normalized.append(coords)
            points = tuple(normalized)

    dims = {len(point) for point in points}
    if len(dims) != 1:
        raise TypeError(f"{field_name} cannot mix 2D and 3D coordinates.")
    return points


@dataclass(frozen=True)
class SelectionBox:
    min_corner: Sequence[float]
    max_corner: Sequence[float]
    inclusive: bool = True

    def __post_init__(self) -> None:
        min_corner = tuple(float(x) for x in self.min_corner)
        max_corner = tuple(float(x) for x in self.max_corner)

        if len(min_corner) not in (2, 3) or len(max_corner) not in (2, 3):
            raise TypeError("SelectionBox corners must have length 2 or 3.")
        if len(min_corner) != len(max_corner):
            raise TypeError("SelectionBox corners must have the same dimensionality.")

        lo = tuple(min(a, b) for a, b in zip(min_corner, max_corner))
        hi = tuple(max(a, b) for a, b in zip(min_corner, max_corner))

        object.__setattr__(self, "min_corner", lo)
        object.__setattr__(self, "max_corner", hi)
        object.__setattr__(self, "inclusive", bool(self.inclusive))

    @property
    def ndim(self) -> int:
        return len(self.min_corner)


@dataclass(frozen=True)
class Selection:
    ids: tuple[int, ...] | None = None
    selection_set_id: tuple[int, ...] | None = None
    selection_set_name: tuple[str, ...] | None = None
    coordinates: tuple[tuple[float, ...], ...] | None = None
    box: SelectionBox | None = None
    file_id: int | None = None
    combine: str = "union"

    def __post_init__(self) -> None:
        ids = _normalize_int_tuple(self.ids, field_name="ids")
        selection_set_id = _normalize_int_tuple(
            self.selection_set_id, field_name="selection_set_id"
        )
        selection_set_name = _normalize_name_tuple(
            self.selection_set_name, field_name="selection_set_name"
        )
        coordinates = _normalize_points(self.coordinates, field_name="coordinates")
        box = self.box

        if box is not None and not isinstance(box, SelectionBox):
            if not isinstance(box, (list, tuple)) or len(box) != 2:
                raise TypeError("box must be a SelectionBox or a pair of corners.")
            box = SelectionBox(box[0], box[1])

        combine = str(self.combine).strip().lower()
        if combine not in {"union", "intersection"}:
            raise ValueError("combine must be 'union' or 'intersection'.")

        file_id = None if self.file_id is None else int(self.file_id)

        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "selection_set_id", selection_set_id)
        object.__setattr__(self, "selection_set_name", selection_set_name)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "box", box)
        object.__setattr__(self, "file_id", file_id)
        object.__setattr__(self, "combine", combine)

    def is_empty(self) -> bool:
        return (
            self.ids is None
            and self.selection_set_id is None
            and self.selection_set_name is None
            and self.coordinates is None
            and self.box is None
        )

    @classmethod
    def from_node_filters(
        cls,
        *,
        selection: Selection | None = None,
        node_ids: int | Sequence[int] | Sequence[Sequence[int]] | np.ndarray | None = None,
        selection_set_id: int | Sequence[int] | np.ndarray | None = None,
        selection_set_name: str | Sequence[str] | None = None,
        coordinates: Sequence[Sequence[float]] | Sequence[float] | np.ndarray | None = None,
        selection_box: SelectionBox | Sequence[Sequence[float]] | None = None,
        file_id: int | None = None,
        combine: str | None = None,
    ) -> Selection:
        if selection is not None and not isinstance(selection, cls):
            raise TypeError(
                f"selection must be a {cls.__name__} instance or None. Got {type(selection)!r}."
            )

        extra = cls(
            ids=node_ids,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            coordinates=coordinates,
            box=selection_box,
            file_id=file_id,
            combine=combine or (selection.combine if selection is not None else "union"),
        )

        if selection is None:
            return extra

        if selection.file_id is not None and extra.file_id is not None and selection.file_id != extra.file_id:
            raise ValueError(
                f"Conflicting file_id values: {selection.file_id} and {extra.file_id}."
            )

        box = selection.box or extra.box
        if selection.box is not None and extra.box is not None and selection.box != extra.box:
            raise ValueError("Multiple selection boxes were provided. Pass only one box.")

        return cls(
            ids=_merge_optional_tuple(selection.ids, extra.ids),
            selection_set_id=_merge_optional_tuple(
                selection.selection_set_id, extra.selection_set_id
            ),
            selection_set_name=_merge_optional_tuple(
                selection.selection_set_name, extra.selection_set_name
            ),
            coordinates=_merge_optional_tuple(selection.coordinates, extra.coordinates),
            box=box,
            file_id=selection.file_id if selection.file_id is not None else extra.file_id,
            combine=combine or selection.combine,
        )


def _merge_optional_tuple(
    left: tuple | None,
    right: tuple | None,
) -> tuple | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right
