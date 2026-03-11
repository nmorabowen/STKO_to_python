from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .selection import (
    SelectionBox,
    _merge_optional_tuple,
    _normalize_int_tuple,
    _normalize_name_tuple,
)


@dataclass(frozen=True)
class ElementSelection:
    ids: tuple[int, ...] | None = None
    selection_set_id: tuple[int, ...] | None = None
    selection_set_name: tuple[str, ...] | None = None
    box: SelectionBox | None = None
    file_id: int | None = None
    element_type: tuple[str, ...] | None = None
    combine: str = "union"

    def __post_init__(self) -> None:
        ids = _normalize_int_tuple(self.ids, field_name="ids")
        selection_set_id = _normalize_int_tuple(
            self.selection_set_id, field_name="selection_set_id"
        )
        selection_set_name = _normalize_name_tuple(
            self.selection_set_name, field_name="selection_set_name"
        )
        element_type = _normalize_name_tuple(
            self.element_type, field_name="element_type"
        )
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
        object.__setattr__(self, "box", box)
        object.__setattr__(self, "file_id", file_id)
        object.__setattr__(self, "element_type", element_type)
        object.__setattr__(self, "combine", combine)

    def is_empty(self) -> bool:
        return (
            self.ids is None
            and self.selection_set_id is None
            and self.selection_set_name is None
            and self.box is None
            and self.element_type is None
        )

    @classmethod
    def from_element_filters(
        cls,
        *,
        selection: ElementSelection | None = None,
        element_ids: int | Sequence[int] | None = None,
        selection_set_id: int | Sequence[int] | None = None,
        selection_set_name: str | Sequence[str] | None = None,
        selection_box: SelectionBox | Sequence[Sequence[float]] | None = None,
        file_id: int | None = None,
        element_type: str | Sequence[str] | None = None,
        combine: str | None = None,
    ) -> ElementSelection:
        if selection is not None and not isinstance(selection, cls):
            raise TypeError(
                f"selection must be a {cls.__name__} instance or None. Got {type(selection)!r}."
            )

        extra = cls(
            ids=element_ids,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            box=selection_box,
            file_id=file_id,
            element_type=element_type,
            combine=combine or (selection.combine if selection is not None else "union"),
        )

        if selection is None:
            return extra

        if (
            selection.file_id is not None
            and extra.file_id is not None
            and selection.file_id != extra.file_id
        ):
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
            box=box,
            file_id=selection.file_id if selection.file_id is not None else extra.file_id,
            element_type=_merge_optional_tuple(selection.element_type, extra.element_type),
            combine=combine or selection.combine,
        )
