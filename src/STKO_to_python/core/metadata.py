"""ModelMetadata — free-form metadata bag attached to a dataset.

Replaces the ``@dataclass``-flavored ``MetaData`` in ``core/dataclasses.py``
per refactor spec §4.1. The dataclass frame was only ever needed for its
default-factory support (``date_created``) and its field-vs-extras
separation; a plain class with ``__slots__`` expresses the same intent
without the implicit complexity.

Both names remain importable:
    from STKO_to_python.core.metadata import ModelMetadata   # canonical
    from STKO_to_python.core.dataclasses import MetaData    # back-compat

They are the **same class object** — ``MetaData is ModelMetadata``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable


class ModelMetadata:
    """Free-form metadata bag.

    Arbitrary keys are allowed and stored in an internal dict so that
    new fields can be added over time without touching pickle layouts
    or forcing migrations.

    Attributes are routed through ``_extras``:
        - Reads fall back from the instance to ``_extras`` via
          ``__getattr__``.
        - Writes always land in ``_extras`` via ``__setattr__``.
    This is the same contract as the old ``MetaData`` class.

    Thread-safety
    -------------
    Read-mostly. Concurrent writers should serialize externally.
    """

    __slots__ = ("_extras",)

    def __init__(self, **extras: Any) -> None:
        # ``__setattr__`` below routes into ``_extras``; bypass it once
        # during initial construction so the dict itself can be created.
        object.__setattr__(
            self,
            "_extras",
            {"date_created": datetime.now(timezone.utc), **extras},
        )

    # ------------------------------------------------------------------ #
    # Attribute access
    # ------------------------------------------------------------------ #
    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails; the slot ``_extras`` is
        # already resolved via ``object.__getattribute__``. This is the
        # fallback for anything else.
        try:
            return self._extras[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        # Route all writes through ``_extras`` so the slot stays pristine
        # (the only allowed attribute). The one exception is ``_extras``
        # itself, which ``__init__`` sets via ``object.__setattr__``.
        if name == "_extras":
            object.__setattr__(self, name, value)
            return
        self._extras[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self._extras[name]
        except KeyError:
            raise AttributeError(name) from None

    def __contains__(self, name: str) -> bool:
        return name in self._extras

    def __iter__(self) -> Iterable[str]:
        return iter(self._extras)

    def __len__(self) -> int:
        return len(self._extras)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self._extras.items())
        return f"ModelMetadata({items})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelMetadata):
            return NotImplemented
        return self._extras == other._extras

    # ------------------------------------------------------------------ #
    # Pickle — __slots__ class without __dict__ needs explicit state.
    # ------------------------------------------------------------------ #
    def __getstate__(self) -> dict:
        return {"_extras": dict(self._extras)}

    def __setstate__(self, state: dict) -> None:
        # Tolerant: drop unknown keys silently (spec §6 pickle policy).
        object.__setattr__(self, "_extras", dict(state.get("_extras", {})))

    # ------------------------------------------------------------------ #
    # Dict-like helpers (preserve the legacy ``MetaData`` API)
    # ------------------------------------------------------------------ #
    def set(self, key: str, value: Any) -> None:
        """Set an extra attribute."""
        self._extras[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get an extra attribute (returns default if missing)."""
        return self._extras.get(key, default)

    def has(self, key: str) -> bool:
        """Check if an extra attribute exists."""
        return key in self._extras

    def keys(self):
        return self._extras.keys()

    def values(self):
        return self._extras.values()

    def items(self):
        return self._extras.items()

    def to_dict(self, include_date: bool = True) -> dict[str, Any]:
        """Export metadata to a plain dict.

        Parameters
        ----------
        include_date:
            If ``False``, omit the auto-populated ``date_created`` key.
        """
        out = dict(self._extras)
        if not include_date:
            out.pop("date_created", None)
        return out

    # Alias for the new spec-aligned name.
    as_dict = to_dict
