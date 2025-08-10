from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class MetaData:
    date_created: datetime = field(default_factory=datetime.utcnow)
    _extras: dict[str, Any] = field(default_factory=dict, repr=False)

    def __getattr__(self, name):
        if name in self._extras:
            return self._extras[name]
        raise AttributeError(f"{name} not found")

    def __setattr__(self, name, value):
        if name in self.__dataclass_fields__:
            super().__setattr__(name, value)
        else:
            self._extras[name] = value

    # ---- helper methods ----
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
        """Return extra attribute keys."""
        return self._extras.keys()

    def values(self):
        """Return extra attribute values."""
        return self._extras.values()

    def items(self):
        """Return extra attribute items."""
        return self._extras.items()

    def to_dict(self, include_date=True) -> dict[str, Any]:
        """Export all metadata to a plain dict."""
        base = {}
        if include_date:
            base["date_created"] = self.date_created
        base.update(self._extras)
        return base
