"""PlotSettings — global plotting preferences (canonical home).

Replaces the ``@dataclass(slots=True)`` ``ModelPlotSettings`` per
refactor spec §4.2. The canonical class lives here; the legacy name
remains importable from :mod:`STKO_to_python.plotting.plot_dataclasses`
as an alias — ``ModelPlotSettings is PlotSettings``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class PlotSettings:
    """Global, model-level plotting preferences.

    These are *defaults* that plotting utilities can read and then
    optionally override per-plot call.

    Attributes
    ----------
    color, linewidth, linestyle, marker, alpha:
        Matplotlib Line2D defaults. ``None`` means "unset" (no kwarg
        emitted when building a Line2D kwargs dict).
    label_base:
        Base string used by :meth:`make_label` to build composite
        labels like ``"<label_base> <suffix>"``.
    """

    __slots__ = (
        "color",
        "linewidth",
        "linestyle",
        "label_base",
        "marker",
        "alpha",
    )

    def __init__(
        self,
        *,
        color: Optional[str] = None,
        linewidth: Optional[float] = None,
        linestyle: Optional[str] = None,
        label_base: Optional[str] = None,
        marker: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> None:
        self.color = color
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.label_base = label_base
        self.marker = marker
        self.alpha = alpha

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{name}={getattr(self, name)!r}"
            for name in self.__slots__
            if getattr(self, name) is not None
        )
        return f"PlotSettings({fields})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlotSettings):
            return NotImplemented
        return all(getattr(self, n) == getattr(other, n) for n in self.__slots__)

    # ------------------------------------------------------------------ #
    # Pickle — __slots__ class without __dict__ needs explicit state.
    # ------------------------------------------------------------------ #
    def __getstate__(self) -> dict:
        return {name: getattr(self, name) for name in self.__slots__}

    def __setstate__(self, state: dict) -> None:
        # Tolerant per spec §6 pickle policy.
        for name in self.__slots__:
            setattr(self, name, state.get(name))

    # ------------------------------------------------------------------ #
    # Matplotlib integration
    # ------------------------------------------------------------------ #
    def to_mpl_kwargs(self, **overrides: Any) -> Dict[str, Any]:
        """Build a dict of Matplotlib Line2D kwargs.

        ``None``-valued settings are omitted. ``overrides`` always win.
        """
        out: Dict[str, Any] = {}
        for name in ("color", "linewidth", "linestyle", "marker", "alpha"):
            v = getattr(self, name)
            if v is not None:
                out[name] = v
        out.update(overrides)
        return out

    def make_label(
        self,
        *,
        suffix: Optional[str] = None,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """Construct a label from ``label_base`` + optional ``suffix``.

        Rules:
        - if ``label_base`` is set and ``suffix`` is given → ``"base suffix"``
        - if only ``label_base`` is set → ``"base"``
        - if ``label_base`` is ``None`` → ``suffix`` or ``default``
        """
        if self.label_base is None:
            return suffix if suffix is not None else default
        if suffix is None:
            return self.label_base
        return f"{self.label_base} {suffix}"
