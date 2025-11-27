from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(slots=True)
class ModelPlotSettings:
    """
    Global, model-level plotting preferences.

    These are *defaults* that plotting utilities can read and then
    optionally override per-plot call.
    """
    color: Optional[str] = None
    linewidth: Optional[float] = None
    linestyle: Optional[str] = None
    label_base: Optional[str] = None

    # optional but handy
    marker: Optional[str] = None
    alpha: Optional[float] = None

    def to_mpl_kwargs(self, **overrides: Any) -> Dict[str, Any]:
        """
        Build a dict of Matplotlib Line2D kwargs from the stored settings,
        with optional overrides (overrides win).
        """
        out: Dict[str, Any] = {}

        if self.color is not None:
            out["color"] = self.color
        if self.linewidth is not None:
            out["linewidth"] = self.linewidth
        if self.linestyle is not None:
            out["linestyle"] = self.linestyle
        if self.marker is not None:
            out["marker"] = self.marker
        if self.alpha is not None:
            out["alpha"] = self.alpha

        # caller overrides always win
        out.update(overrides)
        return out

    def make_label(
        self,
        *,
        suffix: Optional[str] = None,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """
        Construct a label from the model's label_base and an optional suffix.

        Rules:
        - if label_base is set and suffix is given → "label_base suffix"
        - if only label_base is set → "label_base"
        - if label_base is None → suffix or default
        """
        if self.label_base is None:
            return suffix if suffix is not None else default

        if suffix is None:
            return self.label_base

        return f"{self.label_base} {suffix}"
