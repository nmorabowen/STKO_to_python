"""Phase 0 context-manager contract for ``MPCODataSet``.

The Phase 1 partition pool will hold HDF5 handles; users must be able to
release them deterministically via ``with MPCODataSet(...) as ds:``. The
Phase 0 stub is a no-op — but the API surface has to be in place now so
downstream code can adopt the idiom before the pool lands.

These tests do not construct a full dataset (that would require a real
.mpco file). They verify the contract on the class itself.
"""
from __future__ import annotations

import inspect

from STKO_to_python.core.dataset import MPCODataSet


def test_dataset_defines_enter() -> None:
    assert hasattr(MPCODataSet, "__enter__"), (
        "MPCODataSet must implement __enter__ so `with MPCODataSet(...) as ds:` works"
    )


def test_dataset_defines_exit() -> None:
    assert hasattr(MPCODataSet, "__exit__"), (
        "MPCODataSet must implement __exit__ so the with-block releases resources"
    )


def test_enter_returns_self_by_contract() -> None:
    """Calling __enter__ on an instance returns the instance itself.

    We bypass __init__ with ``__new__`` so this test does not need a real
    .mpco file.
    """
    ds = MPCODataSet.__new__(MPCODataSet)
    assert ds.__enter__() is ds


def test_exit_is_noop_and_does_not_swallow() -> None:
    """Phase 0 __exit__ returns None — exceptions propagate unchanged."""
    ds = MPCODataSet.__new__(MPCODataSet)
    result = ds.__exit__(None, None, None)
    assert result is None, (
        "__exit__ must return None/falsy so raised exceptions propagate"
    )


def test_exit_signature_matches_protocol() -> None:
    """``__exit__(self, exc_type, exc_val, exc_tb)`` — the 4-arg CM protocol."""
    sig = inspect.signature(MPCODataSet.__exit__)
    params = list(sig.parameters)
    assert params == ["self", "exc_type", "exc_val", "exc_tb"], (
        f"MPCODataSet.__exit__ has unexpected signature: {params}"
    )
