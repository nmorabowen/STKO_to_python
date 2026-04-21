"""Phase 4.3.1 skeleton tests for ``AggregationEngine``.

These tests pin down that the class exists, is importable from its
canonical location, declares ``__slots__``, and exposes the expected
engineering-method surface as ``NotImplementedError`` stubs. Real
behavior arrives in Phase 4.3.2, method by method.
"""
from __future__ import annotations

import inspect

import pytest

from STKO_to_python.dataprocess import AggregationEngine as AggEngineReexport
from STKO_to_python.dataprocess.aggregation import AggregationEngine


# ---------------------------------------------------------------------- #
# Import surface
# ---------------------------------------------------------------------- #
def test_package_reexport_is_same_class():
    assert AggEngineReexport is AggregationEngine


def test_module_and_qualname():
    assert AggregationEngine.__module__ == "STKO_to_python.dataprocess.aggregation"
    assert AggregationEngine.__qualname__ == "AggregationEngine"


# ---------------------------------------------------------------------- #
# Class shape — no @dataclass, no mixins, slots declared
# ---------------------------------------------------------------------- #
def test_has_slots():
    # per spec §2 / §6: every new class declares __slots__
    assert hasattr(AggregationEngine, "__slots__")


def test_single_inheritance_from_object():
    # no mixins; the only base is object
    assert AggregationEngine.__bases__ == (object,)


def test_construct_no_args():
    eng = AggregationEngine()
    assert isinstance(eng, AggregationEngine)


def test_repr_contains_class_name():
    assert "AggregationEngine" in repr(AggregationEngine())


# ---------------------------------------------------------------------- #
# Method surface — names we expect to see
# ---------------------------------------------------------------------- #
EXPECTED_PUBLIC_METHODS = (
    "delta_u",
    "drift",
    "residual_drift",
    "interstory_drift_envelope",
    "interstory_drift_envelope_pd",
    "story_pga_envelope",
    "residual_interstory_drift_profile",
    "residual_drift_envelope",
    "roof_torsion",
    "base_rocking",
    "asce_torsional_irregularity",
    "orbit",
)

# Methods that have been migrated from NodalResults into AggregationEngine
# (Phase 4.3.2). They must no longer raise NotImplementedError.
IMPLEMENTED_METHODS = frozenset({
    "delta_u",
    "drift",
    "residual_drift",
    "roof_torsion",
    "base_rocking",
    "asce_torsional_irregularity",
    "interstory_drift_envelope",
    "story_pga_envelope",
    "residual_interstory_drift_profile",
    "residual_drift_envelope",
})

STUB_METHODS = tuple(m for m in EXPECTED_PUBLIC_METHODS if m not in IMPLEMENTED_METHODS)

EXPECTED_PRIVATE_HELPERS = (
    "_resolve_story_nodes_by_z_tol",
)


@pytest.mark.parametrize("name", EXPECTED_PUBLIC_METHODS + EXPECTED_PRIVATE_HELPERS)
def test_method_exists_and_is_callable(name):
    m = getattr(AggregationEngine, name, None)
    assert m is not None, f"AggregationEngine is missing method {name!r}"
    assert callable(m), f"AggregationEngine.{name} is not callable"


@pytest.mark.parametrize("name", EXPECTED_PUBLIC_METHODS + EXPECTED_PRIVATE_HELPERS)
def test_method_first_param_is_results(name):
    """Every engineering method takes the NodalResults as the first
    positional argument after ``self``. This contract keeps
    ``NodalResults.<method>`` forwarding symmetric in Phase 4.3.2."""
    sig = inspect.signature(getattr(AggregationEngine, name))
    params = list(sig.parameters.values())
    # params[0] is 'self' on an unbound method accessed via the class
    assert params[0].name == "self"
    assert params[1].name == "results", (
        f"{name}: expected second parameter 'results', got {params[1].name!r}"
    )


# ---------------------------------------------------------------------- #
# Stubs raise NotImplementedError (Phase 4.3.1)
# ---------------------------------------------------------------------- #
def _dummy_kwargs_for(method) -> dict:
    """Build a kwargs dict filling every required keyword-only parameter
    with ``None``. Optional parameters keep their defaults. ``results``
    is always supplied; ``self`` is bound."""
    sig = inspect.signature(method)
    kwargs = {}
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty:
            kwargs[p.name] = None
    kwargs["results"] = None
    return kwargs


@pytest.mark.parametrize("name", STUB_METHODS)
def test_public_method_stub_raises_not_implemented(name):
    eng = AggregationEngine()
    method = getattr(eng, name)
    with pytest.raises(NotImplementedError):
        method(**_dummy_kwargs_for(method))


# _resolve_story_nodes_by_z_tol moved out of skeleton state in Phase 4.3.2.c;
# behavior is covered by tests/integration/test_aggregation_forwarders.py.
