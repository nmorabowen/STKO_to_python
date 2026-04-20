"""Unit tests for ``MpcoFormatPolicy``."""
from __future__ import annotations

import pytest

from STKO_to_python.io.format_policy import MpcoFormatPolicy


@pytest.fixture
def policy() -> MpcoFormatPolicy:
    return MpcoFormatPolicy()


# ---------------------------------------------------------------------- #
# Identity and shape                                                     #
# ---------------------------------------------------------------------- #

def test_policy_has_no_instance_state(policy: MpcoFormatPolicy) -> None:
    """__slots__ is empty; the policy is stateless."""
    assert MpcoFormatPolicy.__slots__ == ()


def test_repr(policy: MpcoFormatPolicy) -> None:
    assert repr(policy) == "MpcoFormatPolicy()"


# ---------------------------------------------------------------------- #
# Model-stage recognition                                                #
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("key", ["MODEL_STAGE[1]", "MODEL_STAGE[2]", "MODEL_STAGE[42]"])
def test_recognizes_model_stage_keys(policy: MpcoFormatPolicy, key: str) -> None:
    assert policy.is_model_stage_group(key) is True


@pytest.mark.parametrize(
    "key",
    ["INFO", "MODEL", "RESULTS", "", "model_stage[1]", "Model_Stage[1]"],
)
def test_rejects_non_stage_keys(policy: MpcoFormatPolicy, key: str) -> None:
    assert policy.is_model_stage_group(key) is False


# ---------------------------------------------------------------------- #
# Path templates: pin the exact strings the library uses today            #
# ---------------------------------------------------------------------- #

def test_model_nodes_path(policy: MpcoFormatPolicy) -> None:
    assert policy.model_nodes_path("MODEL_STAGE[1]") == "/MODEL_STAGE[1]/MODEL/NODES"


def test_model_elements_path(policy: MpcoFormatPolicy) -> None:
    assert policy.model_elements_path("MODEL_STAGE[3]") == "/MODEL_STAGE[3]/MODEL/ELEMENTS"


def test_results_on_nodes_path(policy: MpcoFormatPolicy) -> None:
    assert (
        policy.results_on_nodes_path("MODEL_STAGE[1]")
        == "/MODEL_STAGE[1]/RESULTS/ON_NODES"
    )


def test_results_on_elements_path(policy: MpcoFormatPolicy) -> None:
    assert (
        policy.results_on_elements_path("MODEL_STAGE[2]")
        == "/MODEL_STAGE[2]/RESULTS/ON_ELEMENTS"
    )


def test_path_templates_match_legacy_dataset_constants() -> None:
    """The new policy templates must produce the exact same strings as
    the class-level constants still pinned on ``MPCODataSet``. If this
    ever diverges the refactor has broken path compatibility.
    """
    from STKO_to_python.core.dataset import MPCODataSet

    policy = MpcoFormatPolicy()
    stage = "MODEL_STAGE[7]"

    assert MPCODataSet.MODEL_NODES_PATH.format(model_stage=stage) == policy.model_nodes_path(stage)
    assert MPCODataSet.MODEL_ELEMENTS_PATH.format(model_stage=stage) == policy.model_elements_path(stage)
    assert (
        MPCODataSet.RESULTS_ON_NODES_PATH.format(model_stage=stage)
        == policy.results_on_nodes_path(stage)
    )
    assert (
        MPCODataSet.RESULTS_ON_ELEMENTS_PATH.format(model_stage=stage)
        == policy.results_on_elements_path(stage)
    )
