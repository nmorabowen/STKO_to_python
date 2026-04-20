"""Centralized MPCO HDF5 format policy.

The ``.mpco`` file format (written by STKO's MPCORecorder) has a handful
of conventions — where model stages live, how path templates are built,
what a "result on nodes" group is named — that today are spread across
``ModelInfo``, ``Nodes``, ``Elements``, and inline string checks. This
class is the single place that owns those conventions.

Phase 1 lands the class with the format questions the current codebase
already answers inline. Subsequent phases (notably the query engines in
Phase 2) will migrate each consumer to call the policy instead of
reinventing the check. That migration is deliberately done one consumer
at a time so a single PR per consumer stays small and reviewable.

See ``docs/architecture-refactor-proposal.md`` §3.1 for the full design
intent and the MPCO-recorder skill for authoritative format details.
"""
from __future__ import annotations


_MODEL_STAGE_PREFIX = "MODEL_STAGE"


class MpcoFormatPolicy:
    """Pure-functional knowledge of the MPCO HDF5 layout.

    The policy is deliberately stateless: every method is a pure function
    of its arguments. One instance can be shared across every manager,
    reader, and query engine in a dataset. ``__slots__`` is empty because
    there is nothing instance-specific to store.

    Thread-safety
    -------------
    Safe for concurrent access. No mutable state.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "MpcoFormatPolicy()"

    # ------------------------------------------------------------------ #
    # Model-stage group identification                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def is_model_stage_group(key: str) -> bool:
        """True iff ``key`` names a top-level ``MODEL_STAGE[...]`` group.

        MPCO appends one group per ``Domain::hasDomainChanged()`` stamp:
        ``MODEL_STAGE[1]``, ``MODEL_STAGE[2]``, etc. The check used today
        across the library is a simple ``startswith`` on the key.

        Parameters
        ----------
        key : str
            An HDF5 group name (e.g. from ``file.keys()``).
        """
        return key.startswith(_MODEL_STAGE_PREFIX)

    # ------------------------------------------------------------------ #
    # Path templates                                                      #
    # ------------------------------------------------------------------ #
    #
    # The four path templates below replicate the class-level constants
    # currently pinned on ``MPCODataSet`` (``MODEL_NODES_PATH``, …). They
    # remain accessible through the dataset for backward compatibility
    # via thin properties; new code should consume them here.

    @staticmethod
    def model_nodes_path(model_stage: str) -> str:
        """HDF5 path to the model's NODES group for a given stage."""
        return f"/{model_stage}/MODEL/NODES"

    @staticmethod
    def model_elements_path(model_stage: str) -> str:
        """HDF5 path to the model's ELEMENTS group for a given stage."""
        return f"/{model_stage}/MODEL/ELEMENTS"

    @staticmethod
    def results_on_nodes_path(model_stage: str) -> str:
        """HDF5 path to per-node results for a given stage."""
        return f"/{model_stage}/RESULTS/ON_NODES"

    @staticmethod
    def results_on_elements_path(model_stage: str) -> str:
        """HDF5 path to per-element results for a given stage."""
        return f"/{model_stage}/RESULTS/ON_ELEMENTS"
