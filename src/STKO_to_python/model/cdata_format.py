"""Centralized ``.cdata`` text-format policy.

The ``.cdata`` sidecar (companion to ``.mpco``) is a line-oriented text
format with a handful of section markers and a few encoding
conventions (e.g. ``LENGTH NAME`` for names that may contain spaces).
This module is the single place that owns those format facts.

It mirrors :class:`STKO_to_python.io.format_policy.MpcoFormatPolicy`:
pure-functional, stateless, safe to share across threads.

See ``docs/architecture-refactor-proposal.md`` §3.1 for the design
intent shared with the MPCO policy.
"""
from __future__ import annotations


class CDataFormatPolicy:
    """Pure-functional knowledge of the ``.cdata`` text-file layout.

    The policy is stateless: section markers and conventions are class
    attributes; helper queries are ``@staticmethod`` / ``@classmethod``.
    One instance can be shared across every reader and parser.

    Thread-safety
    -------------
    Safe for concurrent access. No mutable state.
    """

    __slots__ = ()

    # ------------------------------------------------------------------ #
    # Section marker tokens (the line that opens each section)            #
    # ------------------------------------------------------------------ #

    MARKER_SELECTION_SET = "*SELECTION_SET"
    MARKER_LOCAL_AXES = "*LOCAL_AXES"
    MARKER_SECTION_OFFSET = "*SECTION_OFFSET"
    MARKER_BEAM_PROFILE = "*BEAM_PROFILE"
    MARKER_BEAM_PROFILE_ASSIGNMENT = "*BEAM_PROFILE_ASSIGNMENT"
    MARKER_ELEMENT_INFO = "*ELEMENT_INFO"

    def __repr__(self) -> str:
        return "CDataFormatPolicy()"

    # ------------------------------------------------------------------ #
    # Marker queries                                                      #
    # ------------------------------------------------------------------ #

    @classmethod
    def known_markers(cls) -> frozenset[str]:
        """Every section-opening marker this parser recognizes."""
        return frozenset(
            {
                cls.MARKER_SELECTION_SET,
                cls.MARKER_LOCAL_AXES,
                cls.MARKER_SECTION_OFFSET,
                cls.MARKER_BEAM_PROFILE,
                cls.MARKER_BEAM_PROFILE_ASSIGNMENT,
                cls.MARKER_ELEMENT_INFO,
            }
        )

    @staticmethod
    def is_section_marker(line: str) -> bool:
        """True iff *line* (after strip) is a section-opening ``*MARKER`` line.

        Any unrecognized ``*`` line is still treated as a section
        boundary by the reader; this helper checks for the *known* set.
        """
        return line.strip() in CDataFormatPolicy.known_markers()

    @staticmethod
    def is_any_marker(line: str) -> bool:
        """True iff *line* (after strip) starts with ``*`` — a section boundary.

        The reader uses this when walking data lines inside a section
        to detect the start of the next section, even if the marker is
        one this policy version does not recognize.
        """
        s = line.strip()
        return bool(s) and s.startswith("*")
