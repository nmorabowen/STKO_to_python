# Test fixtures

This directory holds golden test fixtures for the test suite.

## `golden.mpco`

Not yet checked in. When a small, representative `.mpco` file is available
it should be placed here as `golden.mpco`. Tests that depend on it request
the `mpco_fixture_path` pytest fixture and are skipped automatically when
the file is absent, so CI stays green in the meantime.

Requirements for the fixture:

- Small (ideally < 5 MB) — store as plain bytes, not LFS.
- At least one `MODEL_STAGE` group.
- A mix of nodal and element results.
- Representative of the element/integration-rule mix the library parses.

See §9 of `docs/architecture-refactor-proposal.md` for the full golden-test
contract.
