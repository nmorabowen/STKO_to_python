This is a helper class to parse results obtained from opensees using MPCO recorders.

The class works with STKO outputs

## Versioning policy

We follow semver and tag releases on `main`:

- **MAJOR** (`vX.0.0`) — breaking changes to the public API.
- **MINOR** (`v1.X.0`) — new backward-compatible features (new methods, new result types, new plot helpers).
- **PATCH** (`v1.x.Y`) — bug fixes, docs, tests, internal refactors with no API change.

When merging a PR (or a batch of related PRs) that warrants a release:
1. Bump `version` in `pyproject.toml`.
2. After the PR merges, tag the merge commit on `main` (`git tag vX.Y.Z <sha> && git push origin vX.Y.Z`).
3. Tags are lightweight unless a real GitHub release with artifacts is being cut.