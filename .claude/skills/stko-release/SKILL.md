---
name: stko-release
description: >
  Checklist for cutting a STKO_to_python release: bumping `version` in pyproject.toml,
  moving CHANGELOG.md [Unreleased] into a version section, the link references at the
  bottom of the changelog, tagging the merge commit on main, and checking that the
  tag-triggered release.yml built and attached the wheel. Use before any PR that
  changes the version or edits CHANGELOG.md release sections, and before pushing a
  tag. Not for installing or using the library (the end-user stko-to-python skill).
---

# Cutting a release — checklist

Read this before bumping the version, editing a `CHANGELOG.md` release section, or
pushing a `v*` tag. The policy lives in `AGENTS.md` "Versioning policy"; this checklist
covers the steps that have actually been missed (`AGENTS.md`: "A version bump is not a
release until it is tagged").

Out of scope: installing or using the library (the end-user `stko-to-python` skill).

## In the release PR

- [ ] Choose MAJOR / MINOR / PATCH according to `AGENTS.md` "Versioning policy". A
      public-API break needs a deprecation shim first (`docs/architecture.md`
      "Back-compat contract").
- [ ] Bump `version` in `pyproject.toml`. That line is the only version source:
      `release.yml` refuses a tag that doesn't match it.
- [ ] `CHANGELOG.md`: rename `[Unreleased]` to `## [X.Y.Z] — YYYY-MM-DD` and open a new
      empty `[Unreleased]` above it. Also collect what merged since the last tag and
      never made it into `[Unreleased]`:
      `git log --merges --first-parent v<last>..origin/main --oneline`.
- [ ] Update the link references at the bottom of `CHANGELOG.md`: add
      `[X.Y.Z]: .../compare/v<prev>...vX.Y.Z` and point `[Unreleased]` at
      `compare/vX.Y.Z...HEAD`. They stop at 1.8.0 today, and the `[1.6.0]` / `[1.7.0]`
      compare links point at tags that don't exist.
- [ ] Every release section needs a tag. If you describe a version as released, it
      must also be in `pyproject.toml` and tagged; otherwise fold it into the next
      release, as 1.12.0 did for 1.9–1.11.
- [ ] Docs that name the current version: `docs/architecture.md` "Versioning policy"
      still says **v1.1.0**. `grep -rn "Current release" docs` finds it.
- [ ] `mkdocs build --strict` is clean, and every check on the PR is green
      (`gh pr checks <n>`). `AGENTS.md`: "Never merge on a red check".

## After the merge (the step that gets skipped)

- [ ] Tag the merge commit on `main`, not the branch head:
      `git fetch origin && git tag vX.Y.Z <merge-sha> && git push origin vX.Y.Z`.
      Tags are lightweight unless you are cutting a real GitHub release with artifacts.
- [ ] Confirm that `release.yml` ran green for the tag and that the release carries
      the wheel and the sdist: `gh run list --workflow release.yml --limit 3`, then
      `gh release view vX.Y.Z`.
- [ ] `git ls-remote --tags origin` lists the new tag. Before the next release, check
      that every `## [x.y.z]` section in `CHANGELOG.md` has a tag.

## Don't

- Don't push to `main` or merge the PR yourself unless the owner asked you to.
- Don't move or re-point a tag that has been pushed; cut a new PATCH instead.
- If a feature PR bumps the version, it is a release PR: tag it after the merge.
  That is how 1.6.0 (f7e2b62) went untagged.
