# MPCOResults

Multi-case container for a tree of `NodalResults` (typically keyed by
Tier / Case / station / rupture). Offers a family of MPCO-specific
DataFrame extractors (drift, PGA, torsion, base rocking, with
``_long`` / ``_mod`` variants) reachable through the
**`.df` accessor** introduced in Phase 4.5.

`mpco_results.df` and `mpco_results.create_df` reference the same
`MPCO_df` instance; `.df` is the preferred spelling.

## MPCOResults

::: STKO_to_python.MPCOList.MPCOResults.MPCOResults

## MPCO_df

The DataFrame-extractor accessor. Reach it via
`mpco_results.df` (preferred) or `mpco_results.create_df` (legacy).

::: STKO_to_python.MPCOList.MPCOdf.MPCO_df
