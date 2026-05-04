# Cookbook

Opinionated, runnable end-to-end recipes for the engineering
workflows the library is built around. Each tutorial:

- opens with the engineering question, not the API;
- runs against a checked-in fixture under `stko_results_examples/`;
- produces 1–2 plots so the result is concrete;
- closes with a *Variations* footer so adapting it to your own model
  is a one-line change.

| # | Tutorial | Fixture | Element family |
|---|---|---|---|
| 1 | [Base shear from column forces](01-base-shear-from-column-forces.md) | `elasticFrame/elasticFrame_mesh_results` | `5-ElasticBeam3d` |
| 2 | [Stress contour on a shear wall](02-stress-contour-on-a-wall.md) | `elasticFrame/QuadFrame_results` | `203-ASDShellQ4` |
| 3 | [Peak moment per beam over a pushover](03-peak-moment-per-beam.md) | `elasticFrame/elasticFrame_mesh_displacementBased_results` | `64-DispBeamColumn3d` |
| 4 | [Volume integral of σ_11 over a brick subdomain](04-volume-integral-on-bricks.md) | `solid_partition_example` (gitignored — skip-if-absent) | `56-Brick` |
| 5 | [Selector + mask pipeline](05-selector-and-mask-pipeline.md) | `elasticFrame/elasticFrame_mesh_displacementBased_results` | `64-DispBeamColumn3d` |

For broader tours of the API on a single fixture see the
[Examples](../examples/usage_tour.md) section. For the reference
documentation behind the calls used here see
[ElementResults](../ElementResults.md) and
[Canonical names](../api/canonical-names.md).
