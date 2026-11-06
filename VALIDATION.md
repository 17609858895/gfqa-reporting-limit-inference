# Release preparation checks

Checks completed on 7 November 2026 using the recorded Python environment.

- All 40 Python files pass AST parsing and in-memory compilation. Static import closure is complete using the included local helpers and declared third-party dependencies. Analysis and plotting entry points were not imported to discover dependencies.
- The release whitelist contains code, documentation, environment/version records, comparison diagnostics and one original schematic background. No raw observations, prediction arrays, fitted models, result tables, manuscript files, prompts, backups or caches are included. Text scans found no credential-like strings, email addresses or user-specific absolute paths.
- Numerical utility checks pass for interval likelihoods, finite-difference gradients, empirical coverage bounds, conformal order statistics and strict logarithmic equality at five screening thresholds. These checks do not train models or load monitoring data.
- The original GFQA ZIP's size, official MD5 and locally recorded SHA-256 were verified by a read-only scan. The two metadata tables used by the analysis match the preparation script's values and missingness. The third methods table is also supported by the preparation script; the working analysis did not require a converted counterpart. See `input_preparation_verification.json`.
- Both fixed-version Natural Earth download URLs were fetched and parsed. Their GeoJSON contents match the working figure geometries. See `basemap_preparation_verification.json`.
- The added country summary has 101 rows with matching keys and maximum numeric discrepancy of 1.11 × 10−16. The 21-row robustness summary matches all numeric values exactly. These are floating-point comparisons of every output cell, using a 10−12 tolerance; no observational values are included in the verification record. See `summary_equivalence.json`.
- The original schematic background was re-encoded with exact pixel identity. Its release copy passes the strict ancillary-metadata scan. Source comparison found 27 unchanged working-source files and five documented adapted copies, with no unrecorded change.
- The serial runner's complete 29-command plan was exercised in its default planning mode. It did not execute the analysis stages or create analytical outputs. Full conversion, model training, regenerated predictions and a clean-environment installation were not rerun for release preparation.

Run `python scripts/validate_release.py` to check the inventory and hashes against `CODE_MANIFEST.json`, and `python scripts/smoke_checks.py` for the numerical utility checks. A complete reproduction additionally requires downloading the fixed raw data release and explicitly executing the documented stages.
