# Release preparation checks

## v1.3.0 (20 November 2026)

- Added `s19_interpretive_diagnostics.py`, an unchanged copy of the working script, and listed it at the end of the sensitivity stage of the serial runner (33 commands in plan mode).
- Static checks passed (AST parsing, in-memory compilation, closed import graph, inventory hashes, and scans for credentials, email addresses and user-specific paths). The numerical utility checks in `smoke_checks.py` also passed.
- Executed from this release copy against the existing crossfit and benchmark test predictions (copied locally, not included), `s19` reproduced the 312-row diagnostics table used for the manuscript byte for byte. No learner was refitted, and the other stages were not rerun for this release.

## v1.2.0 (19 November 2026)

- Synchronised `generate_six_main_figures.py` and `generate_main_figures_06_10.py` with the manuscript figures and added `generate_toc_graphic.py`. Release-only differences remain the documented portability changes listed in `SOURCE_ADAPTATIONS.md`.
- Static checks passed (AST parsing, in-memory compilation, closed import graph, inventory hashes, and scans for credentials, email addresses and user-specific paths). Executed from this release copy against the existing analytical outputs (linked locally, not included), the figure stage was compared pixel by pixel with the manuscript figures, panels and table-of-contents graphic; the result is recorded below.
- Result: all 6 combined figures, 39 independent panels and the table-of-contents graphic (46 PNG files) were pixel-identical to the manuscript versions.

## v1.1.0 (14 November 2026)

- The plotting source was synchronised with the six-figure manuscript and `s18_annual_compliance.py` was added. Only the documented portability changes differ from the working copies: local helper imports, `inputs/basemap/`, `figures/qa/` and `assets/monitoring_vignette.png` paths, and neutral module docstrings.
- The six combined figures, their 37 independent panels and `s18` outputs were regenerated in the working analysis project from its existing analytical outputs; no learner was refitted. The release copy itself was checked statically: AST parsing, in-memory compilation, closed import graph, inventory hashes and text scans for credentials, email addresses and user-specific paths. The serial runner's plan mode lists the updated stages.
- Executed from this release copy against the existing analytical outputs (linked locally, not included), the figure stage reproduced all 6 combined figures and 37 independent panels pixel-identically to the manuscript figures.
- The vignette was re-encoded with exact pixel identity and no ancillary metadata.

## v1.0.0 (7 November 2026)

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
