# Source adaptations

The release preserves analysis script names `s01`–`s17` and the model utility implementations. The following changes apply to this repository copy only; the working analysis project was not edited.

1. `s01_zip_to_parquet.py` resolves the repository root and reads `inputs/GFQA_v3.zip`; converted measurements remain under `data_processed/`.
2. `s05_compliance_blindspot.py` compares `y_hi`/`y_lo` directly with `log10(threshold)`. This preserves equality at the threshold and avoids round-off introduced by transforming a logarithmic endpoint back to the concentration scale. The numerical thresholds are unchanged. The PFOA basis text identifies an illustrative screening comparator, rather than an individual-PFOA legal standard.
3. `s16_retest_priority.py` documentation uses follow-up measurement terminology and explicitly distinguishes the country-level full modelling period from station records since 2015. Calculations and historical column names are unchanged.
4. The compact figure helper imports local copies of the publication-style and palette helpers. Its default basemap path is `inputs/basemap/`; it no longer refers to a user-specific skill directory. The workflow panel reads the single required original background at `assets/method_background.png`, re-encoded without ancillary metadata with its pixels unchanged. No rendered manuscript figure or prompt is included.
5. `s00_prepare_inputs.py` makes the metadata conversion and release verification explicit. `assemble_figure_inputs.py` makes two previously assembled plot tables reproducible from analytical outputs; comparison diagnostics are in `summary_equivalence.json`.
6. `run_pipeline.py` supplies a serial command plan, the study benchmark cap, local logs and failure stopping. The old workstation-specific shell chains are not included. `s04` remains available for historical feasibility inspection but is not executed by the study plan.

The numbered analytical scripts, numerical model helpers and current plotting source were inspected as text/AST. Modules that train models or read/write analysis files were not imported to discover dependencies. Preparation of this release did not fit learners or regenerate analytical results.
