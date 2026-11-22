# PTC-SSL on Kaggle — notebooks 01–04 (data pipeline)

Run each notebook as a separate Kaggle notebook ("File → Import Notebook" → upload the .ipynb).
After each run, click **Save Version → Save & Run All**, then attach its output to the next notebook:
*Add Input → Your Work → Notebooks*.

| # | Notebook | Attach outputs of | Settings | Approx. time |
|---|---|---|---|---|
| 01 | 01_dataset_acquisition | — | Internet **On**, CPU | 10–25 min |
| 02 | 02_data_profiling | 01 | CPU | 2–4 min |
| 03 | 03_preprocessing | 01, 02 | CPU | 3–6 min |
| 04 | 04_temporal_windowing | 01, 02, 03 | CPU | 1–3 min |

Every notebook stops with an error if an integrity, unit, leakage or label-shift check fails.
Do not continue past a failed notebook.

After 04, send back these files (no patient data in them):
- results/tables/integrity_checks.csv, dataset_counts_by_hospital.csv
- results/tables/table1_cohort.csv, unit_verification.csv, constraint_validation.csv, window_length_coverage.csv
- results/tables/split_summary.csv, sample_counts.csv, label_subset_sizes.csv
- results/data_profile_report.md

These real numbers decide the settings for notebooks 05–11: the window length, which physiology
constraints go into L_phys, and the label-fraction feasibility.

Testing note: the code was tested end to end on a SYNTHETIC, schema-identical fixture (tests/), run
as separate sessions with chained inputs. That fixture is used only to check the code, never for
results. All reported numbers must come from the real Kaggle runs.
