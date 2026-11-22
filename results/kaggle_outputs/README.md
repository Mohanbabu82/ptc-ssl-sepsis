# Original Kaggle result packages

Unzipped contents of `PTC_SSL_V2A_RESULTS.zip` (notebook `ptc-ssl-v2a-core-validity`) and `PTC_SSL_V2B_RESULTS.zip` (notebook `ptc-ssl-v2b-baselines-transfer`), downloaded from each notebook's Output tab on 2026-11-22.

- `results/tables/`: per-run metrics (`runs_metrics_v2a.csv`, `runs_metrics_v2b.csv`), multi-seed summaries, bootstrap comparisons, factorial effects, perturbation metrics, hospital shift, cohort, QC, unit and constraint checks, label-subset sizes.
- `results/configs/`: normalisation statistics, QC ranges, organ systems, validated constraints, experiment configurations.
- `results/figures/`: figures written by the notebooks.
- `logs/`: per-run pretraining and fine-tuning loss curves.

All files are aggregate statistics; no raw patient data are included. Hour-level test-set predictions (about 0.9 GB) remain in the notebooks' `artifacts/predictions_v2a` and `artifacts/predictions_v2b` output folders on Kaggle.
