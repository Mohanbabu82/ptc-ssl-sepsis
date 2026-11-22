# PTC-SSL — Physiology-Guided Self-Supervised Temporal Representation Learning for Label-Efficient Early Sepsis Prediction

Code for the capstone project and the IEEE Access manuscript. Data: PhysioNet/CinC Challenge 2019
(v1.0.0, CC BY 4.0, 40,336 patients). Everything runs inside Kaggle notebooks; no local install is needed.

## How to reproduce (one notebook, recommended)

1. On Kaggle: **Create → New Notebook → File → Import Notebook**, upload `notebooks/PTC_SSL_full_run.ipynb`.
2. Settings: **Accelerator = GPU T4 x2**, **Internet = On**.
3. Click **Save Version → Save & Run All (Commit)**. The run downloads the data from PhysioNet,
   verifies it (MD5 + structural checks), and runs stages 01–04 and 06 in order.
   Final run used for the paper: 2026-11-20, Tesla T4, 12,860 s (3.6 h), 0 failed or skipped jobs.
4. Download from the notebook's **Output** tab:
   - `PTC_SSL_IEEE_ACCESS_RESEARCH_PACKAGE.zip` — code, configs, logs, tables, figures (no raw patient data)
   - `PTC_SSL_PAPER_RESULTS.zip` — the tables and figures used in the paper

The run stops with an error if any integrity, unit, leakage, causality or label-shift check fails.

## Revision 2 (reviewer checklist): two more Kaggle runs

Import and commit each notebook below as its own Kaggle notebook, with the same settings (GPU T4 x2, Internet on,
Save & Run All). Each is self-contained (it re-runs stages 01–04), so they can run one after the other or, if your
Kaggle quota allows two GPU sessions, at the same time. Each needs roughly 9–11 h of the 30 h weekly GPU quota.

| Notebook | What it answers | Output |
|---|---|---|
| `notebooks/PTC_SSL_v2a_core_validity_full_run.ipynb` | Does the physiological prior help? Full 2³ factorial ablation, 10 seeds, 1–100% labels, measurement pattern vs values, test-time perturbations, validation-only recalibration, hierarchical bootstrap (2,000) with Holm correction | `PTC_SSL_V2A_RESULTS.zip` |
| `notebooks/PTC_SSL_v2b_baselines_transfer_full_run.ipynb` | TS2Vec-style, SimMTM-style and Transformer baselines; sensitivity (λ_phys, λ_temp, hidden size, block length, masking); why A→B helps and B→A hurts (FiO₂, missingness, unsupervised domain adaptation, hospital shift) | `PTC_SSL_V2B_RESULTS.zip` |

Shared code: `src/exp_lib.py` (inserted into both notebooks). Tests: `python tests/test_07_numpy.py` after
`python tests/test_chain_and_negative.py` (synthetic fixture; model code needs PyTorch and is exercised by each
notebook's smoke test on Kaggle).

## Stages

| Stage | Script / notebook | What it does |
|---|---|---|
| 01 | `src/01_dataset_acquisition.py` | Download with MD5/ETag verification, structural checks, parquet export, dataset fingerprint (SHA-256) |
| 02 | `src/02_data_profiling.py` | Cohort table, missingness, unit verification, physiological-range QC, constraint validation (C1–C6) |
| 03 | `src/03_preprocessing.py` | Exclusions, QC, patient-level splits (pooled, A→B, B→A), train-only normalisation, organ-system map, constraint statistics |
| 04 | `src/04_temporal_windowing.py` | Causal 24-h windows, automated look-ahead tests, six-hour label shift, nested label subsets (5 seeds) |
| 05 | `src/05_experiments.py` | Single-seed CPU pilot (superseded; kept for transparency) |
| 06 | `src/06_full_experiments.py` | Final experiments: 5 seeds × {LightGBM, supervised GRU, generic SSL, PTC-SSL} × {5, 10, 25, 100%} labels; 5 ablations; A→B / B→A external validation (3 seeds); seed-ensemble paired bootstrap; per-seed wins; ECE; result packages |
| — | `src/common_setup.py` | Shared paths, seeding, logging, table I/O, utility-score parameters |

`notebooks/01_…06_…ipynb` are the same stages as separate notebooks; `PTC_SSL_full_run.ipynb` combines 01–04 and 06.
`build.py` builds the notebooks from `src/`.

## Results

`results/` holds the final five-seed tables parsed from the GPU run log:
`multiseed_summary.csv` (mean ± SD per protocol, arm, label fraction), `paired_bootstrap_seed_ensemble.csv`
(PTC-SSL minus comparator, 95% CI, p), `per_seed_wins.csv`.

## Tests

`tests/` checks the code on a SYNTHETIC, schema-identical fixture (`make_fixture.py`). The fixture is used
only to test the code, never for results; every reported number comes from the real Kaggle run.
Run with `python run_test.py`.

## Rules followed

Real PhysioNet data and original labels only; patient-level splits; validation and test patients kept out of
pretraining; strictly causal inputs; no tuning on the test set; official utility score; raw patient data excluded
from every downloadable package. Not a clinical tool.

## Package layout (final, revision 2)

| Folder | Contents |
|---|---|
| `src/` | Stage scripts 01-04 (data), 05 (pilot), 06 (first run), `exp_lib.py` + 07a/07b (revision 2) |
| `notebooks/` | Kaggle notebooks; `PTC_SSL_v2a_core_validity_full_run.ipynb` and `PTC_SSL_v2b_baselines_transfer_full_run.ipynb` produced every reported number |
| `tests/` | Synthetic-fixture tests (`python run_test.py`) |
| `results/` | `parse.py` turns the notebook logs (`logs/v2a_lines.txt`, `logs/v2b_lines.txt`: the RESULT/MS/CP/FX/PS/SH/HC/FQ lines) into tables; `make_xlsx.py` writes `PTC_SSL_Complete_Results.xlsx` |
| `figures/` | Figure scripts for the paper (`fig1.py`, `figs.py`, `eqimg.py`) and the report (`report_figs.py`, reads `results/store.pkl`) |
| `manuscript/` | `content.py` (IEEE Access text, tables, references) and `build.py` (fills the IEEE Access Word template) |

Reproduce: run the two full-run notebooks on Kaggle (GPU T4, internet on, Save & Run All), copy the result lines from their logs to `results/logs/`, then `python parse.py && python make_xlsx.py` in `results/` and `python report_figs.py` in `figures/`.

## Results in this repository

- `results/data/PTC_SSL_Complete_Results.xlsx`: every training run (480 + 330), perturbation evaluation, statistical comparison, factorial effect, hospital-shift and measurement-rate table, with a README sheet.
- `results/data/*.csv`: the same tables as CSV.
- `results/logs/`: full Kaggle run logs and the parsed result lines (`v2a_lines.txt`, `v2b_lines.txt`).
- `results/figures/`: all paper and report figures.
- `results/kaggle_outputs/`: the original result packages written by the two Kaggle notebooks (tables, configurations, figures, training logs).

No raw patient data are included. The data are available from PhysioNet: https://physionet.org/content/challenge-2019/1.0.0/ (CC BY 4.0).
