"""Assemble one-notebook Kaggle runs: stages 01-04 followed by an experiment notebook (06, 07a or 07b)."""
import json
from build import parse, to_ipynb, SRC, OUT

PREP = ["01_dataset_acquisition", "02_data_profiling", "03_preprocessing", "04_temporal_windowing"]
RUNS = {
    "PTC_SSL_v2a_core_validity_full_run": ("07a_core_validity", "about 9–11 h",
        "07a core validity (10 seeds incl. 1% and 2% labels, full 2³ factorial ablation, measurement-pattern vs values, "
        "test-time perturbations, validation-only recalibration, hierarchical bootstrap with Holm correction)",
        "`PTC_SSL_V2A_RESULTS.zip`"),
    "PTC_SSL_v2b_baselines_transfer_full_run": ("07b_baselines_transfer", "about 8–10 h",
        "07b stronger baselines (TS2Vec-style, SimMTM-style, Transformer), sensitivity analysis, cross-hospital "
        "diagnosis (FiO₂, missingness, unsupervised domain adaptation) and hospital shift analysis",
        "`PTC_SSL_V2B_RESULTS.zip`"),
}
for out_name, (exp, hours, what, zipname) in RUNS.items():
    head = (f"# PTC-SSL revision 2 — {exp} (stages 01–04 + {exp.split('_')[0]} in one Kaggle notebook)\n"
            f"**Settings:** Accelerator = GPU T4 x2, Internet = On. Use **Save Version → Save & Run All (Commit)**; it runs "
            f"headless ({hours}; an internal time guard stops new training runs after 11 h so the statistics always finish) "
            f"and keeps every output.\n\n"
            f"Sections: 01 dataset acquisition · 02 profiling · 03 preprocessing · 04 windowing · {what}.\n\n"
            f"Outputs: `results/` (tables, figures, configs, logs) and {zipname}. No raw patient data is packaged. "
            f"A smoke test runs every arm on a few thousand rows before the long jobs, so a bug stops the run within minutes.")
    cells = [("markdown", head)]
    for nb in PREP + [exp]:
        cells += parse((SRC / f"{nb}.py").read_text())
    nbj = to_ipynb(cells)
    (OUT / f"{out_name}.ipynb").write_text(json.dumps(nbj, indent=1))
    print(f"built {out_name}.ipynb: {len(nbj['cells'])} cells")
