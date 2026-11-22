# %% [markdown]
# # 07a · Core validity experiments (revision 2)
# Answers the first two reviewer questions: **does the physiological prior itself help, and does the model learn
# physiology or the measurement pattern?** Runs after 01–04 in the same session. Every job and setting below was
# fixed before any of these runs; the test set is never used for tuning.
#
# 1. **Main comparison, 10 seeds**: LightGBM / supervised GRU / generic SSL / PTC-SSL at 1, 2, 5, 10, 25, 100% labels.
# 2. **Full 2³ factorial ablation** of organ masking (O), L_phys (P), L_temp (T) at 5% and 10% labels, 10 seeds
#    (generic SSL = no component, PTC-SSL = all three).
# 3. **Measurement pattern vs values**: values-only and missingness-only inputs for PTC-SSL, the supervised GRU and
#    LightGBM, 5% and 10% labels, 10 seeds; descriptive measurement-frequency analysis by hospital.
# 4. **Test-time perturbations** of the measurement pattern (observations only removed, never invented):
#    30% of labs dropped, 30% of vitals dropped, lactate removed, labs capped at one per 24 h.
# 5. **Validation-only recalibration** (Platt scaling fitted on the label-budgeted validation set).
# 6. **Statistics**: 2,000 paired patient-level bootstrap resamples; per-seed differences with a hierarchical
#    bootstrap over seeds and patients, reported separately from seed-ensemble differences; Holm correction.
#
# Jobs run seed by seed (seed is the outer loop), so if the time guard stops the run early every analysis still has
# the same, smaller number of seeds. Every run is appended to `results/tables/runs_metrics_v2a.csv` and skipped on resume.

# %% [common]

# %%
NB = "07a_core_validity"
log = get_logger(NB)

# %% [lib]

# %%
SEEDS = list(range(int(os.environ.get("PTCSSL_SEEDS", 10))))
MAIN_ARMS = ["lightgbm", "supervised_gru", "generic_ssl", "ptc_ssl"]
FAC_ARMS = ["fac_O", "fac_P", "fac_T", "fac_OP", "fac_OT", "fac_PT"]
MISS_ARMS = ["abl_values_only", "abl_missing_only", "supervised_values_only", "supervised_missing_only",
             "lgbm_values_only", "lgbm_missing_only"]
JOBS = [
    {"name": "main", "fractions": [0.01, 0.02, 0.05, 0.10, 0.25, 1.00], "arms": MAIN_ARMS},
    {"name": "factorial", "fractions": [0.05, 0.10], "arms": FAC_ARMS},
    {"name": "missingness", "fractions": [0.05, 0.10], "arms": MISS_ARMS},
]
PERTURB = {"arms": ["lightgbm", "supervised_gru", "generic_ssl", "ptc_ssl", "abl_values_only", "abl_missing_only"],
           "fractions": [0.05, 1.00], "seeds": SEEDS[:5],
           "names": ["drop_labs_30", "drop_vitals_30", "no_lactate", "labs_max_1_per_24h"]}
if os.environ.get("PTCSSL_QUICK"):
    SEEDS = SEEDS[:2]; PERTURB["seeds"] = SEEDS[:1]
save_json({"CFG": CFG, "SEEDS": SEEDS, "JOBS": JOBS, "PERTURB": PERTURB, "VARIANTS": VARIANTS}, RES / "configs" / "experiment_config_v2a.json")
use_protocol("pooled")
set_spec("ptc_ssl")
IN_DIM = causality_check()
log.info("Batch builder causality check passed; input dim=%d", IN_DIM)
missing_cols = [f"train_s{s}_f{int(round(f*100)):03d}" for s in SEEDS for j in JOBS for f in j["fractions"]
                if f"train_s{s}_f{int(round(f*100)):03d}" not in subsets.columns]
assert not missing_cols, f"label subsets missing (re-run 04 with 10 seeds and 1%/2% fractions): {missing_cols[:5]}"

# %% [markdown]
# ## Descriptive analysis: how measurement frequency differs before sepsis, by hospital (training patients only)

# %%
def measurement_frequency_table():
    tr = patients[patients["split_pooled"] == "train"]
    rows = []
    rs = np.random.default_rng(GLOBAL_SEED)
    for hosp in ("A", "B"):
        th = tr[tr.hospital == hosp]
        for septic in (1, 0):
            grp = th[th.septic == septic]
            cnt, hours = np.zeros(NV), 0
            for pid in grp.index:
                s = starts[pid]; n = int(patients.loc[pid, "n_rows"])
                if septic:
                    ts = int(patients.loc[pid, "t_sepsis_idx"])
                    a, b = ts - 30, ts - 6                       # the 24 h before the label turns positive
                else:
                    if n < 30:
                        continue
                    b = int(rs.integers(24, n)); a = b - 24
                if a < 0:
                    continue
                cnt += M[s + a:s + b].sum(0); hours += (b - a)
            for j, v in enumerate(DYNAMIC_VARS):
                rows.append({"hospital": hosp, "septic": septic, "variable": v, "kind": "lab" if v in LABS else "vital",
                             "measurements_per_24h": 24 * cnt[j] / max(hours, 1), "patient_hours": hours})
    t = pd.DataFrame(rows)
    w = t.pivot_table(index=["hospital", "variable", "kind"], columns="septic", values="measurements_per_24h").reset_index()
    w.columns = ["hospital", "variable", "kind", "nonseptic_per_24h", "septic_per_24h"]
    w["ratio_septic_to_nonseptic"] = w.septic_per_24h / w.nonseptic_per_24h.replace(0, np.nan)
    return w

with StageTimer("measurement-frequency analysis", log):
    freq = measurement_frequency_table()
    freq.to_csv(RES / "tables" / "measurement_frequency_by_hospital.csv", index=False)
    for _, r in freq.iterrows():
        print("FQ", r.hospital, r.variable, r.kind, f"nonseptic={r.nonseptic_per_24h:.3f} septic={r.septic_per_24h:.3f} ratio={r.ratio_septic_to_nonseptic:.2f}")

# %% [markdown]
# ## Smoke test: every arm and every perturbation end to end on a few rows (fails fast on any bug)

# %%
with StageTimer("smoke test", log):
    _saved = dict(CFG)
    CFG.update(pre_epochs=1, pre_windows_per_epoch=1024, ft_max_epochs=1, ft_rows_per_epoch=1024, val_rows_cap=1500)
    _IDXsave = IDX
    _rs = np.random.default_rng(0)
    IDX = {k: np.sort(_rs.choice(v, min(len(v), 3000), replace=False)) for k, v in _IDXsave.items()}
    _te_p = np.unique(row_pidx[IDX["test"]])[:40]
    IDX["test"] = _IDXsave["test"][np.isin(row_pidx[_IDXsave["test"]], _te_p)]
    _models = {}
    for arm in MAIN_ARMS + FAC_ARMS + MISS_ARMS:
        st_ = pretrain(arm, 99, tag="_smoke") if arm in VARIANTS and VARIANTS[arm]["ssl"] != "none" else None
        pte, met, _secs, _inf, _ep, _model, _thr = run_one(arm, 0.10, 0, st_)
        if arm in ("ptc_ssl", "lightgbm"):
            _models[arm] = _model
        log.info("smoke %s ok: AUROC=%.3f ECE_cal=%.3f", arm, met["AUROC"], met["ECE_cal"])
    for name in PERTURB["names"]:
        arrs, info = make_perturbation(name, IDX["test"])
        for arm, _model in _models.items():
            with using(arrs):
                pp = predict_any(arm, _model, IDX["test"])
            log.info("smoke perturbation %s %s ok: %s AUROC=%.3f", name, arm, info, roc_auc_score(y_all[IDX["test"]], pp))
    del _models
    _bs = Boot(IDX["test"], 8, 0); _bs.add("x", pte)
    _one = _bs.run(identity=True)["x"][0]
    assert abs(_one[0] - roc_auc_score(y_all[IDX["test"]], pte)) < 1e-4 and abs(_one[1] - average_precision_score(y_all[IDX["test"]], pte)) < 1e-4
    _bs.run()
    IDX = _IDXsave
    CFG.clear(); CFG.update(_saved)
    _UCACHE.clear()
    for f in CKPT.glob("*_smoke.pt"):
        f.unlink()
set_spec("ptc_ssl")
log.info("SMOKE TEST PASSED")

# %% [markdown]
# ## Run all jobs (seed-outer, resumable, time-guarded)

# %%
RUNS = RES / "tables" / "runs_metrics_v2a.csv"
PERT = RES / "tables" / "perturbation_metrics.csv"
PRED_DIR = ART / "predictions_v2a"; PRED_DIR.mkdir(exist_ok=True)
def done_keys():
    if not RUNS.exists():
        return set()
    d = pd.read_csv(RUNS)
    return set(zip(d.arm, d.fraction.round(2), d.seed))
def pred_path(proto, arm, frac, seed):
    return PRED_DIR / f"pred_{arm}_{proto}_f{int(round(frac * 100)):03d}_s{seed}.npy"

PERT_ARRS = {}
with StageTimer("build perturbed test inputs", log):
    pinfo = []
    for name in PERTURB["names"]:
        PERT_ARRS[name], info = make_perturbation(name, IDX["test"])
        pinfo.append(info); print("PI", json.dumps(info))
    pd.DataFrame(pinfo).to_csv(RES / "tables" / "perturbation_definitions.csv", index=False)

skipped = []
for seed in SEEDS:
    pre_state = {}
    for job in JOBS:
        for frac in job["fractions"]:
            for arm in job["arms"]:
                if (arm, round(frac, 2), seed) in done_keys():
                    continue
                if (time.time() - T_START) / 3600 > CFG["time_budget_h"]:
                    skipped.append((job["name"], arm, frac, seed)); continue
                if arm in VARIANTS and VARIANTS[arm]["ssl"] != "none" and arm not in pre_state:
                    with StageTimer(f"pretrain {arm} s{seed}", log):
                        pre_state[arm] = pretrain(arm, seed)
                log.info("=== %s | %s | frac=%.2f | seed=%d | elapsed %.2f h ===", job["name"], arm, frac, seed,
                         (time.time() - T_START) / 3600)
                pte, met, secs, inf, epochs, model, thr = run_one(arm, frac, seed, pre_state.get(arm))
                rec = {"job": job["name"], "protocol": "pooled", "arm": arm, "fraction": frac, "seed": seed, "train_seconds": secs,
                       "test_inference_seconds": inf, "epochs_or_trees": epochs,
                       "n_train_patients": int(subsets[f"train_s{seed}_f{int(round(frac*100)):03d}"].sum()),
                       "n_train_septic": int((subsets[f"train_s{seed}_f{int(round(frac*100)):03d}"] & subsets["septic"].astype(bool)).sum()), **met}
                np.save(pred_path("pooled", arm, frac, seed), pte.astype(np.float32))
                if arm in PERTURB["arms"] and frac in PERTURB["fractions"] and seed in PERTURB["seeds"]:
                    for name in PERTURB["names"]:
                        with using(PERT_ARRS[name]):
                            pp = predict_any(arm, model, IDX["test"])
                            pm = evaluate(IDX["test"], pp, thr, full=False)
                        prow = {"arm": arm, "fraction": frac, "seed": seed, "perturbation": name, **pm,
                                "AUROC_clean": met["AUROC"], "AUPRC_clean": met["AUPRC"], "utility_clean": met["utility"]}
                        pd.DataFrame([prow]).to_csv(PERT, mode="a", header=not PERT.exists(), index=False)
                        print("PERT", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in prow.items()}), flush=True)
                pd.DataFrame([rec]).to_csv(RUNS, mode="a", header=not RUNS.exists(), index=False)
                print("RESULT", json.dumps({k: (round(v, 4) if isinstance(v, (float, np.floating)) else v) for k, v in rec.items()}, default=str), flush=True)
                del model
save_json({"skipped_runs": skipped}, RES / "skipped_runs_v2a.json")
del PERT_ARRS
log.info("All jobs finished; skipped=%d; elapsed %.2f h", len(skipped), (time.time() - T_START) / 3600)

# %% [markdown]
# ## Statistics: per-seed (hierarchical) and seed-ensemble paired bootstrap, factorial effects, Holm correction

# %%
runs = pd.read_csv(RUNS)
KEYM = ["AUROC", "AUPRC", "patient_AUROC", "utility", "Brier", "ECE", "Brier_cal", "ECE_cal", "sensitivity", "specificity",
        "false_alarms_per_patient_day", "non_septic_patients_with_alarm_pct", "septic_detected_pct",
        "septic_alarm_in_window_pct", "median_lead_time_h"]
agg = runs.groupby(["job", "arm", "fraction"])[KEYM].agg(["mean", "std", "count"])
agg.columns = [f"{a}_{b}" for a, b in agg.columns]
agg = agg.reset_index()
agg.to_csv(RES / "tables" / "multiseed_summary_v2a.csv", index=False)

Bn = CFG["bootstrap_B"]
with StageTimer("paired bootstrap", log):
    BS = bootstrap_protocol("pooled", runs, pred_path, Bn)
rng = np.random.default_rng(GLOBAL_SEED)
CMP = []
for frac in sorted(runs.fraction.unique()):
    for other in ["generic_ssl", "supervised_gru", "lightgbm"]:
        for r in compare(BS, "ptc_ssl", other, frac, SEEDS, rng, Bn):
            CMP.append({"family": "main", **r})
for frac in (0.05, 0.10):
    for other in FAC_ARMS + ["abl_values_only", "abl_missing_only"]:
        for r in compare(BS, "ptc_ssl", other, frac, SEEDS, rng, Bn):
            CMP.append({"family": "ablation", **r})
    for a, b in [("abl_missing_only", "abl_values_only"), ("supervised_missing_only", "supervised_values_only"),
                 ("lgbm_missing_only", "lgbm_values_only"), ("supervised_missing_only", "supervised_gru"),
                 ("lgbm_missing_only", "lightgbm"), ("abl_missing_only", "supervised_missing_only")]:
        for r in compare(BS, a, b, frac, SEEDS, rng, Bn):
            CMP.append({"family": "missingness", **r})
cmp_df = pd.DataFrame(CMP)
for fam, g in cmp_df.groupby("family"):
    cmp_df.loc[g.index, "perseed_p_holm"] = holm(g.perseed_p.values)
    cmp_df.loc[g.index, "ensemble_p_holm"] = holm(g.ensemble_p.values)
cmp_df.to_csv(RES / "tables" / "comparisons_v2a.csv", index=False)
fx = pd.DataFrame([r for frac in (0.05, 0.10) for r in factorial_effects(BS, frac, SEEDS, rng, Bn)])
if len(fx):
    for m, g in fx.groupby("metric"):
        fx.loc[g.index, "p_holm"] = holm(g.p.values)
fx.to_csv(RES / "tables" / "factorial_effects.csv", index=False)

pert = pd.read_csv(PERT) if PERT.exists() else pd.DataFrame()
if len(pert):
    pert["dAUROC"] = pert.AUROC - pert.AUROC_clean
    pert["dAUPRC"] = pert.AUPRC - pert.AUPRC_clean
    pert["dutility"] = pert.utility - pert.utility_clean
    pagg = pert.groupby(["arm", "fraction", "perturbation"])[["dAUROC", "dAUPRC", "dutility"]].agg(["mean", "std", "count"])
    pagg.columns = [f"{a}_{b}" for a, b in pagg.columns]
    pagg = pagg.reset_index()
    pagg.to_csv(RES / "tables" / "perturbation_summary.csv", index=False)

# %%
import matplotlib.pyplot as plt
main = agg[agg.job == "main"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, met in zip(axes, ["AUROC", "AUPRC", "utility"]):
    for arm, g in main.groupby("arm"):
        g = g.sort_values("fraction")
        ax.errorbar(g.fraction * 100, g[f"{met}_mean"], yerr=g[f"{met}_std"], marker="o", capsize=3, label=arm)
    ax.set_xscale("log"); ax.set_xlabel("% labelled training patients"); ax.set_title(f"{met} (mean ± SD over seeds)")
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(RES / "figures" / f"fig_label_efficiency_v2a.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Compact printout (parsed for the paper) and result package

# %%
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 1000)
print("\n==== MULTISEED SUMMARY v2a ====")
for _, r in agg.iterrows():
    print("MS", r.job, r.arm, r.fraction, " ".join(f"{m}={r[m + '_mean']:.4f}|{r[m + '_std']:.4f}|{int(r[m + '_count'])}" for m in KEYM))
print("\n==== COMPARISONS (per-seed hierarchical | seed-ensemble) ====")
for _, r in cmp_df.iterrows():
    print("CP", r.family, r.fraction, r.comparison, r.metric, f"n={r.n_seeds}",
          f"perseed={r.perseed_diff:+.4f} [{r.perseed_lo:+.4f},{r.perseed_hi:+.4f}] p={r.perseed_p:.4f} holm={r.perseed_p_holm:.4f}",
          f"ens={r.ensemble_diff:+.4f} [{r.ensemble_lo:+.4f},{r.ensemble_hi:+.4f}] p={r.ensemble_p:.4f} holm={r.ensemble_p_holm:.4f}")
print("\n==== FACTORIAL EFFECTS ====")
for _, r in fx.iterrows():
    print("FX", r.fraction, r.term, r.metric, f"n={r.n_seeds} effect={r.effect:+.4f} [{r.lo:+.4f},{r.hi:+.4f}] p={r.p:.4f} holm={r.p_holm:.4f}")
if len(pert):
    print("\n==== PERTURBATIONS (change from clean test set; mean | sd | n) ====")
    for _, r in pagg.iterrows():
        print("PS", r.arm, r.fraction, r.perturbation, " ".join(f"{m}={r[m + '_mean']:+.4f}|{r[m + '_std']:.4f}|{int(r[m + '_count'])}" for m in ("dAUROC", "dAUPRC", "dutility")))
print("\nSKIPPED:", skipped)

with zipfile.ZipFile(WORK / "PTC_SSL_V2A_RESULTS.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in list(RES.rglob("*")) + list(LOGS.glob("*")):
        if f.is_file():
            z.write(f, f.relative_to(WORK))
for f in ART.glob("X_*.npy"):
    f.unlink()
print("✅ 07a complete. Elapsed %.2f h" % ((time.time() - T_START) / 3600))
