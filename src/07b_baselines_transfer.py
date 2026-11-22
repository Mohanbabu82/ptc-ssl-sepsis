# %% [markdown]
# # 07b · Stronger baselines, sensitivity analysis and cross-hospital diagnosis (revision 2)
# Runs after 01–04 in the same session. Settings fixed before these runs; the test set is never used for tuning.
#
# 1. **Other SSL objectives and encoders** (pooled, 5 / 10 / 100% labels, seeds 0–4): TS2Vec-style hierarchical
#    contrastive pretraining and SimMTM-style multi-mask reconstruction, both re-implemented on the SAME GRU encoder,
#    inputs, windows, pretraining data and budget; a Transformer encoder trained supervised and with PTC-SSL.
#    The four main arms are re-run here so every comparison is paired within this notebook.
# 2. **Sensitivity** (pooled, 5% labels, seeds 0–4): λ_phys and λ_temp ∈ {0, 0.01, 0.05, 0.1, 0.5, 1}, GRU hidden
#    size ∈ {64, 96, 128, 256}, mask block ∈ {4, 8, 12, 16} h, masking ∈ {random, variable-based, organ-based}.
# 3. **Cross-hospital diagnosis** (A→B and B→A, seeds 0–4): the target hospital is split by patient into an
#    unlabelled *adaptation* half and an *evaluation* half; every arm is scored on the evaluation half only.
#    Arms: the four main arms (10% and 100% labels) and, at 10% labels, missingness-only PTC-SSL, PTC-SSL without
#    FiO₂, and generic SSL / PTC-SSL whose pretraining also sees the unlabelled adaptation half (unsupervised
#    domain adaptation; target labels are never read).
# 4. **Hospital shift analysis**: measurement rates, value distributions (standardised mean difference, KS),
#    a hospital classifier on values-only vs missingness-only features, and calibration in the large per direction.

# %% [common]

# %%
NB = "07b_baselines_transfer"
log = get_logger(NB)

# %% [lib]

# %%
SEEDS = list(range(int(os.environ.get("PTCSSL_SEEDS", 5))))
MAIN_ARMS = ["lightgbm", "supervised_gru", "generic_ssl", "ptc_ssl"]
SSL_ARMS = ["ts2vec_gru", "simmtm_gru", "supervised_tf", "ptc_tf"]
SENS_ARMS = ([f"sens_lp_{v:g}" for v in (0.0, 0.01, 0.05, 0.5, 1.0)] + [f"sens_lt_{v:g}" for v in (0.0, 0.01, 0.05, 0.5, 1.0)]
             + [f"sens_hidden_{h}" for h in (64, 128, 256)] + [f"sens_block_{b}" for b in (4, 12, 16)]
             + ["sens_mask_random", "sens_mask_variable"])
EXT_EXTRA = ["abl_missing_only", "ptc_no_fio2", "generic_uda", "ptc_uda"]
POOLED_JOBS = [
    {"name": "main", "fractions": [0.05, 0.10, 1.00], "arms": MAIN_ARMS},
    {"name": "ssl_baselines", "fractions": [0.05, 0.10, 1.00], "arms": SSL_ARMS},
    {"name": "sensitivity", "fractions": [0.05], "arms": SENS_ARMS},
]
EXT_JOBS = [
    {"name": "external", "fractions": [0.10, 1.00], "arms": MAIN_ARMS},
    {"name": "external_diag", "fractions": [0.10], "arms": EXT_EXTRA},
]
if os.environ.get("PTCSSL_QUICK"):
    SEEDS = SEEDS[:2]
save_json({"CFG": CFG, "SEEDS": SEEDS, "POOLED_JOBS": POOLED_JOBS, "EXT_JOBS": EXT_JOBS, "VARIANTS": VARIANTS},
          RES / "configs" / "experiment_config_v2b.json")
use_protocol("pooled")
set_spec("ptc_ssl")
IN_DIM = causality_check()
log.info("Batch builder causality check passed; input dim=%d", IN_DIM)

# %% [markdown]
# ## Hospital shift analysis (descriptive; training-split patients of each hospital)

# %%
def ks_stat(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a) - np.searchsorted(b, grid, side="right") / len(b))))

with StageTimer("hospital shift analysis", log):
    hosp = patients["hospital"].to_numpy()[row_pidx]
    rs = np.random.default_rng(GLOBAL_SEED)
    shift = []
    for j, v in enumerate(DYNAMIC_VARS):
        rec = {"variable": v, "kind": "lab" if v in LABS else "vital"}
        for h in ("A", "B"):
            rows_h = hosp == h
            rec[f"meas_rate_{h}"] = float(M[rows_h, j].mean())
            vals = V[rows_h & M[:, j], j]
            rec[f"median_{h}"] = float(np.median(vals)) if len(vals) else np.nan
            rec[f"_vals_{h}"] = rs.choice(vals, min(len(vals), 20000), replace=False) if len(vals) else np.array([])
        a, b = rec.pop("_vals_A"), rec.pop("_vals_B")
        if len(a) > 50 and len(b) > 50:
            rec["smd"] = float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12))
            rec["ks"] = ks_stat(a, b)
        rec["meas_rate_ratio_A_over_B"] = rec["meas_rate_A"] / max(rec["meas_rate_B"], 1e-9)
        shift.append(rec)
    shift = pd.DataFrame(shift)
    shift.to_csv(RES / "tables" / "hospital_shift_by_variable.csv", index=False)
    for _, r in shift.iterrows():
        print("SH", r.variable, r.kind, f"rateA={r.meas_rate_A:.4f} rateB={r.meas_rate_B:.4f} ratio={r.meas_rate_ratio_A_over_B:.2f}",
              f"medA={r.median_A:.3g} medB={r.median_B:.3g} smd={r.get('smd', np.nan):+.3f} ks={r.get('ks', np.nan):.3f}")
    # how well can the HOSPITAL be identified from values-only vs measurement-pattern-only features? (patient-level split)
    tr_p = patients.index[patients["split_pooled"] == "train"].to_numpy()
    te_p = patients.index[patients["split_pooled"] == "test"].to_numpy()
    Rtr = IDX["train"][rs.choice(len(IDX["train"]), min(200_000, len(IDX["train"])), replace=False)]
    Rte = IDX["test"][rs.choice(len(IDX["test"]), min(100_000, len(IDX["test"])), replace=False)]
    yh = lambda R: (patients["hospital"].to_numpy()[row_pidx[R]] == "B").astype(int)
    for mode in ("values", "missing", "full"):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, random_state=0, verbose=-1, n_jobs=-1)
        clf.fit(tab_features(Rtr, mode), yh(Rtr))
        auc = roc_auc_score(yh(Rte), clf.predict_proba(tab_features(Rte, mode))[:, 1])
        print("HC", mode, f"hospital_classifier_AUROC={auc:.4f}")
        pd.DataFrame([{"features": mode, "hospital_AUROC": auc}]).to_csv(RES / "tables" / "hospital_classifier.csv", mode="a",
                                                                       header=not (RES / "tables" / "hospital_classifier.csv").exists(), index=False)

# %% [markdown]
# ## Smoke test (every arm, both protocols, a few rows)

# %%
with StageTimer("smoke test", log):
    _saved = dict(CFG)
    CFG.update(pre_epochs=1, pre_windows_per_epoch=1024, ft_max_epochs=1, ft_rows_per_epoch=1024, val_rows_cap=1500)
    for proto, arms in (("pooled", MAIN_ARMS + SSL_ARMS + SENS_ARMS), ("A2B", MAIN_ARMS + EXT_EXTRA)):
        use_protocol(proto, uda_split=(proto != "pooled"))
        _IDXsave = IDX
        _rs = np.random.default_rng(0)
        IDX = {k: np.sort(_rs.choice(v, min(len(v), 3000), replace=False)) for k, v in _IDXsave.items()}
        _te_p = np.unique(row_pidx[_IDXsave["test"]])[:40]
        IDX["test"] = _IDXsave["test"][np.isin(row_pidx[_IDXsave["test"]], _te_p)]
        for arm in arms:
            st_ = pretrain(arm, 99, tag="_smoke") if arm in VARIANTS and VARIANTS[arm]["ssl"] != "none" else None
            pte, met, *_ = run_one(arm, 0.10, 0, st_)
            log.info("smoke %s %s ok: AUROC=%.3f", proto, arm, met["AUROC"])
        _bs = Boot(IDX["test"], 8, 0); _bs.add("x", pte)
        _one = _bs.run(identity=True)["x"][0]
        assert abs(_one[0] - roc_auc_score(y_all[IDX["test"]], pte)) < 1e-4
        IDX = _IDXsave
        _UCACHE.clear()
        for f in CKPT.glob("*_smoke.pt"):
            f.unlink()
    CFG.clear(); CFG.update(_saved)
use_protocol("pooled")
log.info("SMOKE TEST PASSED")

# %% [markdown]
# ## Run all jobs (seed-outer, resumable, time-guarded)

# %%
RUNS = RES / "tables" / "runs_metrics_v2b.csv"
PRED_DIR = ART / "predictions_v2b"; PRED_DIR.mkdir(exist_ok=True)
def done_keys():
    if not RUNS.exists():
        return set()
    d = pd.read_csv(RUNS)
    return set(zip(d.protocol, d.arm, d.fraction.round(2), d.seed))
def pred_path(proto, arm, frac, seed):
    return PRED_DIR / f"pred_{arm}_{proto}_f{int(round(frac * 100)):03d}_s{seed}.npy"

skipped = []
for seed in SEEDS:
    for proto, jobs in (("pooled", POOLED_JOBS), ("A2B", EXT_JOBS), ("B2A", EXT_JOBS)):
        if P != proto:
            use_protocol(proto, uda_split=(proto != "pooled")); _UCACHE.clear()
        pre_state = {}
        for job in jobs:
            for frac in job["fractions"]:
                for arm in job["arms"]:
                    if (proto, arm, round(frac, 2), seed) in done_keys():
                        continue
                    if (time.time() - T_START) / 3600 > CFG["time_budget_h"]:
                        skipped.append((job["name"], proto, arm, frac, seed)); continue
                    if arm in VARIANTS and VARIANTS[arm]["ssl"] != "none" and arm not in pre_state:
                        with StageTimer(f"pretrain {arm} {proto} s{seed}", log):
                            pre_state[arm] = pretrain(arm, seed)
                    log.info("=== %s | %s | %s | frac=%.2f | seed=%d | elapsed %.2f h ===", job["name"], proto, arm, frac, seed,
                             (time.time() - T_START) / 3600)
                    pte, met, secs, inf, epochs, model, thr = run_one(arm, frac, seed, pre_state.get(arm))
                    rec = {"job": job["name"], "protocol": proto, "arm": arm, "fraction": frac, "seed": seed, "train_seconds": secs,
                           "test_inference_seconds": inf, "epochs_or_trees": epochs,
                           "n_train_patients": int(subsets[f"train_s{seed}_f{int(round(frac*100)):03d}"].sum()),
                           "n_test_rows": len(IDX["test"]), **met}
                    np.save(pred_path(proto, arm, frac, seed), pte.astype(np.float32))
                    pd.DataFrame([rec]).to_csv(RUNS, mode="a", header=not RUNS.exists(), index=False)
                    print("RESULT", json.dumps({k: (round(v, 4) if isinstance(v, (float, np.floating)) else v) for k, v in rec.items()}, default=str), flush=True)
                    del model
save_json({"skipped_runs": skipped}, RES / "skipped_runs_v2b.json")
log.info("All jobs finished; skipped=%d; elapsed %.2f h", len(skipped), (time.time() - T_START) / 3600)

# %% [markdown]
# ## Statistics

# %%
runs = pd.read_csv(RUNS)
KEYM = ["AUROC", "AUPRC", "patient_AUROC", "utility", "Brier", "ECE", "Brier_cal", "ECE_cal", "mean_pred", "mean_pred_cal",
        "obs_rate", "sensitivity", "specificity", "false_alarms_per_patient_day", "septic_detected_pct", "median_lead_time_h"]
agg = runs.groupby(["job", "protocol", "arm", "fraction"])[KEYM].agg(["mean", "std", "count"])
agg.columns = [f"{a}_{b}" for a, b in agg.columns]
agg = agg.reset_index()
agg.to_csv(RES / "tables" / "multiseed_summary_v2b.csv", index=False)

Bn = CFG["bootstrap_B"]
rng = np.random.default_rng(GLOBAL_SEED)
CMP = []
for proto in ("pooled", "A2B", "B2A"):
    sub = runs[runs.protocol == proto]
    if sub.empty:
        continue
    use_protocol(proto, uda_split=(proto != "pooled"))
    with StageTimer(f"paired bootstrap {proto}", log):
        BS = bootstrap_protocol(proto, sub, pred_path, Bn)
    if proto == "pooled":
        pairs = [(f, "ptc_ssl", o, "baselines") for f in (0.05, 0.10, 1.00)
                 for o in ["generic_ssl", "supervised_gru", "lightgbm", "ts2vec_gru", "simmtm_gru"]]
        pairs += [(f, "ptc_tf", "supervised_tf", "encoder") for f in (0.05, 0.10, 1.00)]
        pairs += [(f, "ptc_tf", "ptc_ssl", "encoder") for f in (0.05, 0.10, 1.00)]
        pairs += [(0.05, a, "ptc_ssl", "sensitivity") for a in SENS_ARMS]
    else:
        pairs = [(f, "ptc_ssl", o, "external") for f in (0.10, 1.00) for o in ["generic_ssl", "supervised_gru", "lightgbm"]]
        pairs += [(0.10, a, "ptc_ssl", "external_diag") for a in EXT_EXTRA] + [(0.10, "generic_uda", "generic_ssl", "external_diag")]
    for frac, a, b, fam in pairs:
        for r in compare(BS, a, b, frac, SEEDS, rng, Bn):
            CMP.append({"protocol": proto, "family": fam, **r})
    del BS
cmp_df = pd.DataFrame(CMP)
for (proto, fam), g in cmp_df.groupby(["protocol", "family"]):
    cmp_df.loc[g.index, "perseed_p_holm"] = holm(g.perseed_p.values)
    cmp_df.loc[g.index, "ensemble_p_holm"] = holm(g.ensemble_p.values)
cmp_df.to_csv(RES / "tables" / "comparisons_v2b.csv", index=False)

# %%
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 1000)
print("\n==== MULTISEED SUMMARY v2b ====")
for _, r in agg.iterrows():
    print("MS", r.job, r.protocol, r.arm, r.fraction, " ".join(f"{m}={r[m + '_mean']:.4f}|{r[m + '_std']:.4f}|{int(r[m + '_count'])}" for m in KEYM))
print("\n==== COMPARISONS (per-seed hierarchical | seed-ensemble) ====")
for _, r in cmp_df.iterrows():
    print("CP", r.protocol, r.family, r.fraction, r.comparison, r.metric, f"n={r.n_seeds}",
          f"perseed={r.perseed_diff:+.4f} [{r.perseed_lo:+.4f},{r.perseed_hi:+.4f}] p={r.perseed_p:.4f} holm={r.perseed_p_holm:.4f}",
          f"ens={r.ensemble_diff:+.4f} [{r.ensemble_lo:+.4f},{r.ensemble_hi:+.4f}] p={r.ensemble_p:.4f} holm={r.ensemble_p_holm:.4f}")
print("\nSKIPPED:", skipped)

with zipfile.ZipFile(WORK / "PTC_SSL_V2B_RESULTS.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in list(RES.rglob("*")) + list(LOGS.glob("*")):
        if f.is_file():
            z.write(f, f.relative_to(WORK))
for f in ART.glob("X_*.npy"):
    f.unlink()
print("✅ 07b complete. Elapsed %.2f h" % ((time.time() - T_START) / 3600))
