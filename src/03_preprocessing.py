# %% [markdown]
# # 03 · Preprocessing: cohort, QC, patient-level splits, train-only normalisation
# **Inputs:** outputs of `01_dataset_acquisition` and `02_data_profiling`.
#
# **Design decisions (fixed before modelling):**
# - **Cohort exclusion E1:** septic patients whose `SepsisLabel` is already 1 at the first ICU row (onset ≤ 6 h after the record starts), because early prediction is not defined for them. Counts are reported, and the official-utility sensitivity analysis in `11` can include them again.
# - **QC:** implausible values are set to missing using the unit-verified ranges from `02`. FiO2 percentages are converted to fractions only if `02` verified this.
# - **Features:** 34 time-varying variables, plus Age, Gender and log(ICULOS). `Unit1/Unit2/HospAdmTime` are left out: they are administrative and hospital-specific, so they could act as shortcuts under the A→B / B→A shift.
# - **Splits (patient-level, stratified by hospital × septic, fixed seed):**
#   - `pooled`: A∪B → 70 / 15 / 15 train / val / test.
#   - `A2B`: A → 85 / 15 train / val; test = all eligible B.
#   - `B2A`: B → 85 / 15 train / val; test = all eligible A.
# - **SSL pretraining pool = the training patients of each protocol only.** Validation and test patients never enter pretraining.
# - **Normalisation:** fitted per protocol on *training patients only*, using raw observations (each measurement counted once, no forward-filled copies). Skewed labs are log1p-transformed, then winsorised at the training 0.5/99.5 percentiles and z-scored.
# - **Stored representation:** raw QC'd values (NaN = missing), observation mask, causal last-observation-carried-forward (LOCF) values, and hours since the last observation. LOCF and age are strictly causal (no back-filling).
#
# Runs on CPU in about 3–6 min.

# %% [common]

# %%
NB = "03_preprocessing"
log = get_logger(NB)
hourly = load_table(find_artifact("physionet2019_hourly.parquet", "01_dataset_acquisition"))
patients = load_table(find_artifact("patients.parquet", "01_dataset_acquisition"))
qc_cfg = load_json(find_artifact("qc_ranges.json", "02_data_profiling"))
cons = load_json(find_artifact("constraints_validated.json", "02_data_profiling"))

CFG = {
    "exclude_label_at_first_row": True,
    "split_seed": 2026,
    "pooled_fracs": {"train": 0.70, "val": 0.15, "test": 0.15},
    "external_train_frac": 0.85,
    "static_features": ["Age", "Gender", "log_ICULOS"],
    "excluded_columns": ["Unit1", "Unit2", "HospAdmTime"],
    "log1p_vars": ["AST", "BUN", "Alkalinephos", "Creatinine", "Bilirubin_direct", "Bilirubin_total", "Glucose",
                   "Lactate", "TroponinI", "WBC", "Fibrinogen", "Platelets", "PTT"],
    "winsor_pct": [0.5, 99.5],
    "max_age_h": 168.0,
}
ORGAN_SYSTEMS = {   # every dynamic variable belongs to exactly one system
    "cardiovascular": ["HR", "SBP", "MAP", "DBP", "TroponinI"],
    "respiratory": ["O2Sat", "Resp", "EtCO2", "FiO2", "SaO2", "PaCO2"],
    "acid_base_perfusion": ["pH", "HCO3", "BaseExcess", "Lactate"],
    "renal_electrolyte": ["BUN", "Creatinine", "Potassium", "Chloride"],
    "hepatic": ["AST", "Alkalinephos", "Bilirubin_direct", "Bilirubin_total"],
    "hematologic_coagulation": ["Hct", "Hgb", "Platelets", "PTT", "Fibrinogen"],
    "metabolic": ["Glucose", "Calcium", "Magnesium", "Phosphate"],
    "inflammation_thermoregulation": ["Temp", "WBC"],
}
flat = [v for vs in ORGAN_SYSTEMS.values() for v in vs]
assert sorted(flat) == sorted(DYNAMIC_VARS) and len(flat) == len(set(flat)) == 34, "organ map must partition the 34 vars"
print(json.dumps(CFG, indent=1))

# %% [markdown]
# ## 1. Cohort and exclusions

# %%
patients = patients.sort_values(["hospital", "patient_id"]).reset_index(drop=True)
patients["excl_label_at_first_row"] = (patients.septic == 1) & (patients.first_label_idx == 0)
patients["eligible"] = ~patients["excl_label_at_first_row"] if CFG["exclude_label_at_first_row"] else True
flow = patients.groupby("hospital").agg(records=("patient_id", "size"),
                                        excluded_E1=("excl_label_at_first_row", "sum"),
                                        eligible=("eligible", "sum"))
flow["eligible_septic"] = patients[patients.eligible].groupby("hospital").septic.sum()
display(flow)
flow.to_csv(RES / "tables" / "cohort_flow.csv")

# %% [markdown]
# ## 2. Patient-level stratified splits

# %%
def strat_split(df, fracs, seed):
    """Assign split names to rows of df, stratified by (hospital, septic). Exact, deterministic."""
    rng = np.random.default_rng(seed)
    out = pd.Series(index=df.index, dtype=object)
    names = list(fracs)
    for _, grp in df.groupby(["hospital", "septic"]):
        idx = grp.index.values.copy()
        rng.shuffle(idx)
        n = len(idx)
        cuts = np.floor(np.cumsum([fracs[k] for k in names]) * n + 1e-9).astype(int)
        cuts[-1] = n
        start = 0
        for name, c in zip(names, cuts):
            out.loc[idx[start:c]] = name
            start = c
    return out

elig = patients[patients.eligible]
for proto in ("pooled", "A2B", "B2A"):
    patients[f"split_{proto}"] = "excluded"
patients.loc[elig.index, "split_pooled"] = strat_split(elig, CFG["pooled_fracs"], CFG["split_seed"])
ext = {"train": CFG["external_train_frac"], "val": 1 - CFG["external_train_frac"]}
for proto, src, tgt in (("A2B", "A", "B"), ("B2A", "B", "A")):
    s = elig[elig.hospital == src]
    patients.loc[s.index, f"split_{proto}"] = strat_split(s, ext, CFG["split_seed"])
    patients.loc[elig[elig.hospital == tgt].index, f"split_{proto}"] = "test"

PROTOCOLS = ["pooled", "A2B", "B2A"]
split_tab = []
for proto in PROTOCOLS:
    col = f"split_{proto}"
    for sp in ("train", "val", "test"):
        p = patients[patients[col] == sp]
        split_tab.append({"protocol": proto, "split": sp, "patients": len(p), "septic": int(p.septic.sum()),
                          "septic_pct": 100 * p.septic.mean() if len(p) else np.nan,
                          "hospitals": ",".join(sorted(p.hospital.unique()))})
    # leakage guards
    sets = {sp: set(patients.loc[patients[col] == sp, "patient_id"]) for sp in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert patients.loc[~patients.eligible, col].eq("excluded").all()
assert set(patients.loc[patients.split_A2B == "test", "hospital"]) <= {"B"}
assert set(patients.loc[patients.split_A2B.isin(["train", "val"]), "hospital"]) <= {"A"}
assert set(patients.loc[patients.split_B2A == "test", "hospital"]) <= {"A"}
split_tab = pd.DataFrame(split_tab)
split_tab.to_csv(RES / "tables" / "split_summary.csv", index=False)
display(split_tab)
log.info("Splits disjoint and hospital-consistent for all protocols")

# %% [markdown]
# ## 3. QC: implausible values set to missing

# %%
hourly = hourly.merge(patients[["patient_id"]].reset_index().rename(columns={"index": "pidx"}),
                      on="patient_id", how="left", validate="many_to_one")
hourly = hourly.sort_values(["pidx", "t"]).reset_index(drop=True)
assert hourly["pidx"].notna().all()

qc_log = []
for v in DYNAMIC_VARS:
    x = hourly[v].astype(np.float32)
    n_obs0 = int(x.notna().sum())
    n_fix = 0
    if v == "FiO2" and qc_cfg["fio2_percent_to_fraction"]:
        m = (x > 1) & (x <= 100)
        n_fix = int(m.sum())
        x[m] = x[m] / 100.0
    lo, hi = qc_cfg["ranges"][v]
    bad = (x < lo) | (x > hi)
    x[bad] = np.nan
    hourly[v] = x
    qc_log.append({"variable": v, "unit": qc_cfg["units"][v], "observed_before": n_obs0, "fio2_converted": n_fix,
                   "set_missing_implausible": int(bad.sum()), "observed_after": int(x.notna().sum())})
qc_log = pd.DataFrame(qc_log)
qc_log.to_csv(RES / "tables" / "qc_applied.csv", index=False)
display(qc_log[qc_log.set_missing_implausible > 0])

# %% [markdown]
# ## 4. Causal arrays: values, mask, LOCF values, hours since last observation

# %%
with StageTimer("arrays", log):
    V = hourly[DYNAMIC_VARS].to_numpy(np.float32)                 # raw QC'd, NaN = missing
    M = (~np.isnan(V)).astype(np.uint8)
    pidx = hourly["pidx"].to_numpy(np.int64)
    t = hourly["t"].to_numpy(np.int64)
    # patient offsets (rows are contiguous per patient, ordered by t)
    starts = np.flatnonzero(np.r_[True, pidx[1:] != pidx[:-1]])
    lengths = np.diff(np.r_[starts, len(pidx)])
    assert len(starts) == len(patients) and (pidx[starts] == np.arange(len(patients))).all()
    assert (lengths == patients["n_rows"].values).all()
    # causal LOCF + age via groupby ffill (never bfill)
    dfv = pd.DataFrame(V, columns=DYNAMIC_VARS)
    dfv["pidx"] = pidx
    F = dfv.groupby("pidx")[DYNAMIC_VARS].ffill().to_numpy(np.float32)
    last_t = pd.DataFrame(np.where(M == 1, t[:, None], np.nan), columns=DYNAMIC_VARS)
    last_t["pidx"] = pidx
    last_t = last_t.groupby("pidx")[DYNAMIC_VARS].ffill().to_numpy(np.float32)
    AGE = (t[:, None] - last_t).astype(np.float32)               # NaN before first observation
    AGE = np.minimum(AGE, CFG["max_age_h"])

# Causality self-test: LOCF at row r equals the last observed value at or before r within the same patient
rng = np.random.default_rng(0)
for r in rng.choice(len(V), size=min(2000, len(V)), replace=False):
    s = starts[pidx[r]]
    for j in rng.choice(34, 3, replace=False):
        hist = V[s:r + 1, j]
        ok = np.where(~np.isnan(hist))[0]
        exp = hist[ok[-1]] if len(ok) else np.nan
        assert (np.isnan(exp) and np.isnan(F[r, j])) or np.isclose(exp, F[r, j]), (r, j)
        exp_age = (r - s - ok[-1]) if len(ok) else np.nan
        assert (np.isnan(exp_age) and np.isnan(AGE[r, j])) or np.isclose(min(exp_age, CFG["max_age_h"]), AGE[r, j])
log.info("Causal LOCF / age self-test passed")

# %% [markdown]
# ## 5. Train-only normalisation statistics (per protocol)

# %%
def fit_norm(train_rows):
    stats = {}
    for j, v in enumerate(DYNAMIC_VARS):
        x = V[train_rows, j]
        x = x[~np.isnan(x)]
        use_log = v in CFG["log1p_vars"]
        if use_log:
            x = np.log1p(np.clip(x, 0, None))
        if len(x) < 2:
            stats[v] = {"log1p": use_log, "lo": 0.0, "hi": 0.0, "mean": 0.0, "std": 1.0, "n": int(len(x))}
            continue
        lo, hi = np.percentile(x, CFG["winsor_pct"])
        xc = np.clip(x, lo, hi)
        sd = float(xc.std())
        stats[v] = {"log1p": use_log, "lo": float(lo), "hi": float(hi), "mean": float(xc.mean()),
                    "std": sd if sd > 1e-6 else 1.0, "n": int(len(x))}
    age = patients.loc[np.unique(pidx[train_rows]), "age"].to_numpy(float)
    icu = np.log1p(hourly.loc[train_rows, "ICULOS"].to_numpy(float).clip(min=0))
    stats["_static"] = {"Age": {"mean": float(np.nanmean(age)), "std": float(np.nanstd(age) or 1.0)},
                        "log_ICULOS": {"mean": float(np.nanmean(icu)), "std": float(np.nanstd(icu) or 1.0)}}
    return stats

def normalize_values(X, stats):
    """Reference numpy implementation (torch version in later notebooks must match it)."""
    out = np.empty_like(X, dtype=np.float32)
    for j, v in enumerate(DYNAMIC_VARS):
        s = stats[v]
        x = X[:, j].astype(np.float64)
        if s["log1p"]:
            x = np.log1p(np.clip(x, 0, None))
        out[:, j] = ((np.clip(x, s["lo"], s["hi"]) - s["mean"]) / s["std"]).astype(np.float32)
    return out

norm = {}
for proto in PROTOCOLS:
    train_p = patients.index[patients[f"split_{proto}"] == "train"].values
    train_rows = np.isin(pidx, train_p)
    norm[proto] = fit_norm(train_rows)
    save_json(norm[proto], ART / f"norm_stats_{proto}.json")
    save_json(norm[proto], RES / "configs" / f"norm_stats_{proto}.json")
    # guard: recompute must not change if test rows are perturbed (statistics depend on train rows only)
    test_rows = np.isin(pidx, patients.index[patients[f"split_{proto}"] == "test"].values)
    V_backup = V[test_rows].copy()
    V[test_rows] = V[test_rows] * 10 + 7
    assert fit_norm(train_rows) == norm[proto], "normalisation depends on test rows!"
    V[test_rows] = V_backup
log.info("Train-only normalisation fitted and leakage-tested for %s", PROTOCOLS)

# %% [markdown]
# ## 6. Constraint statistics for the physiology loss (training patients only)
# For each accepted constraint, the median offset and robust scale of its residual are estimated on **training** patients of each protocol.
# `07` uses them to make the physiology loss unit-free: `((residual − offset) / scale)`, with a tolerance band.

# %%
VIDX = {v: j for j, v in enumerate(DYNAMIC_VARS)}
RESIDUALS = {   # identical relations to notebook 02 (raw units)
    "C1_MAP_formula": lambda X: X[:, VIDX["MAP"]] - (X[:, VIDX["SBP"]] + 2 * X[:, VIDX["DBP"]]) / 3,
    "C2_BP_ordering": lambda X: np.maximum(X[:, VIDX["MAP"]] - X[:, VIDX["SBP"]], X[:, VIDX["DBP"]] - X[:, VIDX["MAP"]]),
    "C3_Henderson_Hasselbalch": lambda X: X[:, VIDX["pH"]] - (6.1 + np.log10(X[:, VIDX["HCO3"]] / (0.03 * X[:, VIDX["PaCO2"]]))),
    "C4_Hct_3xHgb": lambda X: X[:, VIDX["Hct"]] - 3 * X[:, VIDX["Hgb"]],
    "C5_bilirubin_direct_le_total": lambda X: X[:, VIDX["Bilirubin_direct"]] - X[:, VIDX["Bilirubin_total"]],
    "C6_SaO2_vs_SpO2": lambda X: X[:, VIDX["SaO2"]] - X[:, VIDX["O2Sat"]],
}
assert set(RESIDUALS) == set(cons["definitions"]), "constraint sets of 02 and 03 differ"
cons_stats = {}
for proto in PROTOCOLS:
    train_rows = np.isin(pidx, patients.index[patients[f"split_{proto}"] == "train"].values)
    Xtr = V[train_rows]
    cons_stats[proto] = {}
    for cid in cons["accepted"]:
        d = cons["definitions"][cid]
        cols = [VIDX[v] for v in d["vars"]]
        ok = ~np.isnan(Xtr[:, cols]).any(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = RESIDUALS[cid](Xtr[ok])
        r = r[np.isfinite(r)]
        med = float(np.median(r)) if len(r) else 0.0
        mad = float(1.4826 * np.median(np.abs(r - med))) if len(r) else 1.0
        cons_stats[proto][cid] = {"vars": d["vars"], "kind": d["kind"], "tol": d["tol"], "unit": d["unit"],
                                  "n_train": int(len(r)), "offset": med if d["kind"] == "eq" else 0.0,
                                  "scale": mad if mad > 1e-6 else 1.0}
save_json({"accepted": cons["accepted"], "per_protocol": cons_stats}, ART / "physiology_constraints.json")
save_json({"accepted": cons["accepted"], "per_protocol": cons_stats}, RES / "configs" / "physiology_constraints.json")
save_json(ORGAN_SYSTEMS, ART / "organ_systems.json")
save_json(ORGAN_SYSTEMS, RES / "configs" / "organ_systems.json")
display(pd.DataFrame({p: {c: f"off={s['offset']:.3g}, scale={s['scale']:.3g}, n={s['n_train']}"
                          for c, s in d.items()} for p, d in cons_stats.items()}))

# %% [markdown]
# ## 7. Save arrays and tables

# %%
with StageTimer("save", log):
    np.save(ART / "X_values.npy", V)
    np.save(ART / "X_mask.npy", M)
    np.save(ART / "X_locf.npy", F.astype(np.float32))
    np.save(ART / "X_age.npy", AGE.astype(np.float16))
    np.save(ART / "patient_starts.npy", starts.astype(np.int64))
    np.save(ART / "patient_lengths.npy", lengths.astype(np.int64))
    rows_meta = pd.DataFrame({"pidx": pidx.astype(np.int32), "t": t.astype(np.int32),
                              "ICULOS": hourly["ICULOS"].to_numpy(np.float32),
                              "SepsisLabel": hourly["SepsisLabel"].to_numpy(np.int8)})
    save_table(rows_meta, ART / "rows_meta.parquet")
    save_table(patients, ART / "patients_split.parquet")
save_json({"config": CFG, "dynamic_vars": DYNAMIC_VARS, "n_rows": int(len(V)), "n_patients": int(len(patients)),
           "arrays": {"X_values": "float32 raw QC'd, NaN=missing", "X_mask": "uint8 1=observed",
                      "X_locf": "float32 causal LOCF, NaN before first obs",
                      "X_age": f"float16 hours since last obs (cap {CFG['max_age_h']}), NaN before first obs"}},
          ART / "preprocessing_config.json")
save_json(CFG, RES / "configs" / "preprocessing_config.json")
print("✅ Preprocessing complete. Next: 04_temporal_windowing (attach outputs of 01–03).")
