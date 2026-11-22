# %% [markdown]
# # 04 · Causal temporal windows, targets and label-efficiency subsets
# **Inputs:** outputs of `01`–`03`.
#
# **Design**
# - A sample is one **(patient, hour r)** prediction time. Its input is the last `W` hours `[r−W+1, r]` of the same patient, left-padded.
#   A *carry-in* vector (the LOCF value and its age just before the window) keeps information about labs measured before the window, with no look-ahead.
# - Windows are built on the fly from the arrays of `03`, so nothing is duplicated W times on disk. The builder is written once to `ptcssl_data.py` and imported by every later notebook (single source of truth).
# - **Targets**
#   - `y_official = SepsisLabel` (already shifted: 1 for `t ≥ t_sepsis − 6`). Used for training and for the official utility.
#   - `hours_to_onset = t_sepsis − t`, and horizon targets `y_h = 1 if 0 ≤ hours_to_onset ≤ h` for h ∈ {4, 6, 8, 12}. Post-onset rows are excluded from the horizon analysis.
#     Self-check: `y_6` must equal `SepsisLabel` on every pre-onset row. This confirms the 6-hour label shift is handled correctly.
# - **Label-efficiency subsets** (5, 10, 25, 50, 100% of training patients; 5 seeds): stratified by hospital × septic, patient-level, **nested** (5% ⊂ 10% ⊂ …) for each seed.
#   By default the validation set is subsampled to the same fraction, so model selection does not use more labels than the budget allows.
# - **Leakage tests:** future rows are perturbed and every window must stay identical; windows never mix patients; subsets stay inside their split.
#
# CPU, about 1–3 min.

# %% [common]

# %%
NB = "04_temporal_windowing"
log = get_logger(NB)
CFG = {
    "window_hours": int(os.environ.get("PTCSSL_W", 24)),
    "horizons_h": [4, 6, 8, 12],
    "label_fractions": [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00],
    "subset_seeds": list(range(10)),   # seeds 0-4 identical to revision 1 (same per-stratum permutation)
    "val_follows_label_fraction": True,
    "min_septic_per_subset": 1,
}
PROTOCOLS = ["pooled", "A2B", "B2A"]
V = np.load(find_artifact("X_values.npy", "03_preprocessing"), mmap_mode="r")
M = np.load(find_artifact("X_mask.npy", "03_preprocessing"), mmap_mode="r")
F = np.load(find_artifact("X_locf.npy", "03_preprocessing"), mmap_mode="r")
AGE = np.load(find_artifact("X_age.npy", "03_preprocessing"), mmap_mode="r")
starts = np.load(find_artifact("patient_starts.npy", "03_preprocessing"))
lengths = np.load(find_artifact("patient_lengths.npy", "03_preprocessing"))
rows_meta = load_table(find_artifact("rows_meta.parquet", "03_preprocessing"))
patients = load_table(find_artifact("patients_split.parquet", "03_preprocessing"))
assert V.shape == M.shape == F.shape == AGE.shape and V.shape[1] == 34 and len(rows_meta) == len(V)
log.info("Arrays: %s rows, %d patients, W=%d", V.shape[0], len(patients), CFG["window_hours"])

# %% [markdown]
# ## 1. Shared data module `ptcssl_data.py` (window builder used by notebooks 05–11)

# %%
PTCSSL_DATA_SRC = r'''
"""ptcssl_data.py - causal window construction for PTC-SSL (written by notebook 04; do not edit by hand)."""
import numpy as np

class WindowBuilder:
    """Builds causal windows ending at global row r (inclusive), left-padded to W hours.

    Returns dict of numpy arrays for a batch of rows R:
      values [B,W,34] raw QC'd values (NaN where missing or padding)
      mask   [B,W,34] 1 = observed
      pad    [B,W]    1 = padding (before the patient's first row)
      carry_val [B,34] LOCF value just before the window (NaN if none)
      carry_age [B,34] hours since that observation, measured at the window's first hour (NaN if none)
      pidx   [B]      patient index
    Only rows <= r of the same patient are ever read (verified by notebook 04)."""

    def __init__(self, V, M, F, AGE, starts, row_pidx, W):
        self.V, self.M, self.F, self.AGE = V, M, F, AGE
        self.starts = np.asarray(starts)
        self.row_pidx = np.asarray(row_pidx)
        self.W = int(W)
        self.offsets = np.arange(-self.W + 1, 1)

    def get(self, R):
        R = np.asarray(R, dtype=np.int64)
        p = self.row_pidx[R]
        s0 = self.starts[p]
        idx = R[:, None] + self.offsets[None, :]                  # [B,W]
        valid = idx >= s0[:, None]
        idx_c = np.where(valid, idx, R[:, None])                   # any in-patient row; masked out below
        values = np.asarray(self.V[idx_c.ravel()]).reshape(len(R), self.W, -1).astype(np.float32)
        mask = np.asarray(self.M[idx_c.ravel()]).reshape(len(R), self.W, -1).astype(np.float32)
        values[~valid] = np.nan
        mask[~valid] = 0.0
        first = np.maximum(s0, R - self.W + 1)
        has_carry = first > s0
        prev = np.where(has_carry, first - 1, R)
        carry_val = np.asarray(self.F[prev]).astype(np.float32)
        carry_age = np.asarray(self.AGE[prev]).astype(np.float32) + 1.0
        carry_val[~has_carry] = np.nan
        carry_age[~has_carry] = np.nan
        return {"values": values, "mask": mask, "pad": (~valid).astype(np.float32),
                "carry_val": carry_val, "carry_age": carry_age, "pidx": p}

def normalize_np(X, stats, var_names):
    """Same transform as notebook 03: optional log1p -> winsorise -> z-score. NaN stays NaN."""
    out = np.empty(X.shape, dtype=np.float32)
    for j, v in enumerate(var_names):
        s = stats[v]
        x = X[..., j].astype(np.float64)
        if s["log1p"]:
            x = np.log1p(np.clip(x, 0, None))
        out[..., j] = (np.clip(x, s["lo"], s["hi"]) - s["mean"]) / s["std"]
    return out

def static_features(age, gender, iculos, stats):
    a = (np.asarray(age, float) - stats["_static"]["Age"]["mean"]) / stats["_static"]["Age"]["std"]
    l = (np.log1p(np.clip(np.asarray(iculos, float), 0, None)) - stats["_static"]["log_ICULOS"]["mean"]) \
        / stats["_static"]["log_ICULOS"]["std"]
    return np.stack([np.nan_to_num(a), np.nan_to_num(np.asarray(gender, float), nan=0.5), np.nan_to_num(l)], -1).astype(np.float32)
'''
(ART / "ptcssl_data.py").write_text(PTCSSL_DATA_SRC)
sys.path.insert(0, str(ART))
import importlib
import ptcssl_data
importlib.reload(ptcssl_data)
from ptcssl_data import WindowBuilder, normalize_np, static_features

row_pidx = rows_meta["pidx"].to_numpy(np.int64)
wb = WindowBuilder(V, M, F, AGE, starts, row_pidx, CFG["window_hours"])

# %% [markdown]
# ## 2. Leakage and correctness tests for the window builder

# %%
rng = np.random.default_rng(GLOBAL_SEED)
test_rows = rng.choice(len(V), size=min(3000, len(V)), replace=False)
base = wb.get(test_rows)

# (a) windows never mix patients and padding is exactly the pre-admission part
W = CFG["window_hours"]
for k, r in enumerate(test_rows[:500]):
    p = row_pidx[r]
    n_hist = r - starts[p] + 1
    assert base["pad"][k].sum() == max(0, W - n_hist)
    lo = max(starts[p], r - W + 1)
    assert np.array_equal(np.nan_to_num(base["values"][k, W - (r - lo + 1):], nan=-999),
                          np.nan_to_num(np.asarray(V[lo:r + 1]), nan=-999))

# (b) causality: perturb every row strictly after r (same patient) -> window must be identical
Vc, Mc, Fc, Ac = (np.array(a) for a in (V, M, F, AGE))
wb_c = WindowBuilder(Vc, Mc, Fc, Ac, starts, row_pidx, W)
for r in test_rows[:300]:
    p = row_pidx[r]
    seg = slice(r + 1, starts[p] + lengths[p])          # strictly future rows of the same patient
    backup = [a[seg].copy() for a in (Vc, Mc, Fc, Ac)]
    Vc[seg] += 1000.0; Mc[seg] = 1 - Mc[seg]; Fc[seg] += 1000.0; Ac[seg] += 5.0
    a, b = wb.get([r]), wb_c.get([r])
    for key in ("values", "mask", "carry_val", "carry_age"):
        assert np.array_equal(np.nan_to_num(a[key], nan=-1), np.nan_to_num(b[key], nan=-1)), (r, key)
    Vc[seg], Mc[seg], Fc[seg], Ac[seg] = backup
del Vc, Mc, Fc, Ac
# (c) carry-in equals the latest observation before the window
for k, r in enumerate(test_rows[:300]):
    p = row_pidx[r]
    first = max(starts[p], r - W + 1)
    if first > starts[p]:
        hist = np.asarray(V[starts[p]:first])
        for j in range(34):
            o = np.where(~np.isnan(hist[:, j]))[0]
            exp = hist[o[-1], j] if len(o) else np.nan
            got = base["carry_val"][k, j]
            assert (np.isnan(exp) and np.isnan(got)) or np.isclose(exp, got)
log.info("Window builder tests passed: no patient mixing, exact padding, strict causality, correct carry-in")

# %% [markdown]
# ## 3. Targets: official label, hours to onset, horizon labels (with the 6-hour shift check)

# %%
assert (patients.index == np.arange(len(patients))).all()   # pidx == positional index
pt = patients
t_sep = pt["t_sepsis_idx"].to_numpy()[row_pidx].astype(float)
t_sep[pt["septic"].to_numpy()[row_pidx] == 0] = np.nan
t_row = rows_meta["t"].to_numpy(float)
y_off = rows_meta["SepsisLabel"].to_numpy(np.int8)
hto = t_sep - t_row                           # hours to onset (NaN for non-septic)
targets = pd.DataFrame({"y_official": y_off, "hours_to_onset": hto.astype(np.float32)})
for h in CFG["horizons_h"]:
    y = np.zeros(len(t_row), dtype=np.float32)
    sep = ~np.isnan(hto)
    y[sep] = ((hto[sep] >= 0) & (hto[sep] <= h)).astype(np.float32)
    y[sep & (hto < 0)] = np.nan               # post-onset rows excluded from horizon analysis
    targets[f"y_h{h}"] = y

pre = ~np.isnan(hto) & (hto >= 0)
assert np.array_equal(targets.loc[pre, "y_h6"].to_numpy(), y_off[pre].astype(np.float32)), \
    "6-hour shift check failed: y_h6 must equal SepsisLabel before onset"
assert (y_off[np.isnan(hto)] == 0).all(), "non-septic patients must have SepsisLabel 0"
log.info("Label-shift check passed: SepsisLabel == 1[0 <= t_sepsis - t <= 6] on all pre-onset rows")
save_table(targets, ART / "targets.parquet")

# %% [markdown]
# ## 4. Sample indices per protocol and split

# %%
idx_summary = []
for proto in PROTOCOLS:
    split_of_row = patients[f"split_{proto}"].to_numpy()[row_pidx]
    for sp in ("train", "val", "test"):
        R = np.flatnonzero(split_of_row == sp).astype(np.int64)
        np.save(ART / f"idx_{proto}_{sp}.npy", R)
        idx_summary.append({"protocol": proto, "split": sp, "samples": len(R),
                            "patients": int(len(np.unique(row_pidx[R]))),
                            "positive_rate_official": float(y_off[R].mean()) if len(R) else np.nan})
idx_summary = pd.DataFrame(idx_summary)
idx_summary.to_csv(RES / "tables" / "sample_counts.csv", index=False)
display(idx_summary)

# %% [markdown]
# ## 5. Nested, stratified label-efficiency subsets (patient level)

# %%
def nested_subsets(pids, strata, fracs, seed, min_pos):
    """Return {frac: set(pids)} with nesting guaranteed (same permutation, growing prefix per stratum)."""
    rng = np.random.default_rng(seed)
    out = {f: [] for f in fracs}
    for s in np.unique(strata):
        members = pids[strata == s].copy()
        rng.shuffle(members)
        is_pos = s.endswith("_1")
        for f in fracs:
            k = int(np.ceil(f * len(members) - 1e-9))
            if is_pos:
                k = max(k, min(min_pos, len(members)))
            out[f].extend(members[:k].tolist())
    return {f: set(v) for f, v in out.items()}

sub_rows, sub_tables = [], {}
for proto in PROTOCOLS:
    tab = patients[["patient_id", "hospital", "septic", f"split_{proto}"]].copy()
    for sp in ("train", "val"):
        part = tab[tab[f"split_{proto}"] == sp]
        pids = part.index.to_numpy()
        strata = (part["hospital"] + "_" + part["septic"].astype(str)).to_numpy()
        for seed in CFG["subset_seeds"]:
            subs = nested_subsets(pids, strata, CFG["label_fractions"], 10_000 + seed, CFG["min_septic_per_subset"])
            fr = CFG["label_fractions"]
            for a, b in zip(fr[:-1], fr[1:]):
                assert subs[a] <= subs[b], "subsets must be nested"
            assert subs[1.0] == set(pids)
            for f in fr:
                col = f"{sp}_s{seed}_f{int(round(f * 100)):03d}"
                tab[col] = tab.index.isin(list(subs[f]))
                chosen = tab.loc[tab[col]]
                sub_rows.append({"protocol": proto, "split": sp, "seed": seed, "fraction": f,
                                 "patients": len(chosen), "septic_patients": int(chosen.septic.sum()),
                                 "labelled_hours": int(np.isin(row_pidx, chosen.index.to_numpy()).sum())})
    # no subset may contain patients from another split
    for c in [c for c in tab.columns if c.startswith(("train_s", "val_s"))]:
        sp = c.split("_")[0]
        assert (tab.loc[tab[c], f"split_{proto}"] == sp).all()
    sub_tables[proto] = tab
    save_table(tab.reset_index().rename(columns={"index": "pidx"}), ART / f"label_subsets_{proto}.parquet")
subs_df = pd.DataFrame(sub_rows)
subs_df.to_csv(RES / "tables" / "label_subset_sizes.csv", index=False)
display(subs_df[(subs_df.seed == 0)].pivot_table(index=["protocol", "split"], columns="fraction",
                                                  values="septic_patients"))

# %% [markdown]
# ## 6. Throughput benchmark and configuration

# %%
stats_pooled = load_json(find_artifact("norm_stats_pooled.json", "03_preprocessing"))
R = np.load(ART / "idx_pooled_train.npy")[:4096]
t0 = time.time()
batch = wb.get(R)
xn = normalize_np(batch["values"], stats_pooled, DYNAMIC_VARS)
el = time.time() - t0
assert np.isnan(xn).sum() == np.isnan(batch["values"]).sum(), "normalisation must not create/remove NaN"
log.info("Built + normalised %d windows in %.3fs (%.0f windows/s)", len(R), el, len(R) / max(el, 1e-9))

window_cfg = {**CFG, "n_dynamic": 34, "dynamic_vars": DYNAMIC_VARS, "static_features": ["Age", "Gender", "log_ICULOS"],
              "carry_in": True, "builder_module": "artifacts/ptcssl_data.py",
              "builder_sha256": sha256_file(ART / "ptcssl_data.py"),
              "targets": ["y_official"] + [f"y_h{h}" for h in CFG["horizons_h"]]}
save_json(window_cfg, ART / "window_config.json")
save_json(window_cfg, RES / "configs" / "window_config.json")
print("✅ Windowing complete. Next: 05_supervised_baselines (attach outputs of 01–04; GPU recommended).")
