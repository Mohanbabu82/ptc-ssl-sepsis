# %% [markdown]
# # 05 · Experiments: LightGBM, supervised GRU, generic SSL, PTC-SSL across label fractions
# **Inputs:** outputs of `01`–`04` (same session, or attached as inputs).
#
# **Protocol (fixed before seeing any test result)**
# - Data: `pooled` protocol (train/val/test = 70/15/15, patient-level). Pretraining uses **training patients only**, without labels.
# - Label fractions: 5, 10, 25 and 100% of training patients (nested subsets from `04`). The validation set is cut to the same fraction.
# - **Four arms, one shared encoder** (GRU, hidden 96, 24-h causal window with carry-in, missingness-aware inputs: LOCF value, mask, hours since last observation):
#   1. **LightGBM** on hand-crafted causal features (strong tabular baseline).
#   2. **Supervised GRU** trained from scratch.
#   3. **Generic SSL**: masked reconstruction with *random* masking (15% of observed entries), then fine-tuning.
#   4. **PTC-SSL**: organ-system block masking, plus the physiological consistency loss (constraints validated in `02`, offsets and scales from training patients in `03`), plus temporal consistency between adjacent windows, then fine-tuning.
# - λ_phys = λ_temp = 0.1 and all other hyper-parameters are fixed a priori, identical across arms. **There is no tuning on the test set.** Early stopping and the alarm threshold use the label-budgeted validation set only.
# - Metrics on the held-out test set: hourly AUROC and AUPRC, patient-level AUROC, Brier score, official normalised utility (threshold picked on validation), sensitivity, specificity, F1, false alarms per patient-day, median detection lead time. Patient-level bootstrap CIs and paired bootstrap differences are computed in the summary cells.
#
# Every finished run is appended to `results/tables/runs_metrics.csv` and skipped on re-run (resumable).

# %% [common]

# %%
NB = "05_experiments"
log = get_logger(NB)
import torch
import torch.nn as nn
import torch.nn.functional as TF
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, os.cpu_count() or 1))
CFG = {
    "protocol": "pooled", "fractions": [0.05, 0.10, 0.25, 1.00], "seeds": [0],
    "hidden": 96, "batch": 256, "lr": 1e-3, "pos_weight": 10.0,
    "pre_epochs": 4, "pre_windows_per_epoch": 120_000,
    "ft_max_epochs": 8, "ft_rows_per_epoch": 100_000, "patience": 2, "val_rows_cap": 60_000,
    "rand_mask_p": 0.15, "organ_block_h": 8, "lambda_phys": 0.1, "lambda_temp": 0.1,
    "max_delta_h": 168.0, "clip_z": 5.0,
}
if os.environ.get("PTCSSL_QUICK"):   # smoke test only
    CFG.update(pre_epochs=1, pre_windows_per_epoch=2000, ft_max_epochs=1, ft_rows_per_epoch=2000, val_rows_cap=2000)
log.info("Device=%s threads=%d CFG=%s", DEV, torch.get_num_threads(), CFG)
save_json(CFG, RES / "configs" / "experiment_config.json")

# %%
# ---- load artifacts ----
sys.path.insert(0, str(find_artifact("ptcssl_data.py", "04_temporal_windowing").parent))
from ptcssl_data import normalize_np
V = np.load(find_artifact("X_values.npy", "03_preprocessing"))
M = np.load(find_artifact("X_mask.npy", "03_preprocessing")).astype(bool)
F = np.load(find_artifact("X_locf.npy", "03_preprocessing"))
AGE = np.load(find_artifact("X_age.npy", "03_preprocessing")).astype(np.float32)
starts = np.load(find_artifact("patient_starts.npy", "03_preprocessing"))
rows_meta = load_table(find_artifact("rows_meta.parquet", "03_preprocessing"))
patients = load_table(find_artifact("patients_split.parquet", "03_preprocessing"))
targets = load_table(find_artifact("targets.parquet", "04_temporal_windowing"))
P = CFG["protocol"]
subsets = load_table(find_artifact(f"label_subsets_{P}.parquet", "04_temporal_windowing")).set_index("pidx")
NORM = load_json(find_artifact(f"norm_stats_{P}.json", "03_preprocessing"))
CONS = load_json(find_artifact("physiology_constraints.json", "03_preprocessing"))
ORGANS = load_json(find_artifact("organ_systems.json", "03_preprocessing"))
WCFG = load_json(find_artifact("window_config.json", "04_temporal_windowing"))
W = int(WCFG["window_hours"])
IDX = {sp: np.load(find_artifact(f"idx_{P}_{sp}.npy", "04_temporal_windowing")) for sp in ("train", "val", "test")}
row_pidx = rows_meta["pidx"].to_numpy(np.int64)
y_all = targets["y_official"].to_numpy(np.float32)
NV = 34

with StageTimer("normalise", log):
    VN = normalize_np(V, NORM, DYNAMIC_VARS)          # NaN where missing
    FN = normalize_np(F, NORM, DYNAMIC_VARS)
st_age = ((patients["age"].to_numpy(float) - NORM["_static"]["Age"]["mean"]) / NORM["_static"]["Age"]["std"]).astype(np.float32)
st_gen = np.nan_to_num(patients["gender"].to_numpy(float), nan=0.5).astype(np.float32)
row_icu = ((np.log1p(rows_meta["ICULOS"].to_numpy(float).clip(0)) - NORM["_static"]["log_ICULOS"]["mean"])
           / NORM["_static"]["log_ICULOS"]["std"]).astype(np.float32)
VIDX = {v: j for j, v in enumerate(DYNAMIC_VARS)}
ORGAN_MASKS = np.stack([np.isin(np.arange(NV), [VIDX[v] for v in vs]) for vs in ORGANS.values()])
for sp, R in IDX.items():
    assert patients.loc[np.unique(row_pidx[R]), f"split_{P}"].eq(sp).all(), "split leakage"
log.info("rows: %s", {k: len(v) for k, v in IDX.items()})

# %% [markdown]
# ## Batch builder (numpy, causal) with optional SSL masking
# Masked entries are removed **before** LOCF and time-since-observation are computed, so the corrupted input carries no trace of the hidden value.

# %%
OFF = np.arange(-W + 1, 1)
TSTEP = np.arange(W)
LOGMAX = np.log1p(CFG["max_delta_h"])

def build(R, mask_mode=None, rng=None):
    R = np.asarray(R, dtype=np.int64)
    B = len(R)
    p = row_pidx[R]
    s0 = starts[p]
    idx = R[:, None] + OFF
    valid = idx >= s0[:, None]
    idc = np.where(valid, idx, R[:, None])
    m = M[idc] & valid[..., None]
    x = np.where(m, VN[idc], np.nan).astype(np.float32)
    sel = np.zeros_like(m)
    if mask_mode == "random":
        sel = m & (rng.random(m.shape) < CFG["rand_mask_p"])
    elif mask_mode == "organ":
        sys_id = rng.integers(0, len(ORGAN_MASKS), B)
        L = CFG["organ_block_h"]
        t0 = rng.integers(0, W - L + 1, B)
        tblock = (TSTEP[None, :] >= t0[:, None]) & (TSTEP[None, :] < (t0 + L)[:, None])
        sel = m & tblock[..., None] & ORGAN_MASKS[sys_id][:, None, :]
        empty = ~sel.reshape(B, -1).any(1)                     # nothing observed in block -> fall back to random
        if empty.any():
            sel[empty] = m[empty] & (rng.random(m[empty].shape) < CFG["rand_mask_p"])
    m_in = m & ~sel
    first = np.maximum(s0, R - W + 1)
    has = first > s0
    prev = np.where(has, first - 1, R)
    cv = FN[prev].copy(); cv[~has] = np.nan
    ca = AGE[prev] + 1.0; ca[~has] = np.nan
    last = np.maximum.accumulate(np.where(m_in, TSTEP[None, :, None], -1), axis=1)
    has_in = last >= 0
    ff = np.take_along_axis(np.nan_to_num(np.where(m_in, x, 0.0)), np.maximum(last, 0), axis=1)
    ff = np.where(has_in, ff, np.nan_to_num(cv, nan=0.0)[:, None, :])
    delta = np.where(has_in, TSTEP[None, :, None] - last,
                     np.where(np.isnan(ca)[:, None, :], CFG["max_delta_h"], ca[:, None, :] + TSTEP[None, :, None]))
    delta = np.log1p(np.minimum(delta, CFG["max_delta_h"])) / LOGMAX
    X = np.concatenate([np.clip(ff, -CFG["clip_z"], CFG["clip_z"]), m_in.astype(np.float32),
                        delta.astype(np.float32), (~valid)[..., None].astype(np.float32)], axis=-1).astype(np.float32)
    st = np.stack([np.nan_to_num(st_age[p]), st_gen[p], row_icu[R]], -1).astype(np.float32)
    return X, st, x, sel, m_in

# causality spot-check of the builder: perturbing future rows must not change the batch
_rng = np.random.default_rng(0)
_up = np.unique(row_pidx[IDX["train"]])
_pids = _rng.choice(_up, min(50, len(_up)), replace=False)                     # one row per distinct patient
_R = np.array([_rng.choice(IDX["train"][row_pidx[IDX["train"]] == q]) for q in _pids])
_X1 = build(_R)[0]
_fut = np.zeros(len(VN), bool)
for r in _R:
    q = row_pidx[r]
    stop = starts[q + 1] if q + 1 < len(starts) else len(VN)
    _fut[r + 1:stop] = True
_bkV, _bkF = VN[_fut].copy(), FN[_fut].copy()
VN[_fut] += 100.0; FN[_fut] += 100.0
_X2 = build(_R)[0]
VN[_fut], FN[_fut] = _bkV, _bkF
del _bkV, _bkF
assert np.allclose(_X1, _X2), "batch builder reads future rows!"
log.info("Batch builder causality check passed; input dim=%d", _X1.shape[-1])
IN_DIM = _X1.shape[-1]

# %% [markdown]
# ## Model, physiology loss, temporal loss

# %%
class Net(nn.Module):
    def __init__(self, d_in, h):
        super().__init__()
        self.gru = nn.GRU(d_in, h, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(h + 3, h), nn.ReLU())
        self.recon = nn.Linear(h, NV)
        self.pred = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))     # temporal-consistency predictor
        self.clf = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))
    def encode(self, X, st):
        out, hN = self.gru(X)
        return out, self.fuse(torch.cat([hN[-1], st], -1))
    def forward(self, X, st):
        return self.clf(self.encode(X, st)[1]).squeeze(-1)

def T(a):
    return torch.as_tensor(a, device=DEV)

_mu = T(np.array([NORM[v]["mean"] for v in DYNAMIC_VARS], np.float32))
_sd = T(np.array([NORM[v]["std"] for v in DYNAMIC_VARS], np.float32))
_islog = T(np.array([NORM[v]["log1p"] for v in DYNAMIC_VARS]))
def to_raw(z):
    y = z * _sd + _mu
    return torch.where(_islog, torch.expm1(y.clamp(max=12)), y)

CONS_P = CONS["per_protocol"][P]
def physiology_loss(pred_z, x_true_z, sel, m_in):
    """pred_z/x_true_z: [B,W,34] normalised; sel = masked entries, m_in = visible entries."""
    raw_pred, raw_true = to_raw(pred_z), to_raw(torch.nan_to_num(x_true_z))
    C = torch.where(sel, raw_pred, raw_true)
    avail = sel | m_in
    g = lambda v: C[..., VIDX[v]]
    res = {
        "C1_MAP_formula": lambda: g("MAP") - (g("SBP") + 2 * g("DBP")) / 3,
        "C2_BP_ordering": lambda: torch.maximum(g("MAP") - g("SBP"), g("DBP") - g("MAP")),
        "C3_Henderson_Hasselbalch": lambda: g("pH") - (6.1 + torch.log10(g("HCO3").clamp(min=1e-2) / (0.03 * g("PaCO2").clamp(min=1e-2)))),
        "C4_Hct_3xHgb": lambda: g("Hct") - 3 * g("Hgb"),
        "C5_bilirubin_direct_le_total": lambda: g("Bilirubin_direct") - g("Bilirubin_total"),
        "C6_SaO2_vs_SpO2": lambda: g("SaO2") - g("O2Sat"),
    }
    losses = []
    for cid, c in CONS_P.items():
        J = [VIDX[v] for v in c["vars"]]
        ok = avail[..., J].all(-1) & sel[..., J].any(-1)
        if not ok.any():
            continue
        r = res[cid]()[ok]
        if c["kind"] == "eq":
            v = TF.relu((r - c["offset"]).abs() - c["tol"]) / c["scale"]
        else:
            v = TF.relu(r) / c["scale"]
        losses.append((v ** 2).clamp(max=100).mean())
    return torch.stack(losses).mean() if losses else pred_z.sum() * 0.0

# %% [markdown]
# ## Self-supervised pretraining (training patients only, no labels)

# %%
CKPT = WORK / "checkpoints"; CKPT.mkdir(exist_ok=True)

def pretrain(arm, seed):
    path = CKPT / f"pretrain_{arm}_{P}_s{seed}.pt"
    if path.exists():
        log.info("Using existing checkpoint %s", path.name)
        return torch.load(path, map_location=DEV)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(IN_DIM, CFG["hidden"]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=CFG["lr"])
    pool = IDX["train"]                       # all training patients, labels unused
    mode = "random" if arm == "generic_ssl" else "organ"
    hist = []
    t0 = time.time()
    for ep in range(CFG["pre_epochs"]):
        R_ep = rng.choice(pool, size=min(CFG["pre_windows_per_epoch"], len(pool)), replace=False)
        sums = {"rec": 0.0, "phys": 0.0, "temp": 0.0}; nb = 0
        net.train()
        for i in range(0, len(R_ep), CFG["batch"]):
            R = R_ep[i:i + CFG["batch"]]
            X, st, x, sel, m_in = build(R, mode, rng)
            if not sel.any():
                continue
            Xt, stt, xt, selt, mint = T(X), T(st), T(x), T(sel), T(m_in)
            out, z = net.encode(Xt, stt)
            pred = net.recon(out)
            rec = ((pred - torch.nan_to_num(xt)) ** 2)[selt].mean()
            loss = rec
            if arm == "ptc_ssl":
                phys = physiology_loss(pred, xt, selt, mint)
                Rp = np.where(R > starts[row_pidx[R]], R - 1, R)          # previous hour, same patient
                Xp, stp, *_ = build(Rp)
                with torch.no_grad():
                    zp = net.encode(T(Xp), T(stp))[1]
                temp = (1 - TF.cosine_similarity(net.pred(z), zp, dim=-1)).mean()
                loss = rec + CFG["lambda_phys"] * phys + CFG["lambda_temp"] * temp
                sums["phys"] += float(phys); sums["temp"] += float(temp)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sums["rec"] += float(rec); nb += 1
        rec_ep = {k: v / max(nb, 1) for k, v in sums.items()}
        hist.append({"arm": arm, "seed": seed, "epoch": ep, **rec_ep, "seconds": time.time() - t0})
        log.info("pretrain %s s%d ep%d %s", arm, seed, ep, {k: round(v, 4) for k, v in rec_ep.items()})
    state = net.state_dict()
    torch.save(state, path)
    pd.DataFrame(hist).to_csv(LOGS / f"pretrain_{arm}_s{seed}.csv", index=False)
    return state

# %% [markdown]
# ## Fine-tuning / supervised training and prediction

# %%
def predict(net, R, bs=2048):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(R), bs):
            X, st, *_ = build(R[i:i + bs])
            out.append(torch.sigmoid(net(T(X), T(st))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)

def rows_of(split, frac, seed):
    col = f"{split}_s{seed}_f{int(round(frac * 100)):03d}"
    keep = subsets.index[subsets[col]].to_numpy()
    return IDX[split][np.isin(row_pidx[IDX[split]], keep)]

def train_neural(arm, frac, seed, init_state=None):
    set_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    net = Net(IN_DIM, CFG["hidden"]).to(DEV)
    if init_state is not None:
        net.load_state_dict(init_state)
    opt = torch.optim.Adam(net.parameters(), lr=CFG["lr"])
    Rtr, Rva = rows_of("train", frac, seed), rows_of("val", frac, seed)
    if len(Rva) > CFG["val_rows_cap"]:
        Rva = np.sort(rng.choice(Rva, CFG["val_rows_cap"], replace=False))
    pw = torch.tensor(CFG["pos_weight"], device=DEV)
    best, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for ep in range(CFG["ft_max_epochs"]):
        net.train()
        R_ep = rng.permutation(Rtr)[:CFG["ft_rows_per_epoch"]]
        for i in range(0, len(R_ep), CFG["batch"]):
            R = R_ep[i:i + CFG["batch"]]
            X, st, *_ = build(R)
            loss = TF.binary_cross_entropy_with_logits(net(T(X), T(st)), T(y_all[R]), pos_weight=pw)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        pv = predict(net, Rva)
        ap = average_precision_score(y_all[Rva], pv) if y_all[Rva].any() else 0.0
        if ap > best:
            best, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if bad >= CFG["patience"]:
            break
    net.load_state_dict(best_state)
    return net, Rva, time.time() - t0, ep + 1

# %% [markdown]
# ## LightGBM baseline (causal hand-crafted features)

# %%
import lightgbm as lgb
VITAL_IDX = [VIDX[v] for v in ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]]
CS = np.cumsum(M, axis=0, dtype=np.int32)

def tab_features(R):
    R = np.asarray(R)
    p = row_pidx[R]; s0 = starts[p]
    lo = np.maximum(s0, R - 23)
    cnt24 = CS[R] - np.where((lo > 0)[:, None], CS[np.maximum(lo - 1, 0)], 0)
    idx6 = R[:, None] + np.arange(-5, 1)
    v6 = np.where((idx6 >= s0[:, None])[..., None], VN[np.maximum(idx6, 0)][:, :, VITAL_IDX], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        roll = np.concatenate([np.nanmean(v6, 1), np.nanmin(v6, 1), np.nanmax(v6, 1), np.nanstd(v6, 1)], 1)
    return np.concatenate([FN[R], AGE[R], cnt24.astype(np.float32), roll,
                           np.stack([st_age[p], st_gen[p], row_icu[R]], 1)], 1).astype(np.float32)

class LGBWrap:
    def __init__(self, m): self.m = m

def train_lgbm(frac, seed):
    Rtr, Rva = rows_of("train", frac, seed), rows_of("val", frac, seed)
    t0 = time.time()
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, min_child_samples=50, random_state=seed, verbose=-1, n_jobs=-1)
    m.fit(tab_features(Rtr), y_all[Rtr], eval_set=[(tab_features(Rva), y_all[Rva])], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, Rva, time.time() - t0

# %% [markdown]
# ## Evaluation: official utility, alarm metrics, lead time

# %%
import importlib.util
_spec = importlib.util.spec_from_file_location("official_eval", find_artifact("evaluate_sepsis_score_official.py", "01_dataset_acquisition"))
official = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(official)

def split_by_patient(R):
    p = row_pidx[R]
    cuts = np.flatnonzero(np.r_[True, p[1:] != p[:-1], True])
    return [(p[a], slice(a, b)) for a, b in zip(cuts[:-1], cuts[1:])]

_UCACHE = {}
def utility_norm(R, pred_bin):
    key = (len(R), int(R[0]), int(R[-1]))
    groups = split_by_patient(R)
    if key not in _UCACHE:              # best / inaction utilities depend on labels only
        best = inact = 0.0
        for pid, sl in groups:
            lab = y_all[R][sl]; n = len(lab)
            bp = np.zeros(n)
            if lab.any():
                ts = int(np.argmax(lab)) + 6           # official: t_sepsis = argmax(labels) - dt_optimal
                bp[max(0, ts - 12):min(ts + 3 + 1, n)] = 1
            best += official.compute_prediction_utility(lab, bp)
            inact += official.compute_prediction_utility(lab, np.zeros(n))
        _UCACHE[key] = (best, inact)
    best, inact = _UCACHE[key]
    yR = y_all[R]
    obs = sum(official.compute_prediction_utility(yR[sl], pred_bin[sl]) for _, sl in groups)
    return (obs - inact) / (best - inact)

def choose_threshold(Rva, pva):
    qs = np.unique(np.quantile(pva, np.linspace(0.80, 0.995, 40)))
    scores = [utility_norm(Rva, (pva >= q).astype(int)) for q in qs]
    return float(qs[int(np.argmax(scores))])

def evaluate(R, prob, thr):
    y = y_all[R]
    pb = (prob >= thr).astype(int)
    tp = int(((pb == 1) & (y == 1)).sum()); fp = int(((pb == 1) & (y == 0)).sum())
    fn = int(((pb == 0) & (y == 1)).sum()); tn = int(((pb == 0) & (y == 0)).sum())
    pdf = pd.DataFrame({"p": row_pidx[R], "prob": prob, "y": y, "alarm": pb, "t": rows_meta["t"].values[R]})
    pat = pdf.groupby("p").agg(maxp=("prob", "max"), sep=("y", "max"))
    non = pdf[~pdf.p.isin(pat.index[pat.sep == 1])]
    leads = []
    for pid, g in pdf[pdf.p.isin(pat.index[pat.sep == 1])].groupby("p"):
        ts = patients.loc[pid, "t_sepsis_idx"]
        al = g.loc[(g.alarm == 1) & (g.t <= ts + 3), "t"]
        if len(al):
            leads.append(ts - al.min())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    return {"AUROC": roc_auc_score(y, prob), "AUPRC": average_precision_score(y, prob),
            "patient_AUROC": roc_auc_score(pat.sep, pat.maxp), "Brier": brier_score_loss(y, prob),
            "utility": utility_norm(R, pb), "threshold": thr, "sensitivity": sens, "specificity": spec,
            "precision": prec, "F1": 2 * prec * sens / max(prec + sens, 1e-9),
            "false_alarms_per_patient_day": non.alarm.sum() / max(len(non) / 24, 1e-9),
            "septic_detected_pct": 100 * len(leads) / max((pat.sep == 1).sum(), 1),
            "median_lead_time_h": float(np.median(leads)) if leads else np.nan}

# %% [markdown]
# ## Run all arms × label fractions (resumable)

# %%
RUNS = RES / "tables" / "runs_metrics.csv"
PRED_DIR = ART / "predictions"; PRED_DIR.mkdir(exist_ok=True)
done = set(pd.read_csv(RUNS).apply(lambda r: (r.arm, r.fraction, r.seed), axis=1)) if RUNS.exists() else set()
Rte = IDX["test"]
pre_state = {}
for seed in CFG["seeds"]:
    for arm in ("generic_ssl", "ptc_ssl"):
        with StageTimer(f"pretrain {arm} s{seed}", log) as tm:
            pre_state[(arm, seed)] = pretrain(arm, seed)
        save_json({"seconds": tm.seconds}, LOGS / f"pretrain_time_{arm}_s{seed}.json")
    for frac in CFG["fractions"]:
        for arm in ("lightgbm", "supervised_gru", "generic_ssl", "ptc_ssl"):
            if (arm, frac, seed) in done:
                continue
            log.info("=== %s | frac=%.2f | seed=%d ===", arm, frac, seed)
            if arm == "lightgbm":
                model, Rva, secs = train_lgbm(frac, seed); epochs = model.best_iteration_
                pva = model.predict_proba(tab_features(Rva))[:, 1]
                t1 = time.time(); pte = model.predict_proba(tab_features(Rte))[:, 1]; inf = time.time() - t1
            else:
                model, Rva, secs, epochs = train_neural(arm, frac, seed, pre_state.get((arm, seed)))
                pva = predict(model, Rva)
                t1 = time.time(); pte = predict(model, Rte); inf = time.time() - t1
                torch.save(model.state_dict(), CKPT / f"ft_{arm}_{P}_f{int(frac*100):03d}_s{seed}.pt")
            thr = choose_threshold(Rva, pva)
            met = evaluate(Rte, pte, thr)
            rec = {"protocol": P, "arm": arm, "fraction": frac, "seed": seed, "train_seconds": secs,
                   "test_inference_seconds": inf, "epochs_or_trees": epochs,
                   "n_train_patients": int(subsets[f"train_s{seed}_f{int(round(frac*100)):03d}"].sum()), **met}
            pd.DataFrame([rec]).to_csv(RUNS, mode="a", header=not RUNS.exists(), index=False)
            save_table(pd.DataFrame({"row": Rte, "prob": pte.astype(np.float32)}),
                       PRED_DIR / f"pred_{arm}_{P}_f{int(frac*100):03d}_s{seed}.parquet")
            log.info("%s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})

# %% [markdown]
# ## Results table with patient-level bootstrap CIs and paired differences (PTC-SSL vs each comparator)

# %%
runs = pd.read_csv(RUNS)
display(runs[["arm", "fraction", "seed", "AUROC", "AUPRC", "patient_AUROC", "utility", "Brier",
              "sensitivity", "specificity", "false_alarms_per_patient_day", "median_lead_time_h", "train_seconds"]].round(4))

B = 200
rng = np.random.default_rng(GLOBAL_SEED)
groups = split_by_patient(Rte)
pid_slices = [sl for _, sl in groups]
yte = y_all[Rte]
def boot_idx():
    pick = rng.integers(0, len(pid_slices), len(pid_slices))
    return np.concatenate([np.arange(pid_slices[i].start, pid_slices[i].stop) for i in pick])
BOOTS = [boot_idx() for _ in range(B)]
def load_pred(arm, frac, seed):
    return load_table(PRED_DIR / f"pred_{arm}_{P}_f{int(frac*100):03d}_s{seed}.parquet")["prob"].to_numpy()

ci_rows, diff_rows = [], []
for seed in CFG["seeds"]:
    for frac in CFG["fractions"]:
        preds = {}
        for arm in ("lightgbm", "supervised_gru", "generic_ssl", "ptc_ssl"):
            try:
                preds[arm] = load_pred(arm, frac, seed)
            except FileNotFoundError:
                continue
        bs = {a: np.array([[roc_auc_score(yte[b], pr[b]), average_precision_score(yte[b], pr[b])] for b in BOOTS])
              for a, pr in preds.items()}
        for a, arr in bs.items():
            ci_rows.append({"fraction": frac, "seed": seed, "arm": a,
                            "AUROC_CI": f"{np.percentile(arr[:,0],2.5):.4f}–{np.percentile(arr[:,0],97.5):.4f}",
                            "AUPRC_CI": f"{np.percentile(arr[:,1],2.5):.4f}–{np.percentile(arr[:,1],97.5):.4f}"})
        if "ptc_ssl" in bs:
            for a in bs:
                if a == "ptc_ssl":
                    continue
                d = bs["ptc_ssl"] - bs[a]
                for k, name in ((0, "AUROC"), (1, "AUPRC")):
                    diff_rows.append({"fraction": frac, "seed": seed, "comparison": f"ptc_ssl - {a}", "metric": name,
                                      "mean_diff": d[:, k].mean(), "CI_low": np.percentile(d[:, k], 2.5),
                                      "CI_high": np.percentile(d[:, k], 97.5),
                                      "p_boot_two_sided": float(min(1, 2 * min((d[:, k] <= 0).mean(), (d[:, k] >= 0).mean())))})
ci = pd.DataFrame(ci_rows); diffs = pd.DataFrame(diff_rows)
ci.to_csv(RES / "tables" / "bootstrap_ci.csv", index=False)
diffs.to_csv(RES / "tables" / "paired_bootstrap_differences.csv", index=False)
display(ci); display(diffs.round(4))

# %%
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, metric in zip(axes, ["AUROC", "AUPRC", "utility"]):
    for arm, g in runs.groupby("arm"):
        g = g.groupby("fraction")[metric].mean()
        ax.plot(g.index * 100, g.values, marker="o", label=arm)
    ax.set_xscale("log"); ax.set_xlabel("% labelled training patients"); ax.set_title(metric)
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(RES / "figures" / f"fig_label_efficiency.{ext}", dpi=300, bbox_inches="tight")
print("✅ Experiments complete.")
