# %% [markdown]
# # 06 · Full experiments: 5 seeds, ablations, external validation, multi-seed statistics
# Runs after 01–04 in the same session (or with their outputs attached).
#
# **Jobs (fixed before seeing results; same encoder, inputs, budget and hyper-parameters as notebook 05)**
# 1. **Main**: pooled protocol, arms LightGBM / supervised GRU / generic SSL / PTC-SSL, label fractions 5, 10, 25, 100%, seeds 0–4.
# 2. **Ablations** (pooled, 5% and 10% labels, seeds 0–4): PTC-SSL without organ masking (random masking), without L_phys, without L_temp, values-only input (mask and time-since-observation channels zeroed), missingness-only input (value channels zeroed).
# 3. **External validation**: train/val on one hospital, test on the other (A→B, B→A), 10% and 100% labels, seeds 0–2.
# 4. **Summary**: mean ± SD over seeds; paired patient-level bootstrap on seed-ensembled predictions; per-seed win counts; calibration (ECE, reliability bins); figures; result package.
#
# A smoke test runs every arm on a few rows first, so a bug fails in minutes, not hours. A time guard skips the remaining runs (never the summary) if the budget is exceeded. Every run is appended to `results/tables/runs_metrics.csv` and skipped on resume.

# %% [common]

# %%
NB = "06_full_experiments"
log = get_logger(NB)
import torch
import torch.nn as nn
import torch.nn.functional as TF
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

T_START = time.time()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, os.cpu_count() or 1))
CFG = {
    "hidden": 96, "batch": 256, "lr": 1e-3, "pos_weight": 10.0,
    "pre_epochs": 4, "pre_windows_per_epoch": 120_000,
    "ft_max_epochs": 8, "ft_rows_per_epoch": 100_000, "patience": 2, "val_rows_cap": 60_000,
    "rand_mask_p": 0.15, "organ_block_h": 8, "lambda": 0.1,
    "max_delta_h": 168.0, "clip_z": 5.0, "bootstrap_B": 200,
    "time_budget_h": float(os.environ.get("PTCSSL_BUDGET_H", 8.0)),
}
MAIN_ARMS = ["lightgbm", "supervised_gru", "generic_ssl", "ptc_ssl"]
ABL_ARMS = ["abl_no_organ", "abl_no_phys", "abl_no_temp", "abl_values_only", "abl_missing_only"]
JOBS = [
    {"name": "main", "protocol": "pooled", "seeds": [0, 1, 2, 3, 4], "fractions": [0.05, 0.10, 0.25, 1.00], "arms": MAIN_ARMS},
    {"name": "ablation", "protocol": "pooled", "seeds": [0, 1, 2, 3, 4], "fractions": [0.05, 0.10], "arms": ABL_ARMS},
    {"name": "external", "protocol": "A2B", "seeds": [0, 1, 2], "fractions": [0.10, 1.00], "arms": MAIN_ARMS},
    {"name": "external", "protocol": "B2A", "seeds": [0, 1, 2], "fractions": [0.10, 1.00], "arms": MAIN_ARMS},
]
# SSL variants: (masking, lambda_phys, lambda_temp, input mode)
L = CFG["lambda"]
VARIANTS = {"generic_ssl": ("random", 0.0, 0.0, "full"), "ptc_ssl": ("organ", L, L, "full"),
            "abl_no_organ": ("random", L, L, "full"), "abl_no_phys": ("organ", 0.0, L, "full"),
            "abl_no_temp": ("organ", L, 0.0, "full"), "abl_values_only": ("organ", L, L, "values"),
            "abl_missing_only": ("organ", L, L, "missing")}
if os.environ.get("PTCSSL_QUICK"):
    CFG.update(pre_epochs=1, pre_windows_per_epoch=2000, ft_max_epochs=1, ft_rows_per_epoch=2000, val_rows_cap=2000, bootstrap_B=20)
    for j in JOBS:
        j["seeds"] = j["seeds"][:2]
log.info("Device=%s threads=%d CFG=%s", DEV, torch.get_num_threads(), CFG)
if DEV.type == "cuda":
    log.info("GPU: %s", torch.cuda.get_device_name(0))
save_json({"CFG": CFG, "JOBS": JOBS, "VARIANTS": VARIANTS}, RES / "configs" / "full_experiment_config.json")

# %%
# ---- protocol-independent artifacts ----
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
CONS = load_json(find_artifact("physiology_constraints.json", "03_preprocessing"))
ORGANS = load_json(find_artifact("organ_systems.json", "03_preprocessing"))
W = int(load_json(find_artifact("window_config.json", "04_temporal_windowing"))["window_hours"])
row_pidx = rows_meta["pidx"].to_numpy(np.int64)
row_t = rows_meta["t"].to_numpy()
y_all = targets["y_official"].to_numpy(np.float32)
NV = 34
VIDX = {v: j for j, v in enumerate(DYNAMIC_VARS)}
ORGAN_MASKS = np.stack([np.isin(np.arange(NV), [VIDX[v] for v in vs]) for vs in ORGANS.values()])
st_gen = np.nan_to_num(patients["gender"].to_numpy(float), nan=0.5).astype(np.float32)
CS = np.cumsum(M, axis=0, dtype=np.int32)
INPUT_MODE = "full"

def T(a):
    return torch.as_tensor(a, device=DEV)

def use_protocol(proto):
    """Rebind every protocol-dependent global (normalisation, splits, subsets, constraint stats)."""
    global P, NORM, subsets, IDX, VN, FN, st_age, row_icu, CONS_P, _mu, _sd, _islog
    P = proto
    NORM = load_json(find_artifact(f"norm_stats_{P}.json", "03_preprocessing"))
    subsets = load_table(find_artifact(f"label_subsets_{P}.parquet", "04_temporal_windowing")).set_index("pidx")
    IDX = {sp: np.load(find_artifact(f"idx_{P}_{sp}.npy", "04_temporal_windowing")) for sp in ("train", "val", "test")}
    VN = normalize_np(V, NORM, DYNAMIC_VARS)
    FN = normalize_np(F, NORM, DYNAMIC_VARS)
    st_age = ((patients["age"].to_numpy(float) - NORM["_static"]["Age"]["mean"]) / NORM["_static"]["Age"]["std"]).astype(np.float32)
    row_icu = ((np.log1p(rows_meta["ICULOS"].to_numpy(float).clip(0)) - NORM["_static"]["log_ICULOS"]["mean"])
               / NORM["_static"]["log_ICULOS"]["std"]).astype(np.float32)
    CONS_P = CONS["per_protocol"][P]
    _mu = T(np.array([NORM[v]["mean"] for v in DYNAMIC_VARS], np.float32))
    _sd = T(np.array([NORM[v]["std"] for v in DYNAMIC_VARS], np.float32))
    _islog = T(np.array([NORM[v]["log1p"] for v in DYNAMIC_VARS]))
    for sp, R in IDX.items():
        assert patients.loc[np.unique(row_pidx[R]), f"split_{P}"].eq(sp).all(), "split leakage"
    log.info("protocol %s: rows %s", P, {k: len(v) for k, v in IDX.items()})

use_protocol("pooled")

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
        Lb = CFG["organ_block_h"]
        t0 = rng.integers(0, W - Lb + 1, B)
        tblock = (TSTEP[None, :] >= t0[:, None]) & (TSTEP[None, :] < (t0 + Lb)[:, None])
        sel = m & tblock[..., None] & ORGAN_MASKS[sys_id][:, None, :]
        empty = ~sel.reshape(B, -1).any(1)
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
    if INPUT_MODE == "missing":
        X[..., :NV] = 0.0                       # values hidden: only what was measured and when
    elif INPUT_MODE == "values":
        X[..., NV:3 * NV] = 0.0                 # measurement pattern hidden: only (carried-forward) values
    st = np.stack([np.nan_to_num(st_age[p]), st_gen[p], row_icu[R]], -1).astype(np.float32)
    return X, st, x, sel, m_in

# causality spot-check (as in 05)
_rng = np.random.default_rng(0)
_up = np.unique(row_pidx[IDX["train"]])
_pids = _rng.choice(_up, min(50, len(_up)), replace=False)
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
IN_DIM = _X1.shape[-1]
log.info("Batch builder causality check passed; input dim=%d", IN_DIM)

# %%
class Net(nn.Module):
    def __init__(self, d_in, h):
        super().__init__()
        self.gru = nn.GRU(d_in, h, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(h + 3, h), nn.ReLU())
        self.recon = nn.Linear(h, NV)
        self.pred = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
        self.clf = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))
    def encode(self, X, st):
        out, hN = self.gru(X)
        return out, self.fuse(torch.cat([hN[-1], st], -1))
    def forward(self, X, st):
        return self.clf(self.encode(X, st)[1]).squeeze(-1)

def to_raw(z):
    y = z * _sd + _mu
    return torch.where(_islog, torch.expm1(y.clamp(max=12)), y)

def physiology_loss(pred_z, x_true_z, sel, m_in):
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

CKPT = WORK / "checkpoints"; CKPT.mkdir(exist_ok=True)

def pretrain(arm, seed, tag=""):
    global INPUT_MODE
    mode, lp, lt, inp = VARIANTS[arm]
    INPUT_MODE = inp
    path = CKPT / f"pretrain_{arm}_{P}_s{seed}{tag}.pt"
    if path.exists():
        return torch.load(path, map_location=DEV)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(IN_DIM, CFG["hidden"]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=CFG["lr"])
    pool = IDX["train"]
    hist, t0 = [], time.time()
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
            if lp > 0:
                phys = physiology_loss(pred, xt, selt, mint)
                loss = loss + lp * phys
                sums["phys"] += float(phys)
            if lt > 0:
                Rp = np.where(R > starts[row_pidx[R]], R - 1, R)
                Xp, stp, *_ = build(Rp)
                with torch.no_grad():
                    zp = net.encode(T(Xp), T(stp))[1]
                temp = (1 - TF.cosine_similarity(net.pred(z), zp, dim=-1)).mean()
                loss = loss + lt * temp
                sums["temp"] += float(temp)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sums["rec"] += float(rec); nb += 1
        hist.append({"arm": arm, "protocol": P, "seed": seed, "epoch": ep, **{k: v / max(nb, 1) for k, v in sums.items()},
                     "seconds": time.time() - t0})
        log.info("pretrain %s %s s%d ep%d %s", arm, P, seed, ep, {k: round(v / max(nb, 1), 4) for k, v in sums.items()})
    state = net.state_dict()
    torch.save(state, path)
    pd.DataFrame(hist).to_csv(LOGS / f"pretrain_{arm}_{P}_s{seed}{tag}.csv", index=False)
    return state

def predict(net, R, bs=2048):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(R), bs):
            X, st, *_ = build(R[i:i + bs])
            out.append(torch.sigmoid(net(T(X), T(st))).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)

def rows_of(split, frac, seed):
    col = f"{split}_s{seed}_f{int(round(frac * 100)):03d}"
    keep = subsets.index[subsets[col]].to_numpy()
    return IDX[split][np.isin(row_pidx[IDX[split]], keep)]

def train_neural(arm, frac, seed, init_state=None):
    global INPUT_MODE
    INPUT_MODE = VARIANTS[arm][3] if arm in VARIANTS else "full"
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
    best, best_state, bad, ep = -1.0, None, 0, 0
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

# %%
import lightgbm as lgb
VITAL_IDX = [VIDX[v] for v in ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]]

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

def train_lgbm(frac, seed):
    Rtr, Rva = rows_of("train", frac, seed), rows_of("val", frac, seed)
    t0 = time.time()
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, min_child_samples=50, random_state=seed, verbose=-1, n_jobs=-1)
    m.fit(tab_features(Rtr), y_all[Rtr], eval_set=[(tab_features(Rva), y_all[Rva])], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, Rva, time.time() - t0

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
    if key not in _UCACHE:
        best = inact = 0.0
        for pid, sl in groups:
            lab = y_all[R][sl]; n = len(lab)
            bp = np.zeros(n)
            if lab.any():
                ts = int(np.argmax(lab)) + 6
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

def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    tot = 0.0
    for k in range(bins):
        s = b == k
        if s.any():
            tot += s.mean() * abs(y[s].mean() - p[s].mean())
    return float(tot)

def evaluate(R, prob, thr):
    y = y_all[R]
    pb = (prob >= thr).astype(int)
    tp = int(((pb == 1) & (y == 1)).sum()); fp = int(((pb == 1) & (y == 0)).sum())
    fn = int(((pb == 0) & (y == 1)).sum()); tn = int(((pb == 0) & (y == 0)).sum())
    pdf = pd.DataFrame({"p": row_pidx[R], "prob": prob, "y": y, "alarm": pb, "t": row_t[R]})
    pat = pdf.groupby("p").agg(maxp=("prob", "max"), sep=("y", "max"))
    non = pdf[~pdf.p.isin(pat.index[pat.sep == 1])]
    leads = []
    tsep = patients["t_sepsis_idx"].to_numpy()
    for pid, g in pdf[pdf.p.isin(pat.index[pat.sep == 1])].groupby("p"):
        ts = tsep[pid]
        al = g.loc[(g.alarm == 1) & (g.t <= ts + 3), "t"]
        if len(al):
            leads.append(ts - al.min())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    return {"AUROC": roc_auc_score(y, prob), "AUPRC": average_precision_score(y, prob),
            "patient_AUROC": roc_auc_score(pat.sep, pat.maxp), "Brier": brier_score_loss(y, prob), "ECE": ece(y, prob),
            "utility": utility_norm(R, pb), "threshold": thr, "sensitivity": sens, "specificity": spec,
            "precision": prec, "F1": 2 * prec * sens / max(prec + sens, 1e-9),
            "false_alarms_per_patient_day": non.alarm.sum() / max(len(non) / 24, 1e-9),
            "septic_detected_pct": 100 * len(leads) / max((pat.sep == 1).sum(), 1),
            "median_lead_time_h": float(np.median(leads)) if leads else np.nan}

# %% [markdown]
# ## Smoke test: every arm end-to-end on a few rows (fails fast on any bug)

# %%
def run_one(arm, frac, seed, pre_state):
    if arm == "lightgbm":
        model, Rva, secs = train_lgbm(frac, seed)
        epochs = model.best_iteration_
        pva = model.predict_proba(tab_features(Rva))[:, 1]
        t1 = time.time(); pte = model.predict_proba(tab_features(IDX["test"]))[:, 1]; inf = time.time() - t1
    else:
        model, Rva, secs, epochs = train_neural(arm, frac, seed, pre_state)
        pva = predict(model, Rva)
        t1 = time.time(); pte = predict(model, IDX["test"]); inf = time.time() - t1
    thr = choose_threshold(Rva, pva)
    return pte, evaluate(IDX["test"], pte, thr), secs, inf, epochs

with StageTimer("smoke test", log):
    _saved = dict(CFG)
    CFG.update(pre_epochs=1, pre_windows_per_epoch=1024, ft_max_epochs=1, ft_rows_per_epoch=1024, val_rows_cap=1500)
    _IDXsave = IDX
    _rs = np.random.default_rng(0)
    IDX = {k: np.sort(_rs.choice(v, min(len(v), 3000), replace=False)) for k, v in _IDXsave.items()}
    for arm in MAIN_ARMS + ABL_ARMS:
        st_ = pretrain(arm, 99, tag="_smoke") if arm in VARIANTS else None
        _, met, *_ = run_one(arm, 0.10, 0, st_)
        log.info("smoke %s ok: AUROC=%.3f", arm, met["AUROC"])
    IDX = _IDXsave
    CFG.clear(); CFG.update(_saved)
    _UCACHE.clear()
    for f in CKPT.glob("*_smoke.pt"):
        f.unlink()
INPUT_MODE = "full"
log.info("SMOKE TEST PASSED")

# %% [markdown]
# ## Run all jobs (resumable, time-guarded)

# %%
RUNS = RES / "tables" / "runs_metrics.csv"
PRED_DIR = ART / "predictions"; PRED_DIR.mkdir(exist_ok=True)
def done_keys():
    if not RUNS.exists():
        return set()
    d = pd.read_csv(RUNS)
    return set(zip(d.protocol, d.arm, d.fraction.round(2), d.seed))
def pred_path(proto, arm, frac, seed):
    return PRED_DIR / f"pred_{arm}_{proto}_f{int(round(frac * 100)):03d}_s{seed}.npy"

skipped = []
for job in JOBS:
    if job["protocol"] != P:
        use_protocol(job["protocol"]); _UCACHE.clear()
    for seed in job["seeds"]:
        pre_state = {}
        for frac in job["fractions"]:
            for arm in job["arms"]:
                if (P, arm, round(frac, 2), seed) in done_keys():
                    continue
                if (time.time() - T_START) / 3600 > CFG["time_budget_h"]:
                    skipped.append((job["name"], P, arm, frac, seed)); continue
                if arm in VARIANTS and arm not in pre_state:
                    with StageTimer(f"pretrain {arm} {P} s{seed}", log):
                        pre_state[arm] = pretrain(arm, seed)
                log.info("=== %s | %s | %s | frac=%.2f | seed=%d | elapsed %.2f h ===", job["name"], P, arm, frac, seed,
                         (time.time() - T_START) / 3600)
                pte, met, secs, inf, epochs = run_one(arm, frac, seed, pre_state.get(arm))
                rec = {"job": job["name"], "protocol": P, "arm": arm, "fraction": frac, "seed": seed, "train_seconds": secs,
                       "test_inference_seconds": inf, "epochs_or_trees": epochs,
                       "n_train_patients": int(subsets[f"train_s{seed}_f{int(round(frac*100)):03d}"].sum()), **met}
                pd.DataFrame([rec]).to_csv(RUNS, mode="a", header=not RUNS.exists(), index=False)
                np.save(pred_path(P, arm, frac, seed), pte.astype(np.float32))
                print("RESULT", json.dumps({k: (round(v, 4) if isinstance(v, (float, np.floating)) else v) for k, v in rec.items()}, default=str), flush=True)
save_json({"skipped_runs": skipped}, RES / "skipped_runs.json")
log.info("All jobs finished; skipped=%d; elapsed %.2f h", len(skipped), (time.time() - T_START) / 3600)

# %% [markdown]
# ## Multi-seed statistics

# %%
runs = pd.read_csv(RUNS)
KEYM = ["AUROC", "AUPRC", "patient_AUROC", "utility", "Brier", "ECE", "sensitivity", "specificity",
        "false_alarms_per_patient_day", "septic_detected_pct", "median_lead_time_h"]
agg = runs.groupby(["job", "protocol", "arm", "fraction"])[KEYM].agg(["mean", "std", "count"])
agg.columns = [f"{a}_{b}" for a, b in agg.columns]
agg = agg.reset_index()
agg.to_csv(RES / "tables" / "multiseed_summary.csv", index=False)

B = CFG["bootstrap_B"]
rng = np.random.default_rng(GLOBAL_SEED)
diff_rows, wins_rows, calib_rows = [], [], []
for proto in ["pooled", "A2B", "B2A"]:
    sub = runs[runs.protocol == proto]
    if sub.empty:
        continue
    use_protocol(proto)
    Rte = IDX["test"]; yte = y_all[Rte]
    sl = [s for _, s in split_by_patient(Rte)]
    BOOTS = []
    for _ in range(B):
        pick = rng.integers(0, len(sl), len(sl))
        BOOTS.append(np.concatenate([np.arange(sl[i].start, sl[i].stop) for i in pick]))
    for frac in sorted(sub.fraction.unique()):
        arms_here = sorted(sub[sub.fraction == frac].arm.unique())
        seeds_by_arm = {a: sorted(sub[(sub.fraction == frac) & (sub.arm == a)].seed) for a in arms_here}
        ens = {}
        for a in arms_here:
            ps = [np.load(pred_path(proto, a, frac, s)) for s in seeds_by_arm[a] if pred_path(proto, a, frac, s).exists()]
            if ps:
                ens[a] = np.mean(ps, axis=0)
        if "ptc_ssl" not in ens:
            continue
        bs = {a: np.array([[roc_auc_score(yte[b], pr[b]), average_precision_score(yte[b], pr[b])] for b in BOOTS]) for a, pr in ens.items()}
        for a in bs:
            if a == "ptc_ssl":
                continue
            d = bs["ptc_ssl"] - bs[a]
            for k, name in ((0, "AUROC"), (1, "AUPRC")):
                diff_rows.append({"protocol": proto, "fraction": frac, "comparison": f"ptc_ssl - {a}", "metric": name,
                                  "n_seeds_ptc": len(seeds_by_arm["ptc_ssl"]), "n_seeds_other": len(seeds_by_arm[a]),
                                  "mean_diff": d[:, k].mean(), "CI_low": np.percentile(d[:, k], 2.5), "CI_high": np.percentile(d[:, k], 97.5),
                                  "p_boot_two_sided": float(min(1, 2 * min((d[:, k] <= 0).mean(), (d[:, k] >= 0).mean())))})
            common = sorted(set(seeds_by_arm["ptc_ssl"]) & set(seeds_by_arm[a]))
            for name in ("AUROC", "AUPRC", "utility"):
                pv = sub[(sub.fraction == frac) & (sub.arm == "ptc_ssl") & sub.seed.isin(common)].set_index("seed")[name]
                ov = sub[(sub.fraction == frac) & (sub.arm == a) & sub.seed.isin(common)].set_index("seed")[name]
                dd = (pv - ov).dropna()
                wins_rows.append({"protocol": proto, "fraction": frac, "comparison": f"ptc_ssl - {a}", "metric": name,
                                  "seeds": len(dd), "ptc_wins": int((dd > 0).sum()), "mean_diff": dd.mean(), "sd_diff": dd.std()})
        for a, pr in ens.items():
            q = np.quantile(pr, np.linspace(0, 1, 11))
            b = np.clip(np.searchsorted(q, pr, side="right") - 1, 0, 9)
            for k in range(10):
                s = b == k
                if s.any():
                    calib_rows.append({"protocol": proto, "fraction": frac, "arm": a, "bin": k,
                                       "mean_pred": float(pr[s].mean()), "obs_rate": float(yte[s].mean()), "n": int(s.sum())})
        log.info("bootstrap done %s frac=%.2f", proto, frac)
diffs = pd.DataFrame(diff_rows); wins = pd.DataFrame(wins_rows); calib = pd.DataFrame(calib_rows)
diffs.to_csv(RES / "tables" / "paired_bootstrap_seed_ensemble.csv", index=False)
wins.to_csv(RES / "tables" / "per_seed_wins.csv", index=False)
calib.to_csv(RES / "tables" / "calibration_bins.csv", index=False)

# %%
import matplotlib.pyplot as plt
main = agg[(agg.job == "main") & (agg.protocol == "pooled")]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, met in zip(axes, ["AUROC", "AUPRC", "utility"]):
    for arm, g in main.groupby("arm"):
        g = g.sort_values("fraction")
        ax.errorbar(g.fraction * 100, g[f"{met}_mean"], yerr=g[f"{met}_std"], marker="o", capsize=3, label=arm)
    ax.set_xscale("log"); ax.set_xlabel("% labelled training patients"); ax.set_title(f"{met} (mean ± SD over seeds)")
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(RES / "figures" / f"fig_label_efficiency_multiseed.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Compact printout (read by the paper draft) and result package

# %%
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 500)
print("\n==== MULTISEED SUMMARY (mean | sd | n) ====")
for _, r in agg.iterrows():
    print("MS", r.job, r.protocol, r.arm, r.fraction, " ".join(f"{m}={r[m + '_mean']:.4f}|{r[m + '_std']:.4f}|{int(r[m + '_count'])}" for m in KEYM))
print("\n==== PAIRED BOOTSTRAP ON SEED ENSEMBLES ====")
for _, r in diffs.iterrows():
    print("PB", r.protocol, r.fraction, r.comparison, r.metric, f"{r.mean_diff:+.4f} [{r.CI_low:+.4f},{r.CI_high:+.4f}] p={r.p_boot_two_sided:.3f} seeds={r.n_seeds_ptc}/{r.n_seeds_other}")
print("\n==== PER-SEED WINS ====")
for _, r in wins.iterrows():
    print("WN", r.protocol, r.fraction, r.comparison, r.metric, f"wins={r.ptc_wins}/{r.seeds} mean={r.mean_diff:+.4f} sd={r.sd_diff:.4f}")
print("\nSKIPPED:", skipped)

import zipfile
with zipfile.ZipFile(WORK / "PTC_SSL_PAPER_RESULTS.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in RES.rglob("*"):
        if f.is_file():
            z.write(f, f.relative_to(WORK))
    for f in LOGS.glob("*"):
        z.write(f, f.relative_to(WORK))
with zipfile.ZipFile(WORK / "PTC_SSL_IEEE_ACCESS_RESEARCH_PACKAGE.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in list(RES.rglob("*")) + list(LOGS.glob("*")) + list(CKPT.glob("*.pt")) + [ART / "ptcssl_data.py"]:
        if f.is_file():
            z.write(f, f.relative_to(WORK))
# keep the output small: drop large intermediate arrays (raw patient-level data is never packaged)
for f in ART.glob("X_*.npy"):
    f.unlink()
print("✅ Full experiments complete. Elapsed %.2f h" % ((time.time() - T_START) / 3600))
