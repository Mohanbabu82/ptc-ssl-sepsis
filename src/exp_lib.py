# ===== PTC-SSL experiment library v2 (shared by notebooks 07a and 07b; do not edit per notebook) =====
# Same encoder, inputs, budget and hyper-parameters as notebook 06 unless a variant overrides them.
import torch
import torch.nn as nn
import torch.nn.functional as TF
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import importlib.util, zipfile

T_START = time.time()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, os.cpu_count() or 1))
CFG = {
    "hidden": 96, "batch": 256, "lr": 1e-3, "pos_weight": 10.0,
    "pre_epochs": 4, "pre_windows_per_epoch": 120_000,
    "ft_max_epochs": 8, "ft_rows_per_epoch": 100_000, "patience": 2, "val_rows_cap": 60_000,
    "max_delta_h": 168.0, "clip_z": 5.0,
    "bootstrap_B": int(os.environ.get("PTCSSL_BOOT_B", 2000)),
    "time_budget_h": float(os.environ.get("PTCSSL_BUDGET_H", 11.0)),
}
L = 0.1
BASE = dict(ssl="ptc", mask="organ", lp=L, lt=L, input="full", block=8, hidden=96, enc="gru", mask_p=0.15, n_var=4)
def spec(**kw):
    d = dict(BASE); d.update(kw); return d

# Every neural arm. ssl: none = supervised from scratch; ptc = masked reconstruction (+ optional L_phys, L_temp);
# ts2vec / simmtm = re-implementations of those objectives on the SAME GRU encoder and inputs.
VARIANTS = {
    "supervised_gru": spec(ssl="none"),
    "generic_ssl": spec(mask="random", lp=0.0, lt=0.0),
    "ptc_ssl": spec(),
    # full 2^3 factorial over O = organ-system masking, P = L_phys, T = L_temp
    "fac_O": spec(lp=0.0, lt=0.0), "fac_P": spec(mask="random", lt=0.0), "fac_T": spec(mask="random", lp=0.0),
    "fac_OP": spec(lt=0.0), "fac_OT": spec(lp=0.0), "fac_PT": spec(mask="random"),
    # input ablations (measurement pattern vs values)
    "abl_values_only": spec(input="values"), "abl_missing_only": spec(input="missing"),
    "supervised_missing_only": spec(ssl="none", input="missing"),
    "supervised_values_only": spec(ssl="none", input="values"),
    "ptc_no_fio2": spec(input="no_fio2"),
    # other SSL objectives and encoder
    "ts2vec_gru": spec(ssl="ts2vec", mask="random", lp=0.0, lt=0.0),
    "simmtm_gru": spec(ssl="simmtm", mask="random", mask_p=0.30, lp=0.0, lt=0.0),
    "supervised_tf": spec(ssl="none", enc="tf"),
    "ptc_tf": spec(enc="tf"),
    # unsupervised domain adaptation: SSL pretraining also sees unlabelled windows of the target hospital's adaptation half
    "ptc_uda": spec(uda=True), "generic_uda": spec(mask="random", lp=0.0, lt=0.0, uda=True),
}
for v in (0.0, 0.01, 0.05, 0.5, 1.0):
    VARIANTS[f"sens_lp_{v:g}"] = spec(lp=v)
    VARIANTS[f"sens_lt_{v:g}"] = spec(lt=v)
for h in (64, 128, 256):
    VARIANTS[f"sens_hidden_{h}"] = spec(hidden=h)
for b in (4, 12, 16):
    VARIANTS[f"sens_block_{b}"] = spec(block=b)
VARIANTS["sens_mask_random"] = spec(mask="random")
VARIANTS["sens_mask_variable"] = spec(mask="variable")
FACTORIAL = {"generic_ssl": (0, 0, 0), "fac_O": (1, 0, 0), "fac_P": (0, 1, 0), "fac_T": (0, 0, 1),
             "fac_OP": (1, 1, 0), "fac_OT": (1, 0, 1), "fac_PT": (0, 1, 1), "ptc_ssl": (1, 1, 1)}
LGBM_ARMS = {"lightgbm": "full", "lgbm_missing_only": "missing", "lgbm_values_only": "values"}

if os.environ.get("PTCSSL_QUICK"):
    CFG.update(pre_epochs=1, pre_windows_per_epoch=2000, ft_max_epochs=1, ft_rows_per_epoch=2000, val_rows_cap=2000, bootstrap_B=40)
log.info("Device=%s threads=%d CFG=%s", DEV, torch.get_num_threads(), CFG)
if DEV.type == "cuda":
    log.info("GPU: %s", torch.cuda.get_device_name(0))

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
LAB_COLS = np.array([v in LABS for v in DYNAMIC_VARS])
VITAL_COLS = np.array([v in VITALS for v in DYNAMIC_VARS])
ORGAN_MASKS = np.stack([np.isin(np.arange(NV), [VIDX[v] for v in vs]) for vs in ORGANS.values()])
st_gen = np.nan_to_num(patients["gender"].to_numpy(float), nan=0.5).astype(np.float32)
CS = np.cumsum(M, axis=0, dtype=np.int32)
CUR = dict(BASE)

def T(a):
    return torch.as_tensor(a, device=DEV)

def use_protocol(proto, uda_split=False):
    """Rebind every protocol-dependent global (normalisation, splits, subsets, constraint stats).
    uda_split: for external protocols, split the target hospital (test) into an unlabelled adaptation half
    (IDX['adapt'], used only by *_uda pretraining) and an evaluation half (IDX['test'])."""
    global P, NORM, subsets, IDX, VN, FN, st_age, row_icu, CONS_P, _mu, _sd, _islog
    P = proto
    NORM = load_json(find_artifact(f"norm_stats_{P}.json", "03_preprocessing"))
    subsets = load_table(find_artifact(f"label_subsets_{P}.parquet", "04_temporal_windowing")).set_index("pidx")
    IDX = {sp: np.load(find_artifact(f"idx_{P}_{sp}.npy", "04_temporal_windowing")) for sp in ("train", "val", "test")}
    if uda_split and P != "pooled":
        tp = np.unique(row_pidx[IDX["test"]])
        rs = np.random.default_rng(GLOBAL_SEED)
        sep = patients.loc[tp, "septic"].to_numpy().astype(bool)
        adapt = np.concatenate([rs.permutation(tp[sep])[: sep.sum() // 2], rs.permutation(tp[~sep])[: (~sep).sum() // 2]])
        is_ad = np.isin(row_pidx[IDX["test"]], adapt)
        IDX["adapt"], IDX["test"] = IDX["test"][is_ad], IDX["test"][~is_ad]
        assert not set(np.unique(row_pidx[IDX["adapt"]])) & set(np.unique(row_pidx[IDX["test"]]))
    VN = normalize_np(V, NORM, DYNAMIC_VARS)
    FN = normalize_np(F, NORM, DYNAMIC_VARS)
    st_age = ((patients["age"].to_numpy(float) - NORM["_static"]["Age"]["mean"]) / NORM["_static"]["Age"]["std"]).astype(np.float32)
    row_icu = ((np.log1p(rows_meta["ICULOS"].to_numpy(float).clip(0)) - NORM["_static"]["log_ICULOS"]["mean"])
               / NORM["_static"]["log_ICULOS"]["std"]).astype(np.float32)
    CONS_P = CONS["per_protocol"][P]
    _mu = T(np.array([NORM[v]["mean"] for v in DYNAMIC_VARS], np.float32))
    _sd = T(np.array([NORM[v]["std"] for v in DYNAMIC_VARS], np.float32))
    _islog = T(np.array([NORM[v]["log1p"] for v in DYNAMIC_VARS]))
    for sp in ("train", "val"):
        assert patients.loc[np.unique(row_pidx[IDX[sp]]), f"split_{P}"].eq(sp).all(), "split leakage"
    assert patients.loc[np.unique(row_pidx[IDX["test"]]), f"split_{P}"].eq("test").all(), "split leakage"
    log.info("protocol %s: rows %s", P, {k: len(v) for k, v in IDX.items()})

# ---- causal batch builder (identical to 06, plus variable-based masking and input modes) ----
OFF = np.arange(-W + 1, 1)
TSTEP = np.arange(W)
LOGMAX = np.log1p(CFG["max_delta_h"])
FIO2 = VIDX["FiO2"]

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
        sel = m & (rng.random(m.shape) < CUR["mask_p"])
    elif mask_mode in ("organ", "variable"):
        Lb = min(CUR["block"], W)
        t0 = rng.integers(0, W - Lb + 1, B)
        tblock = (TSTEP[None, :] >= t0[:, None]) & (TSTEP[None, :] < (t0 + Lb)[:, None])
        if mask_mode == "organ":
            vm = ORGAN_MASKS[rng.integers(0, len(ORGAN_MASKS), B)]
        else:   # same block, same number of variables as an average organ system, but variables drawn at random
            vm = np.zeros((B, NV), bool)
            pick = np.argsort(rng.random((B, NV)), axis=1)[:, :CUR["n_var"]]
            np.put_along_axis(vm, pick, True, axis=1)
        sel = m & tblock[..., None] & vm[:, None, :]
        empty = ~sel.reshape(B, -1).any(1)
        if empty.any():
            sel[empty] = m[empty] & (rng.random(m[empty].shape) < CUR["mask_p"])
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
    mode = CUR["input"]
    if mode == "missing":
        X[..., :NV] = 0.0                       # values hidden: only what was measured and when
    elif mode == "values":
        X[..., NV:3 * NV] = 0.0                 # measurement pattern hidden: only (carried-forward) values
    elif mode == "no_fio2":
        X[..., [FIO2, NV + FIO2, 2 * NV + FIO2]] = 0.0
    st = np.stack([np.nan_to_num(st_age[p]), st_gen[p], row_icu[R]], -1).astype(np.float32)
    return X, st, x, sel, m_in

def set_spec(arm):
    CUR.clear(); CUR.update(VARIANTS.get(arm, BASE))

def causality_check():
    _rng = np.random.default_rng(0)
    up = np.unique(row_pidx[IDX["train"]])
    pids = _rng.choice(up, min(50, len(up)), replace=False)
    R = np.array([_rng.choice(IDX["train"][row_pidx[IDX["train"]] == q]) for q in pids])
    X1 = build(R)[0]
    fut = np.zeros(len(VN), bool)
    for r in R:
        q = row_pidx[r]
        stop = starts[q + 1] if q + 1 < len(starts) else len(VN)
        fut[r + 1:stop] = True
    bkV, bkF = VN[fut].copy(), FN[fut].copy()
    VN[fut] += 100.0; FN[fut] += 100.0
    X2 = build(R)[0]
    VN[fut], FN[fut] = bkV, bkF
    assert np.allclose(X1, X2), "batch builder reads future rows!"
    return X1.shape[-1]

# ---- encoder ----
class Net(nn.Module):
    def __init__(self, d_in, h, enc="gru"):
        super().__init__()
        self.enc = enc
        if enc == "gru":
            self.gru = nn.GRU(d_in, h, batch_first=True)
        else:
            self.inp = nn.Linear(d_in, h)
            self.pos = nn.Parameter(torch.zeros(1, W, h)); nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(h, 4, 2 * h, 0.1, batch_first=True)
            self.tf = nn.TransformerEncoder(layer, 2)
        self.fuse = nn.Sequential(nn.Linear(h + 3, h), nn.ReLU())
        self.recon = nn.Linear(h, NV)
        self.pred = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
        self.clf = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))
    def encode(self, X, st):
        if self.enc == "gru":
            out, hN = self.gru(X)
            last = hN[-1]
        else:
            out = self.tf(self.inp(X) + self.pos[:, -X.shape[1]:])
            last = out[:, -1]
        return out, self.fuse(torch.cat([last, st], -1))
    def forward(self, X, st):
        return self.clf(self.encode(X, st)[1]).squeeze(-1)

def new_net():
    return Net(IN_DIM, CUR["hidden"], CUR["enc"]).to(DEV)

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

# ---- TS2Vec-style hierarchical contrastive loss (Yue et al., AAAI 2022), applied to per-hour GRU outputs ----
def _inst_cl(z1, z2):
    B = z1.size(0)
    if B == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], 0).transpose(0, 1)
    sim = torch.matmul(z, z.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1] + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -TF.log_softmax(logits, dim=-1)
    i = torch.arange(B, device=z1.device)
    return (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2

def _temp_cl(z1, z2):
    Tn = z1.size(1)
    if Tn == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], 1)
    sim = torch.matmul(z, z.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1] + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -TF.log_softmax(logits, dim=-1)
    t = torch.arange(Tn, device=z1.device)
    return (logits[:, t, Tn + t - 1].mean() + logits[:, Tn + t, t].mean()) / 2

def hier_cl(z1, z2, alpha=0.5):
    loss, d = z1.new_tensor(0.0), 0
    while z1.size(1) > 1:
        loss = loss + alpha * _inst_cl(z1, z2) + (1 - alpha) * _temp_cl(z1, z2)
        d += 1
        z1 = TF.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = TF.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        loss = loss + alpha * _inst_cl(z1, z2); d += 1
    return loss / d

# ---- SimMTM-style loss (Dong et al., NeurIPS 2023): reconstruct each window from point-wise representations of
# several masked windows, aggregated by series-wise similarity; plus a series-wise contrastive constraint ----
SIMMTM_K, SIMMTM_TAU, SIMMTM_LAMBDA = 3, 0.2, 0.1
def simmtm_loss(net, R, rng):
    X0, st0, x0, _, m0 = build(R)
    views = [build(R, "random", rng) for _ in range(SIMMTM_K)]
    Xa = np.concatenate([X0] + [v[0] for v in views]); sta = np.concatenate([st0] + [v[1] for v in views])
    out, z = net.encode(T(Xa), T(sta))
    B = len(R); N = B * (SIMMTM_K + 1)
    zn = TF.normalize(z, dim=-1)
    S = zn @ zn.T / SIMMTM_TAU
    S = S.masked_fill(torch.eye(N, dtype=torch.bool, device=DEV), -1e9)
    Wt = torch.softmax(S, dim=-1)
    agg = (Wt @ out.reshape(N, -1)).reshape(N, W, -1)[:B]                 # aggregated for the original windows
    pred = net.recon(agg)
    xt, mt = T(x0), T(m0)
    rec = ((pred - torch.nan_to_num(xt)) ** 2)[mt].mean() if m0.any() else pred.sum() * 0.0
    ids = torch.arange(N, device=DEV) % B
    pos = (ids[:, None] == ids[None, :]) & ~torch.eye(N, dtype=torch.bool, device=DEV)
    logp = TF.log_softmax(S, dim=-1)
    con = -(torch.logsumexp(logp.masked_fill(~pos, -1e9), dim=-1)).mean()
    return rec + SIMMTM_LAMBDA * con, float(rec), float(con)

CKPT = WORK / "checkpoints"; CKPT.mkdir(exist_ok=True)

def pretrain(arm, seed, tag=""):
    set_spec(arm)
    sp = CUR
    path = CKPT / f"pretrain_{arm}_{P}_s{seed}{tag}.pt"
    if path.exists():
        return torch.load(path, map_location=DEV)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    net = new_net()
    opt = torch.optim.Adam(net.parameters(), lr=CFG["lr"])
    pool = IDX["train"]
    if sp.get("uda"):
        pool = np.concatenate([pool, IDX["adapt"]])      # unlabelled target-hospital windows; labels never read
    hist, t0 = [], time.time()
    for ep in range(CFG["pre_epochs"]):
        R_ep = rng.choice(pool, size=min(CFG["pre_windows_per_epoch"], len(pool)), replace=False)
        sums = {"rec": 0.0, "phys": 0.0, "temp": 0.0, "con": 0.0}; nb = 0
        net.train()
        for i in range(0, len(R_ep), CFG["batch"]):
            R = R_ep[i:i + CFG["batch"]]
            if sp["ssl"] == "ts2vec":
                X1, st1, *_ = build(R, "random", rng)
                X2, st2, *_ = build(R, "random", rng)
                o1 = net.encode(T(X1), T(st1))[0]; o2 = net.encode(T(X2), T(st2))[0]
                loss = hier_cl(o1, o2)
                sums["con"] += float(loss)
            elif sp["ssl"] == "simmtm":
                loss, rec, con = simmtm_loss(net, R, rng)
                sums["rec"] += rec; sums["con"] += con
            else:
                X, st, x, sel, m_in = build(R, sp["mask"], rng)
                if not sel.any():
                    continue
                Xt, stt, xt, selt, mint = T(X), T(st), T(x), T(sel), T(m_in)
                out, z = net.encode(Xt, stt)
                pred = net.recon(out)
                rec = ((pred - torch.nan_to_num(xt)) ** 2)[selt].mean()
                loss = rec
                if sp["lp"] > 0:
                    phys = physiology_loss(pred, xt, selt, mint)
                    loss = loss + sp["lp"] * phys
                    sums["phys"] += float(phys)
                if sp["lt"] > 0:
                    Rp = np.where(R > starts[row_pidx[R]], R - 1, R)
                    Xp, stp, *_ = build(Rp)
                    with torch.no_grad():
                        zp = net.encode(T(Xp), T(stp))[1]
                    temp = (1 - TF.cosine_similarity(net.pred(z), zp, dim=-1)).mean()
                    loss = loss + sp["lt"] * temp
                    sums["temp"] += float(temp)
                sums["rec"] += float(rec)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); nb += 1
        hist.append({"arm": arm, "protocol": P, "seed": seed, "epoch": ep, **{k: v / max(nb, 1) for k, v in sums.items()},
                     "seconds": time.time() - t0})
        log.info("pretrain %s %s s%d ep%d %s", arm, P, seed, ep, {k: round(v / max(nb, 1), 4) for k, v in sums.items() if v})
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
    set_spec(arm)
    set_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    net = new_net()
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

# ---- LightGBM (full features, or missingness-only / values-only feature sets) ----
VITAL_IDX = [VIDX[v] for v in ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]]
def tab_features(R, mode="full"):
    R = np.asarray(R)
    p = row_pidx[R]; s0 = starts[p]
    lo = np.maximum(s0, R - 23)
    cnt24 = CS[R] - np.where((lo > 0)[:, None], CS[np.maximum(lo - 1, 0)], 0)
    idx6 = R[:, None] + np.arange(-5, 1)
    ok6 = (idx6 >= s0[:, None])[..., None] & M[np.maximum(idx6, 0)][:, :, VITAL_IDX]
    v6 = np.where(ok6, VN[np.maximum(idx6, 0)][:, :, VITAL_IDX], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        roll = np.concatenate([np.nanmean(v6, 1), np.nanmin(v6, 1), np.nanmax(v6, 1), np.nanstd(v6, 1)], 1)
    stat = np.stack([st_age[p], st_gen[p], row_icu[R]], 1)
    if mode == "missing":      # when and how often each variable was measured, nothing about values
        return np.concatenate([np.nan_to_num(AGE[R], nan=CFG["max_delta_h"]), cnt24.astype(np.float32), stat], 1).astype(np.float32)
    if mode == "values":       # carried-forward values only; never-measured = training mean (0), no counts or ages
        return np.concatenate([np.nan_to_num(FN[R], nan=0.0), np.nan_to_num(roll, nan=0.0), stat], 1).astype(np.float32)
    return np.concatenate([FN[R], AGE[R], cnt24.astype(np.float32), roll, stat], 1).astype(np.float32)

def train_lgbm(frac, seed, mode="full"):
    Rtr, Rva = rows_of("train", frac, seed), rows_of("val", frac, seed)
    t0 = time.time()
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, min_child_samples=50, random_state=seed, verbose=-1, n_jobs=-1)
    m.fit(tab_features(Rtr, mode), y_all[Rtr], eval_set=[(tab_features(Rva, mode), y_all[Rva])], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, Rva, time.time() - t0

# ---- evaluation ----
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

def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def platt(pva, yva, pte):
    """Validation-only recalibration (Platt scaling on the logit). Monotone, so ranking metrics are unchanged."""
    if yva.min() == yva.max():
        return pte, (1.0, 0.0)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(pva)[:, None], yva)
    return lr.predict_proba(_logit(pte)[:, None])[:, 1], (float(lr.coef_[0, 0]), float(lr.intercept_[0]))

def evaluate(R, prob, thr, full=True):
    y = y_all[R]
    pb = (prob >= thr).astype(int)
    out = {"AUROC": roc_auc_score(y, prob), "AUPRC": average_precision_score(y, prob), "utility": utility_norm(R, pb)}
    if not full:
        return out
    tp = int(((pb == 1) & (y == 1)).sum()); fp = int(((pb == 1) & (y == 0)).sum())
    fn = int(((pb == 0) & (y == 1)).sum()); tn = int(((pb == 0) & (y == 0)).sum())
    pdf = pd.DataFrame({"p": row_pidx[R], "prob": prob, "y": y, "alarm": pb, "t": row_t[R]})
    pat = pdf.groupby("p").agg(maxp=("prob", "max"), sep=("y", "max"))
    non = pdf[~pdf.p.isin(pat.index[pat.sep == 1])]
    leads, onset_alarm = [], []
    tsep = patients["t_sepsis_idx"].to_numpy()
    for pid, g in pdf[pdf.p.isin(pat.index[pat.sep == 1])].groupby("p"):
        ts = tsep[pid]
        al = g.loc[(g.alarm == 1) & (g.t <= ts + 3), "t"]
        if len(al):
            leads.append(ts - al.min())
            onset_alarm.append(((g.alarm == 1) & (g.t >= ts - 12) & (g.t <= ts + 3)).any())
    sens = tp / max(tp + fn, 1); spec_ = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    nsep = max((pat.sep == 1).sum(), 1)
    out.update({"patient_AUROC": roc_auc_score(pat.sep, pat.maxp), "Brier": brier_score_loss(y, prob), "ECE": ece(y, prob),
                "mean_pred": float(prob.mean()), "obs_rate": float(y.mean()),
                "threshold": thr, "sensitivity": sens, "specificity": spec_, "precision": prec,
                "F1": 2 * prec * sens / max(prec + sens, 1e-9),
                "false_alarms_per_patient_day": non.alarm.sum() / max(len(non) / 24, 1e-9),
                "non_septic_patients_with_alarm_pct": 100 * non.groupby("p").alarm.max().mean(),
                "septic_detected_pct": 100 * len(leads) / nsep,
                "septic_alarm_in_window_pct": 100 * float(np.sum(onset_alarm)) / nsep,
                "median_lead_time_h": float(np.median(leads)) if leads else np.nan})
    return out

def run_one(arm, frac, seed, pre_state):
    """Train one arm, score the test set; returns test preds, metrics and the fitted model (for perturbation tests)."""
    if arm in LGBM_ARMS:
        mode = LGBM_ARMS[arm]
        model, Rva, secs = train_lgbm(frac, seed, mode)
        epochs = model.best_iteration_
        pva = model.predict_proba(tab_features(Rva, mode))[:, 1]
        t1 = time.time(); pte = model.predict_proba(tab_features(IDX["test"], mode))[:, 1]; inf = time.time() - t1
    else:
        model, Rva, secs, epochs = train_neural(arm, frac, seed, pre_state)
        pva = predict(model, Rva)
        t1 = time.time(); pte = predict(model, IDX["test"]); inf = time.time() - t1
    thr = choose_threshold(Rva, pva)
    met = evaluate(IDX["test"], pte, thr)
    pcal, (a, b) = platt(pva, y_all[Rva], pte)
    met.update({"Brier_cal": brier_score_loss(y_all[IDX["test"]], pcal), "ECE_cal": ece(y_all[IDX["test"]], pcal),
                "mean_pred_cal": float(pcal.mean()), "platt_a": a, "platt_b": b})
    return pte, met, secs, inf, epochs, model, thr

def predict_any(arm, model, R):
    if arm in LGBM_ARMS:
        return model.predict_proba(tab_features(R, LGBM_ARMS[arm]))[:, 1]
    set_spec(arm)
    return predict(model, R)

# ---- test-time measurement-pattern perturbations (pooled test patients only; observations are only REMOVED) ----
@contextmanager
def using(arrs):
    g = globals(); old = {k: g[k] for k in arrs}
    g.update(arrs)
    try:
        yield
    finally:
        g.update(old)

def _cap_frequency(Mt, rows, cols, gap_h=24):
    """Keep an observation only if >= gap_h hours after the last KEPT observation of that variable (same patient)."""
    drop = np.zeros_like(Mt)
    pr, tr = row_pidx[rows], row_t[rows]
    for j in np.flatnonzero(cols):
        obs = np.flatnonzero(Mt[:, j])
        last_p, last_t = -1, -1e9
        for i in obs:
            if pr[i] != last_p:
                last_p, last_t = pr[i], tr[i]
                continue
            if tr[i] - last_t < gap_h:
                drop[i, j] = True
            else:
                last_t = tr[i]
    return drop

def make_perturbation(name, rows, seed=0):
    """Return perturbed copies of M, VN, FN, AGE, CS where only `rows` (whole test patients) are modified."""
    rng = np.random.default_rng(seed)
    Mt = M[rows]
    if name == "drop_labs_30":
        drop = Mt & LAB_COLS[None, :] & (rng.random(Mt.shape, dtype=np.float32) < 0.3)
    elif name == "drop_vitals_30":
        drop = Mt & VITAL_COLS[None, :] & (rng.random(Mt.shape, dtype=np.float32) < 0.3)
    elif name == "no_lactate":
        drop = Mt & (np.arange(NV) == VIDX["Lactate"])[None, :]
    elif name == "labs_max_1_per_24h":
        drop = _cap_frequency(Mt, rows, LAB_COLS, 24)
    else:
        raise ValueError(name)
    Mp_t = Mt & ~drop
    Vt = np.where(Mp_t, V[rows], np.nan).astype(np.float32)
    dfv = pd.DataFrame(Vt, columns=DYNAMIC_VARS); dfv["pidx"] = row_pidx[rows]
    Ft = dfv.groupby("pidx")[DYNAMIC_VARS].ffill().to_numpy(np.float32)
    lt = pd.DataFrame(np.where(Mp_t, row_t[rows][:, None], np.nan), columns=DYNAMIC_VARS); lt["pidx"] = row_pidx[rows]
    lt = lt.groupby("pidx")[DYNAMIC_VARS].ffill().to_numpy(np.float32)
    Mp = M.copy(); Mp[rows] = Mp_t
    VNp = VN.copy(); VNp[rows] = normalize_np(Vt, NORM, DYNAMIC_VARS)
    FNp = FN.copy(); FNp[rows] = normalize_np(Ft, NORM, DYNAMIC_VARS)
    AGEp = AGE.copy(); AGEp[rows] = np.minimum(row_t[rows][:, None] - lt, CFG["max_delta_h"]).astype(np.float32)
    CSp = np.cumsum(Mp, axis=0, dtype=np.int32)
    info = {"perturbation": name, "observations_removed": int(drop.sum()), "observations_before": int(Mt.sum())}
    return {"M": Mp, "VN": VNp, "FN": FNp, "AGE": AGEp, "CS": CSp}, info

# ---- fast paired bootstrap engine: patient-level weights, tie-aware weighted AUROC / AUPRC on the GPU ----
class Boot:
    def __init__(self, R, B, seed, chunk=32):
        self.y = torch.as_tensor(y_all[R], device=DEV, dtype=torch.float32)
        pr = row_pidx[R]
        _, self.prow = np.unique(pr, return_inverse=True)
        self.nP = int(self.prow.max()) + 1
        self.prow_t = torch.as_tensor(self.prow, device=DEV)
        self.B, self.chunk = B, chunk
        self.gen = torch.Generator(device=DEV); self.seed = seed
        self.keys, self.order, self.ends, self.ysorted = [], {}, {}, {}
    def add(self, key, p):
        p = torch.as_tensor(np.asarray(p, np.float32), device=DEV)
        o = torch.argsort(p, descending=True, stable=True)
        ps = p[o]
        ends = torch.nonzero(torch.cat([ps[1:] != ps[:-1], torch.ones(1, dtype=torch.bool, device=DEV)])).squeeze(1)
        self.keys.append(key); self.order[key] = o; self.ends[key] = ends; self.ysorted[key] = self.y[o]
    @staticmethod
    def _auc_ap(Ws, ys, ends):
        tp = torch.cumsum(Ws * ys, 1)[:, ends]
        fp = torch.cumsum(Ws * (1 - ys), 1)[:, ends]
        TP, FP = tp[:, -1:], fp[:, -1:]
        tp0 = torch.cat([torch.zeros_like(TP), tp[:, :-1]], 1)
        fp0 = torch.cat([torch.zeros_like(FP), fp[:, :-1]], 1)
        dtp, dfp = tp - tp0, fp - fp0
        auc = (dfp * (tp0 + dtp / 2)).sum(1) / (TP * FP).squeeze(1)
        prec = tp / torch.clamp(tp + fp, min=1e-12)
        ap = (dtp / TP * prec).sum(1)
        return auc, ap
    def run(self, identity=False):
        """Returns {key: (B, 2) array of [AUROC, AUPRC]} for the same B patient resamples (paired across keys).
        identity=True -> one replicate with unit weights (used to check against scikit-learn)."""
        out = {k: [] for k in self.keys}
        self.gen.manual_seed(self.seed)
        nrep = 1 if identity else self.B
        for c0 in range(0, nrep, self.chunk):
            C = min(self.chunk, nrep - c0)
            if identity:
                cnt = torch.ones(1, self.nP, device=DEV)
            else:
                draw = torch.randint(0, self.nP, (C, self.nP), device=DEV, generator=self.gen)
                cnt = torch.zeros(C, self.nP, device=DEV).scatter_add_(1, draw, torch.ones(C, self.nP, device=DEV))
            Wrow = cnt[:, self.prow_t]
            for k in self.keys:
                auc, ap = self._auc_ap(Wrow[:, self.order[k]], self.ysorted[k], self.ends[k])
                out[k].append(torch.stack([auc, ap], 1).cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}

def holm(p):
    p = np.asarray(p, float); n = len(p)
    order = np.argsort(p); adj = np.empty(n); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (n - r) * p[i])); adj[i] = run
    return adj

def pval(d):
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))

def compare(BS, a, b, frac, seeds, rng, Bn):
    """Two estimates for arm a minus arm b at one label fraction:
    (1) per-seed: mean over seeds of the per-seed difference; CI from a hierarchical bootstrap that resamples
        seeds AND patients (accounts for training variability and test-set sampling);
    (2) seed-ensemble: difference of the seed-averaged predictions; CI from the patient bootstrap only."""
    rows = []
    S = [s for s in seeds if (a, frac, s) in BS and (b, frac, s) in BS]
    if len(S) < 2:
        return rows
    sidx = rng.integers(0, len(S), (Bn, len(S)))
    for k, name in ((0, "AUROC"), (1, "AUPRC")):
        A = np.stack([BS[(a, frac, s)][:, k] for s in S], 1)
        Bm = np.stack([BS[(b, frac, s)][:, k] for s in S], 1)
        D = A - Bm
        hier = np.take_along_axis(D, sidx, axis=1).mean(1)
        ens = BS[(a, frac, "ens")][:, k] - BS[(b, frac, "ens")][:, k]
        rows.append({"fraction": frac, "comparison": f"{a} - {b}", "metric": name, "n_seeds": len(S),
                     "perseed_diff": float(np.mean([PT[(a, frac, s)][k] - PT[(b, frac, s)][k] for s in S])),
                     "perseed_lo": np.percentile(hier, 2.5), "perseed_hi": np.percentile(hier, 97.5), "perseed_p": pval(hier),
                     "ensemble_diff": float(PT[(a, frac, "ens")][k] - PT[(b, frac, "ens")][k]),
                     "ensemble_lo": np.percentile(ens, 2.5), "ensemble_hi": np.percentile(ens, 97.5), "ensemble_p": pval(ens)})
    return rows

PT = {}   # point estimates (AUROC, AUPRC) on the real test set for every (arm, frac, seed|'ens')

def bootstrap_protocol(proto, runs_sub, pred_path_fn, Bn, seed=GLOBAL_SEED):
    """Load every saved test prediction of one protocol (+ per-arm seed ensembles) into one paired bootstrap."""
    PT.clear()
    eng = Boot(IDX["test"], Bn, seed)
    yte = y_all[IDX["test"]]
    keys = []
    for (arm, frac), g in runs_sub.groupby(["arm", "fraction"]):
        ps = []
        for s in sorted(g.seed):
            f = pred_path_fn(proto, arm, frac, s)
            if f.exists():
                p = np.load(f); ps.append(p)
                eng.add((arm, frac, s), p); keys.append((arm, frac, s))
                PT[(arm, frac, s)] = (roc_auc_score(yte, p), average_precision_score(yte, p))
        if ps:
            pe = np.mean(ps, 0)
            eng.add((arm, frac, "ens"), pe); keys.append((arm, frac, "ens"))
            PT[(arm, frac, "ens")] = (roc_auc_score(yte, pe), average_precision_score(yte, pe))
    # engine self-check against scikit-learn on unit weights
    chk = eng.run(identity=True)
    for k in keys[:3]:
        assert abs(chk[k][0, 0] - PT[k][0]) < 1e-4 and abs(chk[k][0, 1] - PT[k][1]) < 1e-4, ("bootstrap engine mismatch", k, chk[k][0], PT[k])
    return eng.run()

def factorial_effects(BS, frac, seeds, rng, Bn):
    """Main effects and interactions of O / P / T (+-1 coding, effect = contrast / 4), per-seed hierarchical CI."""
    cells = {v: k for k, v in FACTORIAL.items()}
    S = [s for s in seeds if all((cells[c], frac, s) in BS for c in cells)]
    if len(S) < 2:
        return []
    sidx = rng.integers(0, len(S), (Bn, len(S)))
    terms = {"O": (0,), "P": (1,), "T": (2,), "OxP": (0, 1), "OxT": (0, 2), "PxT": (1, 2), "OxPxT": (0, 1, 2)}
    rows = []
    for k, name in ((0, "AUROC"), (1, "AUPRC")):
        cube = {c: np.stack([BS[(cells[c], frac, s)][:, k] for s in S], 1) for c in cells}
        pt = {c: np.mean([PT[(cells[c], frac, s)][k] for s in S]) for c in cells}
        for tname, idxs in terms.items():
            sign = {c: np.prod([1 if c[i] else -1 for i in idxs]) for c in cells}
            eff_b = sum(sign[c] * cube[c] for c in cells) / 4
            eff_b = np.take_along_axis(eff_b, sidx, axis=1).mean(1)
            eff = sum(sign[c] * pt[c] for c in cells) / 4
            rows.append({"fraction": frac, "term": tname, "metric": name, "n_seeds": len(S), "effect": eff,
                         "lo": np.percentile(eff_b, 2.5), "hi": np.percentile(eff_b, 97.5), "p": pval(eff_b)})
    return rows
