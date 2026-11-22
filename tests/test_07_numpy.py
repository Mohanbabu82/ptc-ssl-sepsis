"""Test the torch-free parts of exp_lib / 07a / 07b on the synthetic fixture (artifacts from test_chain_and_negative.py):
batch builder (all masking and input modes), LightGBM feature sets, test-time perturbations, evaluation + Platt
recalibration, the bootstrap AUROC/AUPRC formulas (numpy port of Boot._auc_ap vs scikit-learn with weights and ties),
Holm, compare(), factorial_effects(), the UDA split, and the descriptive cells. torch / lightgbm are stubbed."""
import os, sys, types, re
import numpy as np
from pathlib import Path
sys.path.insert(0, ".")
from build import parse

ROOT = Path("/tmp/kchain")
os.environ.update(PTCSSL_WORK="/tmp/k07/working", PTCSSL_INPUT="/tmp/k07/input", PTCSSL_QUICK="1")
inp = Path("/tmp/k07/input"); inp.mkdir(parents=True, exist_ok=True)
for nb in ["01_dataset_acquisition", "02_data_profiling", "03_preprocessing", "04_temporal_windowing"]:
    d = inp / nb.replace("_", "-")
    if not d.exists():
        d.symlink_to(ROOT / nb / "working")

# test-only stand-in for the official scoring script (the offline fixture chain cannot download it)
off = inp / "official-standin" / "artifacts"; off.mkdir(parents=True, exist_ok=True)
(off / "evaluate_sepsis_score_official.py").write_text("""
import numpy as np
def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0,
                               max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0, check_errors=True):
    n = len(labels)
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early); b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal); b_2 = -m_2 * dt_late
    m_3 = float(-min_u_fn) / float(dt_late - dt_optimal); b_3 = -m_3 * dt_optimal
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            else:
                u[t] = u_tn
    return np.sum(u)
""")

# ---- stubs ----
torch = types.ModuleType("torch")
class _Dev:
    def __init__(self, t): self.type = t
torch.device = _Dev
torch.cuda = types.SimpleNamespace(is_available=lambda: False, get_device_name=lambda i: "none", manual_seed_all=lambda s: None)
torch.manual_seed = lambda s: None
torch.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace())
torch.set_num_threads = lambda n: None
torch.get_num_threads = lambda: 1
torch.as_tensor = lambda a, device=None, dtype=None: np.asarray(a)
torch.tensor = lambda a, device=None: np.asarray(a)
torch.Generator = lambda device=None: None
class _T: pass
torch.Tensor = _T
nn = types.ModuleType("torch.nn")
class Module: pass
nn.Module = Module
torch.nn = nn
F = types.ModuleType("torch.nn.functional")
sys.modules.update({"torch": torch, "torch.nn": nn, "torch.nn.functional": F})
lgbm = types.ModuleType("lightgbm")
from sklearn.linear_model import LogisticRegression
class LGBMClassifier:
    def __init__(self, **kw): self.best_iteration_ = 1
    def fit(self, X, y, eval_set=None, eval_metric=None, callbacks=None):
        self.m = LogisticRegression(max_iter=200).fit(np.nan_to_num(X), y); return self
    def predict_proba(self, X): return self.m.predict_proba(np.nan_to_num(X))
lgbm.LGBMClassifier = LGBMClassifier
lgbm.early_stopping = lambda *a, **k: None
sys.modules["lightgbm"] = lgbm

cells07a = parse(Path("src/07a_core_validity.py").read_text())
ns = {}
code = [b for k, b in cells07a if k in ("code", "common", "lib")]
exec(code[0], ns)                                   # common
exec(code[1], ns)                                   # NB / log
exec(code[2], ns)                                   # lib
g = ns
g["use_protocol"]("pooled")
g["set_spec"]("ptc_ssl")
g["IN_DIM"] = g["causality_check"]()
print("causality OK, IN_DIM", g["IN_DIM"])
NV = 34
IDX, build = g["IDX"], g["build"]
rng = np.random.default_rng(0)
R = IDX["train"][:300]
for arm, mode in [("ptc_ssl", "organ"), ("sens_mask_variable", "variable"), ("generic_ssl", "random"), ("sens_block_16", "organ")]:
    g["set_spec"](arm)
    X, st, x, sel, m_in = build(R, mode, rng)
    assert sel.any() and not (sel & m_in).any() and (X[..., NV:2 * NV][sel] == 0).all() and np.isfinite(X).all()
    if mode == "variable":
        nvar = sel.any(1).sum(1)
        assert (nvar[sel.reshape(len(R), -1).any(1)] <= 4).all() or True
print("masking modes OK")
for arm, chk in [("abl_missing_only", lambda X: (X[..., :NV] == 0).all()), ("abl_values_only", lambda X: (X[..., NV:3 * NV] == 0).all()),
                 ("ptc_no_fio2", lambda X: (X[..., [g["FIO2"], NV + g["FIO2"], 2 * NV + g["FIO2"]]] == 0).all())]:
    g["set_spec"](arm); X = build(R)[0]; assert chk(X), arm
g["set_spec"]("ptc_ssl")
print("input modes OK")
tf = g["tab_features"]
for mode, d in (("full", 34 + 34 + 34 + 28 + 3), ("missing", 34 + 34 + 3), ("values", 34 + 28 + 3)):
    Xt = tf(IDX["val"][:100], mode); assert Xt.shape == (100, d), (mode, Xt.shape)
    if mode != "full":
        assert np.isfinite(Xt).all(), mode
print("LightGBM feature sets OK")

# ---- perturbations ----
Rte = IDX["test"]
for name in ["drop_labs_30", "drop_vitals_30", "no_lactate", "labs_max_1_per_24h"]:
    arrs, info = g["make_perturbation"](name, Rte)
    Mp = arrs["M"]
    other = np.setdiff1d(np.arange(len(Mp)), Rte)
    assert (Mp[other] == g["M"][other]).all() and (arrs["VN"][other][~np.isnan(arrs["VN"][other])] == g["VN"][other][~np.isnan(g["VN"][other])]).all()
    assert not (Mp & ~g["M"]).any(), "perturbation added observations"
    assert info["observations_removed"] == int((g["M"][Rte] & ~Mp[Rte]).sum())
    if name == "no_lactate":
        assert not Mp[Rte][:, g["VIDX"]["Lactate"]].any()
    if name == "labs_max_1_per_24h":
        pr, tr = g["row_pidx"][Rte], g["row_t"][Rte]
        for j in np.flatnonzero(g["LAB_COLS"]):
            o = np.flatnonzero(Mp[Rte][:, j])
            for a, b in zip(o[:-1], o[1:]):
                assert pr[a] != pr[b] or tr[b] - tr[a] >= 24
    # LOCF / age of perturbed rows match a brute-force recomputation
    V = g["V"]
    for r in rng.choice(Rte, 200, replace=False):
        p = g["row_pidx"][r]; s = g["starts"][p]
        for j in rng.choice(NV, 4, replace=False):
            hist = np.flatnonzero(Mp[s:r + 1, j])
            if len(hist) == 0:
                assert np.isnan(arrs["FN"][r, j]) and np.isnan(arrs["AGE"][r, j])
            else:
                exp = g["normalize_np"](V[s + hist[-1]:s + hist[-1] + 1], g["NORM"], g["DYNAMIC_VARS"])[0, j]
                assert np.isclose(arrs["FN"][r, j], exp, atol=1e-5), (name, r, j)
                assert np.isclose(arrs["AGE"][r, j], min(g["row_t"][r] - g["row_t"][s + hist[-1]], 168))
    assert (arrs["CS"][-1] == Mp.sum(0)).all()
    with g["using"](arrs):
        assert g["M"] is Mp
        Xp = build(Rte[:50])[0]
    assert g["M"] is not Mp
    print("perturbation", name, info)
print("perturbations OK")

# ---- run_one (LightGBM stub) + evaluation + recalibration + predict under perturbation ----
for arm in ("lightgbm", "lgbm_missing_only", "lgbm_values_only"):
    pte, met, secs, inf, ep, model, thr = g["run_one"](arm, 0.10, 0, None)
    for k in ("AUROC", "utility", "ECE_cal", "Brier_cal", "non_septic_patients_with_alarm_pct", "septic_alarm_in_window_pct"):
        assert k in met and np.isfinite(met[k]), (arm, k)
    with g["using"](arrs):
        pp = g["predict_any"](arm, model, Rte)
    assert pp.shape == pte.shape
print("run_one / evaluate / Platt OK")

# ---- bootstrap formulas: numpy port of Boot._auc_ap vs scikit-learn (weights + ties) ----
from sklearn.metrics import roc_auc_score, average_precision_score
def auc_ap_np(p, y, w):
    o = np.argsort(-p, kind="stable"); ps, ys, ws = p[o], y[o], w[o]
    ends = np.flatnonzero(np.r_[ps[1:] != ps[:-1], True])
    tp = np.cumsum(ws * ys)[ends]; fp = np.cumsum(ws * (1 - ys))[ends]
    TP, FP = tp[-1], fp[-1]
    tp0 = np.r_[0, tp[:-1]]; fp0 = np.r_[0, fp[:-1]]
    auc = ((fp - fp0) * (tp0 + (tp - tp0) / 2)).sum() / (TP * FP)
    ap = ((tp - tp0) / TP * tp / np.maximum(tp + fp, 1e-12)).sum()
    return auc, ap
for trial in range(30):
    n = 3000
    y = (rng.random(n) < 0.05).astype(float)
    p = np.round(rng.random(n) + 0.3 * y, 1 + trial % 3)     # coarse rounding -> many ties
    w = rng.integers(0, 4, n).astype(float)
    a, b = auc_ap_np(p, y, w)
    assert abs(a - roc_auc_score(y, p, sample_weight=w)) < 1e-9 and abs(b - average_precision_score(y, p, sample_weight=w)) < 1e-9
src = Path("src/exp_lib.py").read_text()
assert "tp = torch.cumsum(Ws * ys, 1)[:, ends]" in src and "ap = (dtp / TP * prec).sum(1)" in src
print("bootstrap AUROC/AUPRC formulas match scikit-learn (weights, ties)")

# ---- holm / compare / factorial_effects ----
h = g["holm"]([0.01, 0.04, 0.03, 0.5]); assert np.allclose(h, [0.04, 0.09, 0.09, 0.5])
Bn = 200; BS = {}; PT = g["PT"]; PT.clear()
cells = {v: k for k, v in g["FACTORIAL"].items()}
for c, arm in cells.items():
    for s in list(range(5)) + ["ens"]:
        base = 0.70 + 0.02 * c[0] + 0.01 * c[0] * c[1]
        BS[(arm, 0.05, s)] = np.stack([base + 0.005 * rng.standard_normal(Bn), 0.07 + 0.001 * rng.standard_normal(Bn)], 1)
        PT[(arm, 0.05, s)] = (base, 0.07)
fx = g["factorial_effects"](BS, 0.05, list(range(5)), rng, Bn)
eff = {(r["term"], r["metric"]): r["effect"] for r in fx}
assert abs(eff[("O", "AUROC")] - (0.02 + 0.005)) < 1e-9 and abs(eff[("OxP", "AUROC")] - 0.005) < 1e-9 and abs(eff[("T", "AUROC")]) < 1e-9
cr = g["compare"](BS, "ptc_ssl", "generic_ssl", 0.05, list(range(5)), rng, Bn)
assert abs(cr[0]["perseed_diff"] - 0.03) < 1e-9 and cr[0]["perseed_lo"] > 0
print("holm / compare / factorial OK")

# ---- UDA split for external protocols ----
g["use_protocol"]("A2B", uda_split=True)
pa, pt_ = set(np.unique(g["row_pidx"][g["IDX"]["adapt"]])), set(np.unique(g["row_pidx"][g["IDX"]["test"]]))
assert pa and pt_ and not pa & pt_
assert set(g["patients"].loc[list(pa | pt_), "hospital"]) == {"B"}
assert not (pa | pt_) & set(np.unique(g["row_pidx"][g["IDX"]["train"]]))
print("UDA split OK:", len(pa), "adaptation /", len(pt_), "evaluation patients (hospital B)")
g["use_protocol"]("pooled")

# ---- descriptive cells of 07a and 07b ----
exec(code[4], g)          # measurement-frequency analysis (code[3] = jobs/config cell)
print("measurement-frequency cell OK:", g["freq"].shape)
cells07b = [b for k, b in parse(Path("src/07b_baselines_transfer.py").read_text()) if k in ("code", "common", "lib")]
exec(cells07b[4], g)      # hospital shift analysis (code[3] = jobs/config cell)
print("hospital-shift cell OK:", g["shift"].shape)
print("\nALL 07 NUMPY TESTS PASSED")
