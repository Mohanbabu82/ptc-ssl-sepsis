"""Test the torch-free parts of notebook 05 (batch builder + causality, SSL masking, LightGBM features,
utility/threshold/evaluate) on the synthetic fixture. torch/lightgbm are stubbed; model code is not exercised."""
import sys, types, io, contextlib, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, ".")
from build import code_cells
cells = code_cells("src/05_experiments.py")
ns = {}
exec(cells[0], ns)                                     # common setup
cfg_cell = cells[1]
cfg_src = cfg_cell[cfg_cell.index("CFG = {"):cfg_cell.index("if os.environ.get(\"PTCSSL_QUICK\")")]
exec("import warnings\nNB='05t'\nlog=get_logger(NB)\n" + cfg_src, ns)
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
ns.update(roc_auc_score=roc_auc_score, average_precision_score=average_precision_score, brier_score_loss=brier_score_loss)
exec(cells[2], ns)                                     # load artifacts
exec(cells[3], ns)                                     # builder + causality assertion
X, st, x, sel, m_in = ns["build"](ns["IDX"]["train"][:64])
assert X.shape == (64, 24, 34 * 3 + 1) and not np.isnan(X).any() and not sel.any()
rng = np.random.default_rng(0)
for mode in ("random", "organ"):
    X, st, x, sel, m_in = ns["build"](ns["IDX"]["train"][:256], mode, rng)
    assert sel.any() and not (sel & m_in).any() and np.isfinite(x[sel]).all()
    # masked entries must be invisible in the input mask channel
    assert (X[..., 34:68][sel] == 0).all()
print("builder + masking OK, input dim", X.shape[-1])
# lightgbm features (stub lgb)
lgb_cell = cells[[i for i, c in enumerate(cells) if "import lightgbm as lgb" in c][0]]
sys.modules["lightgbm"] = types.ModuleType("lightgbm")
exec(lgb_cell, ns)
f = ns["tab_features"](ns["IDX"]["test"][:500])
assert f.shape[0] == 500 and np.isfinite(f[:, 34:68]).sum() > 0
# counts must equal brute force
R = ns["IDX"]["test"][:200]; Mb = ns["M"]; st0 = ns["starts"]; rp = ns["row_pidx"]
for k, r in enumerate(R):
    lo = max(st0[rp[r]], r - 23)
    assert (f[k, 68:102] == Mb[lo:r + 1].sum(0)).all()
print("tab features OK", f.shape)
eval_cell = cells[[i for i, c in enumerate(cells) if "def evaluate(" in c][0]]
exec(eval_cell, ns)
Rte = ns["IDX"]["test"]; y = ns["y_all"][Rte]
perfect = y.astype(float)
u_inact = ns["utility_norm"](Rte, np.zeros(len(Rte), int))
assert abs(u_inact) < 1e-9, u_inact
thr = ns["choose_threshold"](ns["IDX"]["val"], np.random.default_rng(1).random(len(ns["IDX"]["val"])))
met = ns["evaluate"](Rte, perfect * 0.9 + 0.05, 0.5)
assert met["AUROC"] == 1.0 and met["sensitivity"] == 1.0
print("evaluate OK:", {k: round(v, 3) for k, v in met.items() if isinstance(v, float)})
