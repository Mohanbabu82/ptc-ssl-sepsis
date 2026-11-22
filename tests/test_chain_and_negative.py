"""(1) Run 01->04 as separate Kaggle sessions: each notebook gets a fresh /kaggle/working and sees earlier
outputs only as attached inputs. (2) Negative control: a window builder that peeks one hour ahead must be caught."""
import os, subprocess, shutil, sys, numpy as np
from pathlib import Path
root = Path("/tmp/kchain"); shutil.rmtree(root, ignore_errors=True)
env = dict(os.environ, PTCSSL_OFFLINE="1", PTCSSL_LOCAL_SOURCE="/tmp/fx", PTCSSL_RAW_DIR="/tmp/kchain/raw",
           PTCSSL_EXPECTED_COUNTS='{"A":120,"B":80}', PTCSSL_MIN_CO="30")
nbs = ["01_dataset_acquisition", "02_data_profiling", "03_preprocessing", "04_temporal_windowing"]
for i, nb in enumerate(nbs):
    work, inp = root / nb / "working", root / nb / "input"
    inp.mkdir(parents=True)
    for prev in nbs[:i]:                       # "Add Input -> Your Work -> Notebooks"
        (inp / prev.replace("_", "-")).symlink_to(root / prev / "working")
    r = subprocess.run([sys.executable, "run_test.py", f"src/{nb}.py"], env=dict(env, PTCSSL_WORK=str(work), PTCSSL_INPUT=str(inp)),
                       capture_output=True, text=True)
    print(nb, "->", "PASS" if r.returncode == 0 else "FAIL\n" + r.stdout[-3000:] + r.stderr[-2000:])
    assert r.returncode == 0
# negative control
sys.path.insert(0, str(root / "04_temporal_windowing/working/artifacts"))
import ptcssl_data
art3 = root / "03_preprocessing/working/artifacts"
V = np.load(art3 / "X_values.npy"); M = np.load(art3 / "X_mask.npy"); F = np.load(art3 / "X_locf.npy"); A = np.load(art3 / "X_age.npy")
starts = np.load(art3 / "patient_starts.npy"); lengths = np.load(art3 / "patient_lengths.npy")
import pandas as pd
rp = pd.read_pickle(art3 / "rows_meta.pkl")["pidx"].to_numpy()
class Leaky(ptcssl_data.WindowBuilder):
    def get(self, R):
        R = np.minimum(np.asarray(R) + 1, len(self.row_pidx) - 1)   # BUG: looks one hour ahead
        return super().get(R)
good, bad = ptcssl_data.WindowBuilder(V, M, F, A, starts, rp, 24), Leaky(V, M, F, A, starts, rp, 24)
def detects_leak(wb):
    Vc = V.copy(); wbc = type(wb)(Vc, M, F, A, starts, rp, 24)
    for r in range(0, len(V), 7):
        p = rp[r]; seg = slice(r + 1, starts[p] + lengths[p])
        bk = Vc[seg].copy(); Vc[seg] += 1000
        if not np.array_equal(np.nan_to_num(wb.get([r])["values"], nan=-1), np.nan_to_num(wbc.get([r])["values"], nan=-1)):
            return True
        Vc[seg] = bk
    return False
assert not detects_leak(good), "false positive"
assert detects_leak(bad), "leak not detected!"
print("Negative control PASS: causality test flags a one-hour look-ahead; correct builder passes")
