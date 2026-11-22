"""SYNTHETIC TEST FIXTURE - schema-identical fake PhysioNet 2019 folders for CODE TESTING ONLY.
Never used for any reported result. Values are random draws with plausible ranges."""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
        "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
        "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"]
COLS = VITALS + LABS + ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS", "SepsisLabel"]
CENTER = dict(HR=85, O2Sat=97, Temp=37, Resp=18, EtCO2=33, BaseExcess=0, FiO2=0.4, SaO2=96, AST=40, BUN=20,
              Alkalinephos=80, Calcium=8.5, Chloride=104, Creatinine=1.0, Glucose=130, Lactate=1.8,
              Magnesium=2.0, Phosphate=3.5, Potassium=4.1, TroponinI=0.05, PTT=32, WBC=10, Fibrinogen=300,
              Platelets=200)

def make(root, nA=120, nB=80, seed=0):
    rng = np.random.default_rng(seed)
    root = Path(root)
    pid = 0
    for hosp, n, base in (("A", nA, 0), ("B", nB, 100000)):
        d = root / "training" / f"training_set{hosp}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            pid += 1
            T = int(rng.integers(8, 70))
            X = np.full((T, len(COLS)), np.nan)
            septic = rng.random() < 0.3
            for j, c in enumerate(COLS[:34]):
                p_obs = 0.9 if c in VITALS and c != "EtCO2" else 0.08
                obs = rng.random(T) < p_obs
                if c in CENTER:
                    X[obs, j] = CENTER[c] * (1 + 0.15 * rng.standard_normal(obs.sum()))
            sbp = 120 + 15 * rng.standard_normal(T); dbp = 65 + 8 * rng.standard_normal(T)
            bp_obs = rng.random(T) < 0.85
            X[bp_obs, COLS.index("SBP")] = sbp[bp_obs]; X[bp_obs, COLS.index("DBP")] = dbp[bp_obs]
            X[bp_obs, COLS.index("MAP")] = (sbp[bp_obs] + 2 * dbp[bp_obs]) / 3 + 2 * rng.standard_normal(bp_obs.sum())
            abg = rng.random(T) < 0.1
            hco3 = 24 + 3 * rng.standard_normal(T); paco2 = 40 + 5 * rng.standard_normal(T)
            X[abg, COLS.index("HCO3")] = hco3[abg]; X[abg, COLS.index("PaCO2")] = paco2[abg]
            X[abg, COLS.index("pH")] = 6.1 + np.log10(hco3[abg] / (0.03 * paco2[abg])) + 0.01 * rng.standard_normal(abg.sum())
            cbc = rng.random(T) < 0.1
            hgb = 10 + 1.5 * rng.standard_normal(T)
            X[cbc, COLS.index("Hgb")] = hgb[cbc]; X[cbc, COLS.index("Hct")] = 3 * hgb[cbc] + rng.standard_normal(cbc.sum())
            bil = rng.random(T) < 0.05
            bt = np.abs(1 + 0.5 * rng.standard_normal(T))
            X[bil, COLS.index("Bilirubin_total")] = bt[bil]; X[bil, COLS.index("Bilirubin_direct")] = 0.3 * bt[bil]
            X[:, COLS.index("Age")] = round(float(rng.uniform(20, 90)), 2)
            X[:, COLS.index("Gender")] = float(rng.integers(0, 2))
            u = rng.integers(0, 3)
            X[:, COLS.index("Unit1")] = np.nan if u == 2 else float(u == 0)
            X[:, COLS.index("Unit2")] = np.nan if u == 2 else float(u == 1)
            X[:, COLS.index("HospAdmTime")] = -round(float(rng.uniform(0, 50)), 2)
            X[:, COLS.index("ICULOS")] = np.arange(1, T + 1)
            lab = np.zeros(T)
            if septic:
                start = 0 if rng.random() < 0.15 else int(rng.integers(1, T))
                lab[start:] = 1
            X[:, COLS.index("SepsisLabel")] = lab
            name = f"p{base + pid:06d}.psv"
            with open(d / name, "w") as f:
                f.write("|".join(COLS) + "\n")
                for row in X:
                    f.write("|".join("NaN" if np.isnan(v) else (f"{v:.2f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))) for v in row) + "\n")
    return root

if __name__ == "__main__":
    make(sys.argv[1])
    print("fixture written to", sys.argv[1])
