# ===== PTC-SSL common setup (identical in every notebook; do not edit per notebook) =====
import os, sys, json, time, math, glob, shutil, random, hashlib, logging, platform, warnings
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Kaggle paths (overridable through environment variables for local testing only)
WORK = Path(os.environ.get("PTCSSL_WORK", "/kaggle/working"))
INPUT = Path(os.environ.get("PTCSSL_INPUT", "/kaggle/input"))
ART = WORK / "artifacts"   # data artifacts consumed by later notebooks (may contain patient-level data)
RES = WORK / "results"     # metrics / tables / figures / configs (no raw patient data) -> paper package
LOGS = WORK / "logs"
for _d in (ART, RES, LOGS, RES / "figures", RES / "tables", RES / "configs"):
    _d.mkdir(parents=True, exist_ok=True)

GLOBAL_SEED = 2026

def set_seed(seed: int = GLOBAL_SEED):
    """Seed python, numpy and (if present) torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

set_seed()

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(LOGS / f"{name}.log", mode="a")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.propagate = False
    return logger

try:
    display  # noqa: F821  (exists inside Jupyter/Kaggle)
except NameError:
    def display(x):
        print(x)

def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def md5_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_table(df: pd.DataFrame, path):
    """Save as parquet (Kaggle has pyarrow); fall back to pickle if parquet is unavailable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        alt = path.with_suffix(".pkl")
        df.to_pickle(alt)
        return alt

def load_table(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet" and path.exists():
        return pd.read_parquet(path)
    alt = path.with_suffix(".pkl")
    if alt.exists():
        return pd.read_pickle(alt)
    if path.exists():
        return pd.read_pickle(path)
    raise FileNotFoundError(path)

_ARTIFACT_CACHE = {}

def find_artifact(name: str, producer: str) -> Path:
    """Locate an artifact produced by an earlier notebook.
    Search order: this session's /kaggle/working/artifacts, then any attached input (/kaggle/input/**).
    Table artifacts may exist as .parquet or .pkl."""
    if name in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[name]
    stem, suffix = os.path.splitext(name)
    names = [name] + ([stem + ".pkl"] if suffix == ".parquet" else [])
    for n in names:
        p = ART / n
        if p.exists():
            _ARTIFACT_CACHE[name] = p
            return p
    hits = []
    if INPUT.exists():
        for dirpath, _, files in os.walk(INPUT, followlinks=True):
            if Path(dirpath).name == "artifacts":          # never pick up copies under results/
                hits += [Path(dirpath) / n for n in names if n in files]
    if hits:
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if len(hits) > 1:
            print(f"[find_artifact] {len(hits)} copies of {name} found; using newest: {hits[0]}")
        _ARTIFACT_CACHE[name] = hits[0]
        return hits[0]
    raise FileNotFoundError(
        f"Artifact '{name}' not found. Run notebook {producer} first, then attach its output to this "
        f"notebook (Kaggle: Add Input -> Your Work -> Notebooks -> {producer})."
    )

def env_info() -> dict:
    info = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for mod in ("scipy", "sklearn", "matplotlib", "torch", "lightgbm"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            info[mod] = "not installed"
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info

class StageTimer:
    """Context manager that logs wall-clock time of a stage (also used for the compute-cost table)."""
    def __init__(self, name, logger=None):
        self.name, self.logger = name, logger
    def __enter__(self):
        self.t0 = time.time()
        return self
    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0
        msg = f"[time] {self.name}: {self.seconds:.1f}s"
        (self.logger.info if self.logger else print)(msg)

# PhysioNet 2019 schema (verified against the official data description, physionet.org/content/challenge-2019/1.0.0/)
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
        "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
        "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"]
DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
DYNAMIC_VARS = VITALS + LABS                       # 34 time-varying clinical variables
EXPECTED_COLUMNS = DYNAMIC_VARS + DEMOGRAPHICS + ["SepsisLabel"]   # 41 columns, official order

# Official utility parameters (physionetchallenges/evaluation-2019, evaluate_sepsis_score.py)
UTILITY_PARAMS = dict(dt_early=-12, dt_optimal=-6, dt_late=3, max_u_tp=1.0, min_u_fn=-2.0, u_fp=-0.05, u_tn=0.0)
print(f"[setup] WORK={WORK}  INPUT={INPUT}  seed={GLOBAL_SEED}")
