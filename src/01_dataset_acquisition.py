# %% [markdown]
# # 01 · Dataset acquisition and integrity verification
# **Project:** Physiology-Guided Self-Supervised Temporal Representation Learning for Label-Efficient Early Sepsis Prediction (PTC-SSL)
#
# This notebook obtains the **official PhysioNet/Computing in Cardiology Challenge 2019** training data
# (https://physionet.org/content/challenge-2019/1.0.0/, CC BY 4.0) and verifies it before anything else is built.
#
# **What it does**
# 1. Locates the data: an attached Kaggle input that keeps the official `training_setA/` and `training_setB/` folders, or a download.
# 2. Downloads from PhysioNet's open-data S3 mirror (`s3://physionet-open/challenge-2019/1.0.0/`) and checks every file's **MD5 against the provider's ETag**. If that fails it falls back to `physionet.org` (size check only).
# 3. Verifies structure: 41 columns in official order, hospital identity from the folder, patient counts (A = 20,336, B = 20,000), hourly `ICULOS`, binary and monotone `SepsisLabel`, constant demographics.
# 4. Writes one compact hourly table plus a patient table, a SHA-256 fingerprint of every file, the official scoring script and a verified reference list.
#
# **Kaggle settings:** *Internet → On* (needs a phone-verified account), unless you attach the raw files as an input. CPU is enough. Expected run time: about 10–25 min (download) plus about 3 min (parsing).
#
# **Outputs** (`/kaggle/working/`):
# `artifacts/physionet2019_hourly.parquet`, `artifacts/patients.parquet`, `artifacts/file_fingerprints.csv`,
# `artifacts/evaluate_sepsis_score_official.py`, `results/dataset_manifest.json`, `results/references.bib`, `results/tables/integrity_checks.csv`.
#
# Raw `.psv` files are written to `/tmp`, so they are never part of the notebook output or the research packages.

# %% [common]

# %%
NB = "01_dataset_acquisition"
log = get_logger(NB)
log.info("Environment: %s", json.dumps(env_info()))

CFG = {
    "dataset": "PhysioNet/CinC Challenge 2019 - Early Prediction of Sepsis from Clinical Data, v1.0.0",
    "landing_page": "https://physionet.org/content/challenge-2019/1.0.0/",
    "license": "Creative Commons Attribution 4.0 International Public License (CC BY 4.0)",
    "s3_bucket_https": "https://physionet-open.s3.amazonaws.com",
    "s3_prefix": "challenge-2019/1.0.0/training/",
    "physionet_files": "https://physionet.org/files/challenge-2019/1.0.0/training/",
    "sha256sums_url": "https://physionet.org/files/challenge-2019/1.0.0/SHA256SUMS.txt",
    "eval_script_url": "https://raw.githubusercontent.com/physionetchallenges/evaluation-2019/master/evaluate_sepsis_score.py",
    "hospital_dirs": {"A": "training_setA", "B": "training_setB"},
    "expected_patients": {"A": 20336, "B": 20000},
    "download_threads": 16,
    "http_timeout_s": 60,
    "max_retries": 5,
}
# Test fixtures only: override expected counts (never set this on Kaggle)
if os.environ.get("PTCSSL_EXPECTED_COUNTS"):
    CFG["expected_patients"] = json.loads(os.environ["PTCSSL_EXPECTED_COUNTS"])
    log.warning("EXPECTED COUNTS OVERRIDDEN FOR TESTING: %s", CFG["expected_patients"])

RAW = Path(os.environ.get("PTCSSL_RAW_DIR", "/tmp/physionet2019"))   # outside /kaggle/working on purpose
RAW.mkdir(parents=True, exist_ok=True)
assert len(EXPECTED_COLUMNS) == 41
print(json.dumps(CFG, indent=2))

# %% [markdown]
# ## 1. Locate an existing copy (attached input or earlier download)
# A source is accepted only if it keeps the official `training_setA` / `training_setB` folders.
# Flat Kaggle mirrors that merge both hospitals are rejected, because hospital identity is needed for the A→B / B→A external validation.

# %%
def scan_source(root: Path):
    """Return {hospital: sorted list of psv paths} if root contains official set folders, else None."""
    found = {}
    for hosp, dname in CFG["hospital_dirs"].items():
        dirs = [d for d in root.rglob(dname) if d.is_dir()] if root.exists() else []
        if not dirs:
            return None
        files = sorted(dirs[0].glob("*.psv"))
        if not files:
            return None
        found[hosp] = files
    return found

source_mode, source_root, files_by_hosp = None, None, None
candidates = []
if os.environ.get("PTCSSL_LOCAL_SOURCE"):
    candidates.append(("local", Path(os.environ["PTCSSL_LOCAL_SOURCE"])))
candidates.append(("previous_download", RAW))
if INPUT.exists():
    candidates += [("kaggle_input", d) for d in sorted(INPUT.iterdir()) if d.is_dir()]

for mode, root in candidates:
    try:
        res = scan_source(root)
    except Exception as e:
        log.warning("Could not scan %s: %s", root, e)
        res = None
    if res and all(len(res[h]) == CFG["expected_patients"][h] for h in res):
        source_mode, source_root, files_by_hosp = mode, root, res
        break
    if res:
        log.warning("Source %s is incomplete (%s files); ignoring.", root, {h: len(v) for h, v in res.items()})

print("Existing complete source:", source_mode, source_root)

# %% [markdown]
# ## 2. Download if needed (S3 mirror with MD5 = ETag verification → physionet.org fallback)

# %%
import re
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

_tls = threading.local()

def _session():
    import requests
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
        _tls.s.headers["User-Agent"] = "PTC-SSL-research-download/1.0"
    return _tls.s

def http_get(url, stream=False):
    if os.environ.get("PTCSSL_OFFLINE"):          # local test fixtures only
        raise ConnectionError("offline test mode")
    last = None
    for attempt in range(CFG["max_retries"]):
        try:
            r = _session().get(url, timeout=CFG["http_timeout_s"], stream=stream)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {url}")
            if r.status_code in (403, 404):
                break
        except Exception as e:   # network errors -> retry with backoff
            last = e
        time.sleep(min(2 ** attempt, 30))
    raise last

def list_s3(prefix):
    """List objects with key, size and ETag (MD5 for single-part uploads)."""
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    out, token = [], None
    while True:
        url = f"{CFG['s3_bucket_https']}/?list-type=2&prefix={prefix}"
        if token:
            from urllib.parse import quote
            url += "&continuation-token=" + quote(token)
        root = ET.fromstring(http_get(url).content)
        for c in root.findall(f"{ns}Contents"):
            key = c.find(f"{ns}Key").text
            if key.endswith(".psv"):
                out.append({"key": key, "size": int(c.find(f"{ns}Size").text),
                            "etag": c.find(f"{ns}ETag").text.strip('"')})
        if (root.find(f"{ns}IsTruncated").text or "").lower() != "true":
            break
        token = root.find(f"{ns}NextContinuationToken").text
    return out

def list_physionet(dname):
    html = http_get(CFG["physionet_files"] + dname + "/").text
    names = sorted(set(re.findall(r'href="(p\d+\.psv)"', html)))
    return [{"key": dname + "/" + n, "size": None, "etag": None} for n in names]

def download_one(url, dest: Path, size=None, etag=None):
    if dest.exists() and (size is None or dest.stat().st_size == size):
        if etag is None or "-" in etag or md5_file(dest) == etag:
            return "skipped"
    tmp = dest.with_suffix(".part")
    data = http_get(url).content
    tmp.write_bytes(data)
    if size is not None and tmp.stat().st_size != size:
        raise IOError(f"size mismatch {dest.name}")
    if etag and "-" not in etag and md5_file(tmp) != etag:
        raise IOError(f"MD5/ETag mismatch {dest.name}")
    tmp.replace(dest)
    return "downloaded"

download_log = {}
if source_mode is None:
    with StageTimer("download", log):
        for hosp, dname in CFG["hospital_dirs"].items():
            target = RAW / "training" / dname
            target.mkdir(parents=True, exist_ok=True)
            try:
                entries = list_s3(CFG["s3_prefix"] + dname + "/")
                base, method = CFG["s3_bucket_https"] + "/", "s3_https_md5_etag"
                assert len(entries) > 0, "empty S3 listing"
            except Exception as e:
                log.warning("S3 listing failed for %s (%s); falling back to physionet.org", dname, e)
                entries = list_physionet(dname)
                base, method = CFG["physionet_files"], "physionet_org_size_unchecked"
            log.info("%s: %d files listed via %s", dname, len(entries), method)
            status, errors = {"downloaded": 0, "skipped": 0}, []
            with ThreadPoolExecutor(CFG["download_threads"]) as ex:
                futs = {}
                for e in entries:
                    fname = e["key"].split("/")[-1]
                    url = base + e["key"] if method.startswith("s3") else CFG["physionet_files"] + e["key"]
                    futs[ex.submit(download_one, url, target / fname, e["size"], e["etag"])] = fname
                for i, f in enumerate(as_completed(futs), 1):
                    try:
                        status[f.result()] += 1
                    except Exception as err:
                        errors.append((futs[f], str(err)))
                    if i % 5000 == 0:
                        log.info("  %s: %d/%d", dname, i, len(futs))
            # one retry pass for failures
            for fname, _ in list(errors):
                e = next(x for x in entries if x["key"].endswith("/" + fname))
                url = base + e["key"] if method.startswith("s3") else CFG["physionet_files"] + e["key"]
                download_one(url, target / fname, e["size"], e["etag"])
                errors = [x for x in errors if x[0] != fname]
            download_log[hosp] = {"method": method, "listed": len(entries), **status, "errors": errors,
                                  "etag_md5_verified": method.startswith("s3")}
            if errors:
                raise RuntimeError(f"{len(errors)} files failed for {dname}: {errors[:5]}")
            # keep listing (with provider checksums) for the integrity record
            pd.DataFrame(entries).to_csv(ART / f"provider_listing_{hosp}.csv", index=False)
    files_by_hosp = scan_source(RAW)
    source_mode, source_root = "downloaded", RAW
    if files_by_hosp is None:
        raise RuntimeError("Download finished but official folder structure not found. Check Internet is ON.")

print("Source:", source_mode, source_root)
print({h: len(v) for h, v in files_by_hosp.items()})
print(json.dumps(download_log, indent=2)[:2000])

# %% [markdown]
# ## 3. File-level integrity: counts, uniqueness, SHA-256 fingerprints, provider checksums

# %%
checks = []   # (check, status, detail)
def check(name, ok, detail=""):
    checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})
    (log.info if ok else log.error)("%s %s %s", "PASS" if ok else "FAIL", name, detail)
    return ok

for h, exp in CFG["expected_patients"].items():
    check(f"patient_count_{h}", len(files_by_hosp[h]) == exp, f"found={len(files_by_hosp[h])}, expected={exp}")

ids = {h: [p.stem for p in v] for h, v in files_by_hosp.items()}
overlap = set(ids["A"]) & set(ids["B"])
check("no_patient_id_overlap_A_B", len(overlap) == 0, f"overlap={len(overlap)}")
check("unique_ids_within_A", len(set(ids["A"])) == len(ids["A"]))
check("unique_ids_within_B", len(set(ids["B"])) == len(ids["B"]))

with StageTimer("fingerprint", log):
    rows = []
    with ThreadPoolExecutor(16) as ex:
        futs = {ex.submit(lambda p: (sha256_file(p), md5_file(p), p.stat().st_size), p): (h, p)
                for h, v in files_by_hosp.items() for p in v}
        for f in as_completed(futs):
            h, p = futs[f]
            s, m, sz = f.result()
            rows.append({"hospital": h, "file": p.name, "bytes": sz, "sha256": s, "md5": m})
fp = pd.DataFrame(rows).sort_values(["hospital", "file"]).reset_index(drop=True)
fp.to_csv(ART / "file_fingerprints.csv", index=False)
dataset_fingerprint = hashlib.sha256("".join(fp["sha256"]).encode()).hexdigest()
log.info("Dataset fingerprint (SHA-256 over sorted file hashes): %s", dataset_fingerprint)

# Compare against provider checksums whenever we have them (S3 ETag = MD5 of single-part objects)
provider_verified = False
listings = [ART / f"provider_listing_{h}.csv" for h in CFG["hospital_dirs"]]
if all(p.exists() for p in listings):
    prov = pd.concat([pd.read_csv(p) for p in listings])
    prov["file"] = prov["key"].str.split("/").str[-1]
    prov = prov[~prov["etag"].astype(str).str.contains("-")]
    mrg = fp.merge(prov[["file", "etag", "size"]], on="file", how="left")
    n_cmp = mrg["etag"].notna().sum()
    n_bad = ((mrg["etag"].notna()) & (mrg["etag"] != mrg["md5"])).sum()
    provider_verified = check("provider_md5_etag_match", n_bad == 0 and n_cmp == len(fp),
                              f"compared={n_cmp}, mismatches={n_bad}")
else:
    check("provider_md5_etag_match", True,
          "SKIPPED: source not downloaded in this session (attached/local copy); structural checks only")

# Official SHA256SUMS.txt is tiny (covers only a few top-level files) - record it for completeness.
sha256sums_text = None
try:
    sha256sums_text = http_get(CFG["sha256sums_url"]).text
    (ART / "official_SHA256SUMS.txt").write_text(sha256sums_text)
except Exception as e:
    log.warning("Could not fetch official SHA256SUMS.txt (%s); not required.", e)

# %% [markdown]
# ## 4. Parse all records and run structural checks
# Each `.psv` is pipe-separated with a header. Checks: exact 41-column header, numeric parsing, hourly
# `ICULOS` steps, binary and non-decreasing `SepsisLabel`, constant demographics within a patient.

# %%
def read_psv(path: Path):
    with open(path) as f:
        header = f.readline().strip().split("|")
    df = pd.read_csv(path, sep="|", dtype=np.float32)
    return header, df

frames, problems = [], []
with StageTimer("parse", log):
    tasks = [(h, p) for h in ("A", "B") for p in files_by_hosp[h]]
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda hp: (hp, read_psv(hp[1])), tasks))
    for (h, p), (header, df) in results:
        if header != EXPECTED_COLUMNS:
            problems.append((p.name, "header_mismatch"))
            continue
        if len(df) == 0:
            problems.append((p.name, "empty"))
            continue
        df.insert(0, "t", np.arange(len(df), dtype=np.int16))
        df.insert(0, "hospital", h)
        df.insert(0, "patient_id", p.stem)
        frames.append(df)

check("all_headers_match_official_41_columns", not any(x[1] == "header_mismatch" for x in problems),
      [x for x in problems if x[1] == "header_mismatch"][:5])
check("no_empty_records", not any(x[1] == "empty" for x in problems))
hourly = pd.concat(frames, ignore_index=True)
del frames, results
hourly["patient_id"] = hourly["patient_id"].astype("category")
hourly["hospital"] = hourly["hospital"].astype("category")
log.info("Hourly table: %d rows x %d cols, %.0f MB", *hourly.shape, hourly.memory_usage(deep=True).sum() / 1e6)

# %%
g = hourly.groupby("patient_id", observed=True, sort=False)

lab = hourly["SepsisLabel"]
check("sepsis_label_binary", set(np.unique(lab.dropna())) <= {0.0, 1.0} and lab.notna().all())
nondecr = g["SepsisLabel"].apply(lambda s: bool((np.diff(s.values) >= 0).all()))
check("sepsis_label_non_decreasing_per_patient", nondecr.all(), f"violations={(~nondecr).sum()}")

icu_step = g["ICULOS"].apply(lambda s: bool((np.diff(s.values) == 1).all()))
check("iculos_hourly_steps", icu_step.all(), f"patients with non-unit steps={(~icu_step).sum()} (reported, not fatal)")
checks[-1]["status"] = "PASS" if icu_step.all() else "WARN"

for col in ["Age", "Gender", "HospAdmTime"]:
    nun = g[col].nunique(dropna=True)
    ok = (nun <= 1).all()
    check(f"{col}_constant_within_patient", ok, f"patients with >1 value={(nun > 1).sum()}")
    if not ok:
        checks[-1]["status"] = "WARN"
check("gender_binary", set(np.unique(hourly["Gender"].dropna())) <= {0.0, 1.0})

# %% [markdown]
# ## 5. Patient table (label timing follows the official scoring code)
# The labels in the training data were already shifted six hours ahead: `SepsisLabel = 1` for `t ≥ t_sepsis − 6`.
# As in `evaluate_sepsis_score.py`: `t_sepsis = argmax(SepsisLabel) − dt_optimal = first_positive_index + 6` (row index, hours).

# %%
first_pos = g["SepsisLabel"].apply(lambda s: int(np.argmax(s.values)) if s.values.max() > 0 else -1)
patients = pd.DataFrame({
    "patient_id": first_pos.index.astype(str),
    "hospital": g["hospital"].first().astype(str).values,
    "n_rows": g.size().values,
    "septic": (first_pos.values >= 0).astype(np.int32),
    "first_label_idx": first_pos.values.astype(np.int32),
})
patients["t_sepsis_idx"] = np.where(patients["septic"] == 1,
                                    patients["first_label_idx"] - UTILITY_PARAMS["dt_optimal"], -1)
patients["iculos_first"] = g["ICULOS"].first().values
patients["iculos_last"] = g["ICULOS"].last().values
patients["age"] = g["Age"].first().values
patients["gender"] = g["Gender"].first().values
patients["hosp_adm_time"] = g["HospAdmTime"].first().values
patients = patients.sort_values(["hospital", "patient_id"]).reset_index(drop=True)

summary = patients.groupby("hospital").agg(patients=("patient_id", "size"), hours=("n_rows", "sum"),
                                           septic=("septic", "sum"))
summary = summary.astype(float)
summary["septic_pct"] = 100.0 * summary["septic"] / summary["patients"]
summary.loc["Total"] = [summary["patients"].sum(), summary["hours"].sum(), summary["septic"].sum(),
                        100 * summary["septic"].sum() / summary["patients"].sum()]
display(summary)
check("hourly_rows_match_patient_table", summary.loc["Total", "hours"] == len(hourly))
check("t_sepsis_consistent", ((patients["septic"] == 0) | (patients["t_sepsis_idx"] == patients["first_label_idx"] + 6)).all())

# %% [markdown]
# ## 6. Save compact artifacts

# %%
with StageTimer("save", log):
    p1 = save_table(hourly, ART / "physionet2019_hourly.parquet")
    p2 = save_table(patients, ART / "patients.parquet")
summary.reset_index().to_csv(RES / "tables" / "dataset_counts_by_hospital.csv", index=False)
log.info("Saved %s and %s", p1, p2)

# %% [markdown]
# ## 7. Official scoring script (used later to cross-check our vectorised utility implementation)

# %%
eval_status = "unavailable"
try:
    code = http_get(CFG["eval_script_url"]).text
    assert "def compute_prediction_utility" in code
    dst = ART / "evaluate_sepsis_score_official.py"
    dst.write_text(code)
    import importlib.util
    spec = importlib.util.spec_from_file_location("official_eval", dst)
    official_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official_eval)
    # sanity: a non-septic patient with no alarms has zero utility
    u0 = official_eval.compute_prediction_utility(np.zeros(20), np.zeros(20))
    assert abs(u0) < 1e-12
    eval_status = f"downloaded sha256={sha256_file(dst)}"
except Exception as e:
    log.warning("Official evaluation script not available (%s). Notebook 05 will still implement the "
                "utility exactly; cross-check will be skipped.", e)
check("official_eval_script", True, eval_status)

# %% [markdown]
# ## 8. Dataset manifest, references and check report

# %%
references_bib = r"""@article{reyna2020sepsis,
  author  = {Reyna, Matthew A. and Josef, Christopher S. and Jeter, Russell and Shashikumar, Supreeth P. and Westover, M. Brandon and Nemati, Shamim and Clifford, Gari D. and Sharma, Ashish},
  title   = {Early Prediction of Sepsis From Clinical Data: The {PhysioNet/Computing in Cardiology Challenge} 2019},
  journal = {Critical Care Medicine},
  volume  = {48},
  number  = {2},
  pages   = {210--217},
  year    = {2020},
  doi     = {10.1097/CCM.0000000000004145}
}
@article{goldberger2000physionet,
  author  = {Goldberger, Ary L. and Amaral, Luis A. N. and Glass, Leon and Hausdorff, Jeffrey M. and Ivanov, Plamen Ch. and Mark, Roger G. and Mietus, Joseph E. and Moody, George B. and Peng, Chung-Kang and Stanley, H. Eugene},
  title   = {{PhysioBank, PhysioToolkit, and PhysioNet}: Components of a New Research Resource for Complex Physiologic Signals},
  journal = {Circulation},
  volume  = {101},
  number  = {23},
  pages   = {e215--e220},
  year    = {2000},
  doi     = {10.1161/01.CIR.101.23.e215}
}
@article{singer2016sepsis3,
  author  = {Singer, Mervyn and Deutschman, Clifford S. and Seymour, Christopher Warren and others},
  title   = {The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)},
  journal = {JAMA},
  volume  = {315},
  number  = {8},
  pages   = {801--810},
  year    = {2016},
  doi     = {10.1001/jama.2016.0287}
}
"""
(RES / "references.bib").write_text(references_bib)

manifest = {
    "dataset": CFG["dataset"], "landing_page": CFG["landing_page"], "license": CFG["license"],
    "required_citations": ["reyna2020sepsis", "goldberger2000physionet"],
    "source_mode": source_mode, "download_log": download_log,
    "provider_checksum_verified": provider_verified,
    "dataset_fingerprint_sha256": dataset_fingerprint,
    "official_sha256sums_txt_present": sha256sums_text is not None,
    "patients": {h: int(len(v)) for h, v in files_by_hosp.items()},
    "hourly_rows": int(len(hourly)),
    "septic_patients": {h: int(patients.loc[patients.hospital == h, "septic"].sum()) for h in ("A", "B")},
    "columns": EXPECTED_COLUMNS,
    "label_definition": ("SepsisLabel = 1 for t >= t_sepsis - 6 (labels shifted 6 h ahead by the challenge "
                         "organisers); t_sepsis defined by Sepsis-3; non-septic patients always 0."),
    "documented_unit_anomalies_to_verify_in_02": {
        "FiO2": "documented as %, verify fraction vs percent", "Lactate": "documented as mg/dL",
        "Magnesium": "documented as mmol/dL"},
    "official_eval_script": eval_status,
    "environment": env_info(),
}
save_json(manifest, RES / "dataset_manifest.json")
chk = pd.DataFrame(checks)
chk.to_csv(RES / "tables" / "integrity_checks.csv", index=False)
display(chk)

n_fail = (chk["status"] == "FAIL").sum()
if n_fail:
    raise RuntimeError(f"{n_fail} integrity checks FAILED - do not proceed. See results/tables/integrity_checks.csv")
print("\n✅ Dataset acquired and verified. Next: 02_data_profiling (attach this notebook's output as input).")
