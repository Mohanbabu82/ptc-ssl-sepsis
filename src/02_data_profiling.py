# %% [markdown]
# # 02 · Data profiling, unit verification and validation of physiological constraints
# **Input:** output of `01_dataset_acquisition` (attach it via *Add Input → Your Work → Notebooks*).
#
# **Purpose.** Characterise the cohort before any modelling decision, and test (rather than assume) the documented units
# and the physiological relations that PTC-SSL will use as constraints.
#
# 1. Cohort description by hospital (Table 1 of the paper).
# 2. Missingness and measurement frequency per variable.
# 3. **Unit verification**: documented unit vs observed distribution. The official description lists FiO2 as %, Lactate as mg/dL and Magnesium as mmol/dL, which needs checking against the data.
# 4. Plausibility (QC) ranges in the verified units, with the counts they would remove. Saved for `03`.
# 5. **Constraint validation**: each candidate physiological relation is checked on same-hour co-measured raw values in **each hospital separately**, against acceptance criteria fixed *before* looking at the data (below). No outcome labels are used.
# 6. Evidence for choosing the window length.
#
# **Pre-specified acceptance rule for a constraint:** at least `MIN_CO` co-measured rows in *both* hospitals, and
# (equality relations) at least 70% of rows within tolerance, or (inequality relations) at most 5% violations, in *both* hospitals.
#
# Runs on CPU in about 2–4 min.

# %% [common]

# %%
NB = "02_data_profiling"
log = get_logger(NB)
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

hourly = load_table(find_artifact("physionet2019_hourly.parquet", "01_dataset_acquisition"))
patients = load_table(find_artifact("patients.parquet", "01_dataset_acquisition"))
assert list(hourly.columns[3:]) == EXPECTED_COLUMNS, "unexpected schema - rerun 01"
assert hourly["patient_id"].nunique() == len(patients)
MIN_CO = int(os.environ.get("PTCSSL_MIN_CO", 500))   # minimum co-measured rows per hospital
HOSPS = ["A", "B"]
log.info("Loaded %d hourly rows, %d patients", len(hourly), len(patients))

def savefig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(RES / "figures" / f"{name}.{ext}", bbox_inches="tight")
    plt.show() if matplotlib.get_backend().lower() != "agg" else None
    plt.close(fig)

# %% [markdown]
# ## 1. Cohort description (Table 1)

# %%
def med_iqr(x):
    x = pd.Series(x).dropna()
    return f"{x.median():.1f} [{x.quantile(.25):.1f}–{x.quantile(.75):.1f}]" if len(x) else "NA"

unit_first = hourly.groupby("patient_id", observed=True)[["Unit1", "Unit2"]].first()
patients = patients.merge(unit_first, left_on="patient_id", right_index=True, how="left")
rows = []
for h in HOSPS + ["All"]:
    p = patients if h == "All" else patients[patients.hospital == h]
    s = p[p.septic == 1]
    rows.append({
        "cohort": h, "patients": len(p), "patient_hours": int(p.n_rows.sum()),
        "ICU hours per record, median [IQR]": med_iqr(p.n_rows),
        "age, median [IQR]": med_iqr(p.age), "male, %": f"{100 * p.gender.mean():.1f}",
        "MICU / SICU / unit missing, %": f"{100*(p.Unit1==1).mean():.1f} / {100*(p.Unit2==1).mean():.1f} / {100*p.Unit1.isna().mean():.1f}",
        "septic, n (%)": f"{len(s)} ({100 * len(s) / len(p):.2f})",
        "t_sepsis (row h), median [IQR]": med_iqr(s.t_sepsis_idx),
        "septic with label at first row, n": int((s.first_label_idx == 0).sum()),
        "septic with first label <= 6 h, n": int((s.first_label_idx <= 6).sum()),
    })
table1 = pd.DataFrame(rows).set_index("cohort").T
table1.to_csv(RES / "tables" / "table1_cohort.csv")
display(table1)

# %% [markdown]
# ## 2. Missingness and measurement frequency

# %%
obs = hourly[DYNAMIC_VARS].notna()
miss_rows = []
for h in HOSPS:
    sel = (hourly.hospital == h).values
    o = obs[sel]
    pid = hourly.loc[sel, "patient_id"]
    ever = o.groupby(pid.values).any().mean()
    for v in DYNAMIC_VARS:
        # median gap between consecutive observations (hours), pooled over patients
        tv = hourly.loc[sel & obs[v].values, ["patient_id", "t"]]
        gaps = tv.groupby("patient_id", observed=True)["t"].diff().dropna()
        miss_rows.append({"hospital": h, "variable": v, "group": "vital" if v in VITALS else "lab",
                          "pct_hours_observed": 100 * o[v].mean(), "pct_patients_ever_observed": 100 * ever[v],
                          "median_gap_h": float(gaps.median()) if len(gaps) else np.nan})
miss = pd.DataFrame(miss_rows)
miss.to_csv(RES / "tables" / "missingness_by_variable.csv", index=False)
display(miss.pivot(index="variable", columns="hospital", values="pct_hours_observed").round(2).loc[DYNAMIC_VARS])

fig, ax = plt.subplots(figsize=(7.2, 3.2))
pv = miss.pivot(index="variable", columns="hospital", values="pct_hours_observed").loc[DYNAMIC_VARS]
x = np.arange(len(pv))
ax.bar(x - 0.2, pv["A"], 0.4, label="Hospital A")
ax.bar(x + 0.2, pv["B"], 0.4, label="Hospital B")
ax.set_xticks(x, pv.index, rotation=90)
ax.set_ylabel("Hours with a measurement (%)")
ax.set_yscale("log")
ax.legend(frameon=False)
savefig(fig, "fig_missingness_by_variable")

# %% [markdown]
# ## 3. Unit verification
# For each variable the observed median is compared with the reference median window expected under each candidate unit.
# The windows are broad adult ICU ranges, chosen only to tell units apart (a factor of 10 or more between candidates), not to define normality.

# %%
DOC_UNITS = {"HR": "bpm", "O2Sat": "%", "Temp": "°C", "SBP": "mmHg", "MAP": "mmHg", "DBP": "mmHg",
             "Resp": "breaths/min", "EtCO2": "mmHg", "BaseExcess": "mmol/L", "HCO3": "mmol/L", "FiO2": "%",
             "pH": "-", "PaCO2": "mmHg", "SaO2": "%", "AST": "IU/L", "BUN": "mg/dL", "Alkalinephos": "IU/L",
             "Calcium": "mg/dL", "Chloride": "mmol/L", "Creatinine": "mg/dL", "Bilirubin_direct": "mg/dL",
             "Glucose": "mg/dL", "Lactate": "mg/dL", "Magnesium": "mmol/dL", "Phosphate": "mg/dL",
             "Potassium": "mmol/L", "Bilirubin_total": "mg/dL", "TroponinI": "ng/mL", "Hct": "%", "Hgb": "g/dL",
             "PTT": "s", "WBC": "10^3/µL", "Fibrinogen": "mg/dL", "Platelets": "10^3/µL"}
# candidate units -> (low, high) window for the cohort MEDIAN
CANDIDATES = {
    "FiO2": {"fraction": (0.21, 1.0), "%": (21, 100)},
    "Lactate": {"mmol/L": (0.5, 6), "mg/dL": (4.5, 54)},
    "Magnesium": {"mg/dL": (1.4, 3.0), "mmol/L": (0.55, 1.25), "mmol/dL": (0.055, 0.125)},
    "Calcium": {"mg/dL (total)": (6, 12), "mmol/L (ionised)": (0.8, 1.6)},
    "Temp": {"°C": (34, 40), "°F": (93, 104)},
    "Hct": {"%": (18, 55), "fraction": (0.18, 0.55)},
    "Hgb": {"g/dL": (6, 18), "g/L": (60, 180), "mmol/L": (3.7, 11.2)},
    "Glucose": {"mg/dL": (60, 300), "mmol/L": (3.3, 16.7)},
    "Creatinine": {"mg/dL": (0.3, 6), "µmol/L": (27, 530)},
    "BUN": {"mg/dL": (5, 100), "mmol/L urea": (1.8, 36)},
    "Bilirubin_total": {"mg/dL": (0.1, 10), "µmol/L": (1.7, 171)},
    "PaCO2": {"mmHg": (25, 70), "kPa": (3.3, 9.3)},
    "SaO2": {"%": (80, 100), "fraction": (0.8, 1.0)},
    "O2Sat": {"%": (80, 100), "fraction": (0.8, 1.0)},
}
unit_rows = []
for v in DYNAMIC_VARS:
    for h in HOSPS:
        x = hourly.loc[hourly.hospital == h, v].dropna()
        r = {"variable": v, "hospital": h, "documented_unit": DOC_UNITS[v], "n": len(x),
             "p01": x.quantile(.01) if len(x) else np.nan, "median": x.median() if len(x) else np.nan,
             "p99": x.quantile(.99) if len(x) else np.nan}
        if v in CANDIDATES and len(x):
            fits = [u for u, (lo, hi) in CANDIDATES[v].items() if lo <= r["median"] <= hi]
            doc_fits = [u for u in fits if u.split(" ")[0] == DOC_UNITS[v]]
            if len(fits) == 1:
                r["inferred_unit"] = fits[0]
            elif len(fits) > 1 and len(doc_fits) == 1:   # overlapping windows: data consistent with documentation
                r["inferred_unit"] = doc_fits[0]
            else:
                r["inferred_unit"] = ("ambiguous: " + ",".join(fits)) if fits else "none"
        else:
            r["inferred_unit"] = DOC_UNITS[v] + " (documented; not tested)"
        if v == "FiO2" and len(x):
            r["pct_values_gt_1"] = 100 * (x > 1).mean()
        unit_rows.append(r)
units = pd.DataFrame(unit_rows)
units.to_csv(RES / "tables" / "unit_verification.csv", index=False)
display(units[units.variable.isin(CANDIDATES)].round(3))

def inferred(v):
    u = units[(units.variable == v)]["inferred_unit"].unique()
    if len(u) != 1:
        log.warning("Unit for %s differs between hospitals or is ambiguous: %s", v, u)
    return u[0]

VERIFIED_UNITS = {v: inferred(v) for v in CANDIDATES}
for v, u in VERIFIED_UNITS.items():
    flag = "" if u.split(" ")[0] in DOC_UNITS[v] or DOC_UNITS[v] in u else "   <-- differs from documentation"
    print(f"{v:16s} documented={DOC_UNITS[v]:9s} inferred={u}{flag}")

# %% [markdown]
# ## 4. Plausibility (QC) ranges in the verified units
# Values outside these ranges are physiologically implausible (device or entry errors) and are set to missing in `03`.
# FiO2 recorded as a percentage (1 < value ≤ 100) is converted to a fraction **only if** the dominant verified unit is a fraction.

# %%
QC = {"HR": (20, 300), "O2Sat": (50, 100), "Temp": (25, 45), "SBP": (40, 300), "MAP": (20, 250), "DBP": (10, 200),
      "Resp": (3, 80), "EtCO2": (0, 100), "BaseExcess": (-40, 40), "HCO3": (2, 60), "FiO2": (0.21, 1.0),
      "pH": (6.5, 8.0), "PaCO2": (5, 150), "SaO2": (30, 100), "AST": (1, 20000), "BUN": (1, 300),
      "Alkalinephos": (1, 5000), "Calcium": (2, 20), "Chloride": (50, 160), "Creatinine": (0.1, 30),
      "Bilirubin_direct": (0, 50), "Glucose": (10, 2000), "Lactate": (0.1, 40), "Magnesium": (0.3, 10),
      "Phosphate": (0.2, 20), "Potassium": (1, 12), "Bilirubin_total": (0, 80), "TroponinI": (0, 500),
      "Hct": (5, 75), "Hgb": (1, 25), "PTT": (10, 250), "WBC": (0, 500), "Fibrinogen": (20, 2000),
      "Platelets": (1, 2000)}
# ranges above assume: FiO2 fraction, Lactate mmol/L, Magnesium mg/dL, Calcium mg/dL, Temp °C.
ASSUMED = {"FiO2": "fraction", "Lactate": "mmol/L", "Magnesium": "mg/dL", "Calcium": "mg/dL (total)", "Temp": "°C",
           "Hct": "%", "Hgb": "g/dL", "Glucose": "mg/dL", "Creatinine": "mg/dL", "BUN": "mg/dL",
           "Bilirubin_total": "mg/dL", "PaCO2": "mmHg", "SaO2": "%", "O2Sat": "%"}
mismatch = {v: (VERIFIED_UNITS[v], ASSUMED[v]) for v in ASSUMED if VERIFIED_UNITS[v] != ASSUMED[v]}
if mismatch:
    log.error("Verified units differ from the units assumed by the QC ranges: %s", mismatch)
    raise RuntimeError(f"Unit mismatch {mismatch}: update QC/ASSUMED before continuing (do not ignore).")
FIO2_PERCENT_FIX = VERIFIED_UNITS["FiO2"] == "fraction"

def apply_fio2_fix(s):
    s = s.copy()
    m = (s > 1) & (s <= 100)
    s[m] = s[m] / 100.0
    return s, int(m.sum())

qc_rows = []
for h in HOSPS:
    sel = hourly.hospital == h
    for v in DYNAMIC_VARS:
        x = hourly.loc[sel, v]
        n_fix = 0
        if v == "FiO2" and FIO2_PERCENT_FIX:
            x, n_fix = apply_fio2_fix(x)
        lo, hi = QC[v]
        n_obs = int(x.notna().sum())
        n_out = int(((x < lo) | (x > hi)).sum())
        qc_rows.append({"hospital": h, "variable": v, "unit": ASSUMED.get(v, DOC_UNITS[v]), "low": lo, "high": hi,
                        "n_observed": n_obs, "n_out_of_range": n_out,
                        "pct_out_of_range": 100 * n_out / max(n_obs, 1), "n_fio2_percent_converted": n_fix})
qc = pd.DataFrame(qc_rows)
qc.to_csv(RES / "tables" / "qc_out_of_range.csv", index=False)
display(qc[qc.n_out_of_range > 0].sort_values("pct_out_of_range", ascending=False).head(20))
save_json({"ranges": QC, "units": {v: ASSUMED.get(v, DOC_UNITS[v]) for v in DYNAMIC_VARS},
           "fio2_percent_to_fraction": FIO2_PERCENT_FIX}, ART / "qc_ranges.json")
save_json({"ranges": QC, "verified_units": VERIFIED_UNITS}, RES / "configs" / "qc_ranges.json")

# %% [markdown]
# ## 5. Validation of candidate physiological constraints
# Checked on **raw, same-hour, co-measured** values after QC. No forward filling, no labels.
#
# | id | relation | units | tolerance | type |
# |---|---|---|---|---|
# | C1 | MAP ≈ (SBP + 2·DBP)/3 | mmHg | 10 mmHg | equality |
# | C2 | SBP ≥ MAP ≥ DBP | mmHg | — | inequality |
# | C3 | pH ≈ 6.1 + log10(HCO3 / (0.03·PaCO2)) (Henderson–Hasselbalch) | mmol/L, mmHg | 0.05 | equality |
# | C4 | Hct ≈ 3·Hgb | %, g/dL | 3 % points | equality |
# | C5 | Bilirubin_direct ≤ Bilirubin_total | mg/dL | — | inequality |
# | C6 | SaO2 ≈ O2Sat (arterial vs pulse oximetry) | % | 4 % points | equality |
#
# The HCO3 column may be a chemistry-panel bicarbonate (total CO2) rather than the blood-gas calculated value.
# C3 therefore tests whether the relation holds *in this dataset*, which is the point of validating it.

# %%
qc_ranges = load_json(ART / "qc_ranges.json")
def qc_clean(df):
    out = df.copy()
    for v in DYNAMIC_VARS:
        if v not in out:
            continue
        if v == "FiO2" and qc_ranges["fio2_percent_to_fraction"]:
            out[v], _ = apply_fio2_fix(out[v])
        lo, hi = qc_ranges["ranges"][v]
        out.loc[(out[v] < lo) | (out[v] > hi), v] = np.nan
    return out

CONSTRAINTS = {
    "C1_MAP_formula": dict(vars=["MAP", "SBP", "DBP"], kind="eq", tol=10.0, unit="mmHg",
                           f=lambda d: d.MAP - (d.SBP + 2 * d.DBP) / 3),
    "C2_BP_ordering": dict(vars=["SBP", "MAP", "DBP"], kind="ineq", tol=0.0, unit="mmHg",
                           f=lambda d: np.maximum(d.MAP - d.SBP, d.DBP - d.MAP)),   # >0 = violation
    "C3_Henderson_Hasselbalch": dict(vars=["pH", "HCO3", "PaCO2"], kind="eq", tol=0.05, unit="pH units",
                                     f=lambda d: d.pH - (6.1 + np.log10(d.HCO3 / (0.03 * d.PaCO2)))),
    "C4_Hct_3xHgb": dict(vars=["Hct", "Hgb"], kind="eq", tol=3.0, unit="% points",
                         f=lambda d: d.Hct - 3 * d.Hgb),
    "C5_bilirubin_direct_le_total": dict(vars=["Bilirubin_direct", "Bilirubin_total"], kind="ineq", tol=0.0,
                                         unit="mg/dL", f=lambda d: d.Bilirubin_direct - d.Bilirubin_total),
    "C6_SaO2_vs_SpO2": dict(vars=["SaO2", "O2Sat"], kind="eq", tol=4.0, unit="% points",
                            f=lambda d: d.SaO2 - d.O2Sat),
}
clean = qc_clean(hourly[["hospital"] + DYNAMIC_VARS])
cv_rows, residuals = [], {}
for cid, c in CONSTRAINTS.items():
    for h in HOSPS:
        d = clean.loc[clean.hospital == h, c["vars"]].dropna()
        r = c["f"](d) if len(d) else pd.Series(dtype=float)
        residuals[(cid, h)] = r.values
        row = {"constraint": cid, "hospital": h, "n_co_measured": len(d), "unit": c["unit"], "type": c["kind"],
               "tolerance": c["tol"]}
        if len(d):
            if c["kind"] == "eq":
                row.update(median_residual=float(r.median()), mad=float((r - r.median()).abs().median()),
                           pct_within_tol=float(100 * (r.abs() <= c["tol"]).mean()))
                row["pass"] = bool(len(d) >= MIN_CO and row["pct_within_tol"] >= 70)
            else:
                row.update(pct_violations=float(100 * (r > c["tol"]).mean()))
                row["pass"] = bool(len(d) >= MIN_CO and row["pct_violations"] <= 5)
        else:
            row["pass"] = False
        cv_rows.append(row)
cv = pd.DataFrame(cv_rows)
cv.to_csv(RES / "tables" / "constraint_validation.csv", index=False)
display(cv.round(3))

accepted = [cid for cid in CONSTRAINTS if cv[(cv.constraint == cid)]["pass"].all()]
rejected = [cid for cid in CONSTRAINTS if cid not in accepted]
log.info("Accepted constraints (both hospitals): %s", accepted)
log.info("Rejected constraints: %s", rejected)
save_json({"accepted": accepted, "rejected": rejected, "min_co_measured": MIN_CO,
           "definitions": {cid: {k: v for k, v in c.items() if k != "f"} for cid, c in CONSTRAINTS.items()},
           "validation": cv.to_dict(orient="records")},
          ART / "constraints_validated.json")
save_json({"accepted": accepted, "rejected": rejected}, RES / "configs" / "constraints_validated.json")
if not accepted:
    log.warning("No constraint passed. PTC-SSL physiology loss would be empty: revisit before notebook 07.")

# %%
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
for ax, (cid, c) in zip(axes.ravel(), CONSTRAINTS.items()):
    for h in HOSPS:
        r = residuals[(cid, h)]
        if len(r):
            lo, hi = np.nanpercentile(r, [1, 99])
            ax.hist(np.clip(r, lo, hi), bins=60, alpha=0.55, density=True, label=h)
    if c["kind"] == "eq":
        for s in (-1, 1):
            ax.axvline(s * c["tol"], color="k", lw=0.8, ls="--")
    else:
        ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_title(cid.replace("_", " "), fontsize=8)
    ax.set_xlabel(f"residual ({c['unit']})", fontsize=7)
axes[0, 0].legend(frameon=False, fontsize=7)
fig.tight_layout()
savefig(fig, "fig_constraint_residuals")

# %% [markdown]
# ## 6. Evidence for the temporal window length
# For a window of W hours, how many prediction hours have a full W-hour history? This informs (but does not tune) the choice of W in `04`.

# %%
t = hourly["t"].values
wrows = []
for W in (6, 12, 24, 48, 72):
    for h in HOSPS:
        sel = (hourly.hospital == h).values
        wrows.append({"W_hours": W, "hospital": h, "pct_prediction_hours_with_full_history": 100 * (t[sel] >= W - 1).mean()})
wl = pd.DataFrame(wrows)
wl.to_csv(RES / "tables" / "window_length_coverage.csv", index=False)
display(wl.pivot(index="W_hours", columns="hospital").round(1))
display(patients.groupby("hospital").n_rows.describe(percentiles=[.1, .25, .5, .75, .9]).round(1))

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
for h in HOSPS:
    axes[0].hist(patients.loc[patients.hospital == h, "n_rows"].clip(upper=200), bins=50, alpha=.55, label=h)
    axes[1].hist(patients.loc[(patients.hospital == h) & (patients.septic == 1), "t_sepsis_idx"].clip(upper=200),
                 bins=40, alpha=.55, label=h)
axes[0].set_xlabel("Hours of ICU data per record (clipped at 200)")
axes[1].set_xlabel("Sepsis onset t_sepsis (h from record start)")
axes[0].legend(frameon=False)
fig.tight_layout()
savefig(fig, "fig_los_and_onset")

# %% [markdown]
# ## 7. Profiling report

# %%
def md_table(df):
    try:
        return df.to_markdown()
    except ImportError:
        return "```\n" + df.to_string() + "\n```"

report = f"""# Data profiling report (PhysioNet/CinC 2019)
Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}

## Cohort
{md_table(table1)}

## Unit verification (variables with ambiguous documentation)
{chr(10).join(f'- {v}: documented {DOC_UNITS[v]} -> verified {u}' for v, u in VERIFIED_UNITS.items())}

FiO2 percent-to-fraction conversion applied: {FIO2_PERCENT_FIX}

## Physiological constraints (pre-specified acceptance rule, both hospitals)
Accepted: {accepted}
Rejected: {rejected}

## Files
tables/table1_cohort.csv, tables/missingness_by_variable.csv, tables/unit_verification.csv,
tables/qc_out_of_range.csv, tables/constraint_validation.csv, tables/window_length_coverage.csv,
figures/fig_missingness_by_variable.*, figures/fig_constraint_residuals.*, figures/fig_los_and_onset.*
"""
(RES / "data_profile_report.md").write_text(report)
print(report)
print("✅ Profiling complete. Next: 03_preprocessing (attach outputs of 01 and 02).")
