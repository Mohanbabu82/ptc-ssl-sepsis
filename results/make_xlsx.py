import pandas as pd, json
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
S = pd.read_pickle('store.pkl')
tabs = json.load(open('tables.json'))
cohort = pd.DataFrame(tabs[0][1:], columns=tabs[0][0])
cons = pd.DataFrame(tabs[1][1:], columns=tabs[1][0])
units = pd.DataFrame([
 ['Lactate', 'mg/dL', 'mmol/L', 'Medians 1.80 (A), 1.97 (B)'],
 ['Magnesium', 'mmol/dL', 'mg/dL', 'Median 2.0 in both hospitals'],
 ['FiO2', '%', 'fraction', 'Medians 0.50 (A), 0.40 (B); 16.3% of B values (0.18% of A) > 1 were percentages and were converted']],
 columns=['Variable', 'Documented unit', 'Verified unit', 'Evidence'])
ra = S['runs_a'].sort_values(['job', 'arm', 'fraction', 'seed']); rb = S['runs_b'].sort_values(['job', 'protocol', 'arm', 'fraction', 'seed'])
sheets = [
 ('Summary_by_arm', S['summ']), ('Runs_v2a_core', ra), ('Runs_v2b_baselines_transfer', rb),
 ('Comparisons', S['cps']), ('Factorial_effects', S['fx']), ('Perturbation_summary', S['ps']),
 ('Perturbation_runs', S['pert']), ('Perturbation_info', S['pinfo']), ('Hospital_shift', S['sh']),
 ('Hospital_classifier', S['hc']), ('Measurement_frequency', S['fq']), ('Cohort', cohort),
 ('Constraint_validation', cons), ('Unit_verification', units)]
fn = 'PTC_SSL_Complete_Results.xlsx'
with pd.ExcelWriter(fn, engine='openpyxl') as w:
    pd.DataFrame().to_excel(w, sheet_name='README')
    pd.DataFrame().to_excel(w, sheet_name='Main_comparison')
    for n, d in sheets: d.to_excel(w, sheet_name=n, index=False)
wb = load_workbook(fn)
H = Font(name='Arial', bold=True, color='FFFFFF', size=10); F = Font(name='Arial', size=10)
fill = PatternFill('solid', start_color='1F4E79'); thin = Side(style='thin', color='BFBFBF')
for n, d in sheets:
    ws = wb[n]
    for c in ws[1]: c.font = H; c.fill = fill; c.alignment = Alignment(wrap_text=True, vertical='center')
    ws.freeze_panes = 'A2'; ws.auto_filter.ref = ws.dimensions
    for j, col in enumerate(d.columns, 1):
        L = get_column_letter(j)
        ws.column_dimensions[L].width = min(max(len(str(col)) + 2, 10), 28)
        for c in ws[L][1:]:
            c.font = F
            if isinstance(c.value, float): c.number_format = '0.0000'
    ws.row_dimensions[1].height = 30
# Main comparison: formulas over Runs_v2a_core (main job)
ws = wb['Main_comparison']
ws['A1'] = 'Main pooled comparison: mean over seeds, computed by formula from sheet Runs_v2a_core (job = main)'; ws['A1'].font = Font(name='Arial', bold=True, size=11)
ws['A2'] = 'SD across seeds and bootstrap CIs are in Summary_by_arm and Comparisons.'; ws['A2'].font = Font(name='Arial', italic=True, size=9)
arms = [('lightgbm', 'LightGBM'), ('supervised_gru', 'Supervised GRU'), ('generic_ssl', 'Generic SSL'), ('ptc_ssl', 'PTC-SSL')]
fr = [0.01, 0.02, 0.05, 0.1, 0.25, 1.0]
cols = {c: get_column_letter(i + 1) for i, c in enumerate(ra.columns)}
nr = len(ra) + 1
def rng(c): return f"Runs_v2a_core!${cols[c]}$2:${cols[c]}${nr}"
r = 4
for metric in ['AUROC', 'AUPRC', 'utility', 'ECE', 'ECE_cal']:
    ws.cell(r, 1, f'{metric} (mean of 10 seeds)').font = Font(name='Arial', bold=True, size=10); r += 1
    ws.cell(r, 1, 'Arm \\ labelled fraction').font = H; ws.cell(r, 1).fill = fill
    ws.cell(r, 2, 'arm key').font = H; ws.cell(r, 2).fill = fill
    for j, f in enumerate(fr):
        c = ws.cell(r, 3 + j, f); c.font = H; c.fill = fill; c.number_format = '0%'
    hr = r; r += 1
    for key, name in arms:
        ws.cell(r, 1, name).font = F; ws.cell(r, 2, key).font = F
        for j in range(len(fr)):
            fc = get_column_letter(3 + j)
            c = ws.cell(r, 3 + j, f'=AVERAGEIFS({rng(metric)},{rng("job")},"main",{rng("arm")},$B{r},{rng("fraction")},{fc}${hr})')
            c.font = F; c.number_format = '0.000'
        r += 1
    ws.cell(r, 1, 'n runs').font = Font(name='Arial', italic=True, size=9)
    for j in range(len(fr)):
        fc = get_column_letter(3 + j)
        c = ws.cell(r, 3 + j, f'=COUNTIFS({rng("job")},"main",{rng("fraction")},{fc}${hr})'); c.font = Font(name='Arial', italic=True, size=9)
    r += 2
ws.column_dimensions['A'].width = 30; ws.column_dimensions['B'].width = 16
for j in range(6): ws.column_dimensions[get_column_letter(3 + j)].width = 11
# README
ws = wb['README']
txt = [
 ('PTC-SSL capstone: complete experimental results (revision 2)', True),
 ('Dataset: PhysioNet/Computing in Cardiology Challenge 2019, v1.0.0 (40,336 patients; hospitals A and B). No raw patient data are included in this file.', False),
 ('Source: every number below was parsed from the run logs of the two Kaggle notebooks (one Tesla T4 GPU each), run on 2026-11-20:', False),
 ('  v2a core validity: 480 training runs + 200 perturbation evaluations, 6.03 h, 0 skipped jobs (PTC_SSL_V2A_RESULTS.zip on Kaggle).', False),
 ('  v2b baselines and transfer: 330 training runs, 7.91 h, 0 skipped jobs (PTC_SSL_V2B_RESULTS.zip on Kaggle).', False),
 ('Test-set predictions (per hour) are not in this workbook because of their size; they are in the Kaggle output zips.', False),
 ('', False), ('Sheets', True),
 ('Main_comparison: mean AUROC, AUPRC, utility and ECE per arm and label fraction, computed by formula from Runs_v2a_core.', False),
 ('Summary_by_arm: mean and SD over seeds for every arm, job, protocol and fraction (as logged by the notebooks).', False),
 ('Runs_v2a_core: one row per training run (jobs main, factorial, missingness; pooled protocol; seeds 0-9).', False),
 ('Runs_v2b_baselines_transfer: one row per training run (jobs main, ssl_baselines, sensitivity, external, external_diag; seeds 0-4).', False),
 ('Comparisons: arm minus comparator; per-seed hierarchical bootstrap (2,000 resamples) and seed-ensemble patient bootstrap, with raw and Holm-adjusted p-values.', False),
 ('Factorial_effects: main effects and interactions of organ masking (O), physiological loss (P) and temporal loss (T); +/-1 coding, effect = contrast/4.', False),
 ('Perturbation_summary / Perturbation_runs / Perturbation_info: test-time removal of labs, vitals, lactate, or capping labs at 1 per 24 h.', False),
 ('Hospital_shift / Hospital_classifier / Measurement_frequency: measurement rates, distribution shift and hospital separability.', False),
 ('Cohort / Constraint_validation / Unit_verification: data verification results.', False),
 ('', False), ('Key columns', True),
 ('job: experiment family. protocol: pooled, A2B (train A, test B) or B2A. arm: model variant. fraction: share of labelled training patients. seed: label subset, pretraining and initialisation seed.', False),
 ('AUROC / AUPRC: hourly test-set ranking metrics. utility: official challenge normalised utility. Brier / ECE: before recalibration; Brier_cal / ECE_cal: after validation-only Platt scaling (platt_a, platt_b).', False),
 ('threshold: alarm threshold chosen to maximise utility on the label-budgeted validation set. septic_detected_pct: septic patients alarmed by onset + 3 h. false_alarms_per_patient_day: on non-septic patients.', False),
 ('Arm names: ptc_ssl = PTC-SSL; generic_ssl = random-mask reconstruction; fac_* = factorial cells; abl_missing_only / abl_values_only = input ablations; lgbm_* = LightGBM feature ablations; ts2vec_gru, simmtm_gru = other SSL objectives; *_tf = Transformer; sens_* = sensitivity; *_uda = pretraining also on target-hospital adaptation half; ptc_no_fio2 = FiO2 removed.', False),
]
for i, (t, b) in enumerate(txt, 1):
    c = ws.cell(i, 1, t); c.font = Font(name='Arial', bold=b, size=12 if i == 1 else 10)
ws.column_dimensions['A'].width = 160
wb.move_sheet('README', -wb.index(wb['README']))
wb.save(fn)
print('ok')
