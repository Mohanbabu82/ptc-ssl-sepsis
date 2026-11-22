# Data profiling report (PhysioNet/CinC 2019)
Generated: 2026-11-20T07:52:03.490781+00:00

## Cohort
|                                    | A                  | B                  | All                |
|:-----------------------------------|:-------------------|:-------------------|:-------------------|
| patients                           | 20336              | 20000              | 40336              |
| patient_hours                      | 790215             | 761995             | 1552210            |
| ICU hours per record, median [IQR] | 39.0 [25.0–47.0]   | 38.0 [23.0–47.0]   | 38.0 [24.0–47.0]   |
| age, median [IQR]                  | 64.7 [52.3–75.7]   | 62.0 [50.0–72.0]   | 63.1 [51.0–74.0]   |
| male, %                            | 58.2               | 53.7               | 55.9               |
| MICU / SICU / unit missing, %      | 26.3 / 26.9 / 46.8 | 34.6 / 34.9 / 30.5 | 30.4 / 30.9 / 38.7 |
| septic, n (%)                      | 1790 (8.80)        | 1142 (5.71)        | 2932 (7.27)        |
| t_sepsis (row h), median [IQR]     | 34.0 [13.0–77.0]   | 33.5 [9.0–79.0]    | 34.0 [11.8–77.0]   |
| septic with label at first row, n  | 203                | 223                | 426                |
| septic with first label <= 6 h, n  | 436                | 342                | 778                |

## Unit verification (variables with ambiguous documentation)
- FiO2: documented % -> verified fraction
- Lactate: documented mg/dL -> verified mmol/L
- Magnesium: documented mmol/dL -> verified mg/dL
- Calcium: documented mg/dL -> verified mg/dL (total)
- Temp: documented °C -> verified °C
- Hct: documented % -> verified %
- Hgb: documented g/dL -> verified g/dL
- Glucose: documented mg/dL -> verified mg/dL
- Creatinine: documented mg/dL -> verified mg/dL
- BUN: documented mg/dL -> verified mg/dL
- Bilirubin_total: documented mg/dL -> verified mg/dL
- PaCO2: documented mmHg -> verified mmHg
- SaO2: documented % -> verified %
- O2Sat: documented % -> verified %

FiO2 percent-to-fraction conversion applied: True

## Physiological constraints (pre-specified acceptance rule, both hospitals)
Accepted: ['C1_MAP_formula', 'C2_BP_ordering', 'C3_Henderson_Hasselbalch', 'C4_Hct_3xHgb', 'C5_bilirubin_direct_le_total', 'C6_SaO2_vs_SpO2']
Rejected: []

## Files
tables/table1_cohort.csv, tables/missingness_by_variable.csv, tables/unit_verification.csv,
tables/qc_out_of_range.csv, tables/constraint_validation.csv, tables/window_length_coverage.csv,
figures/fig_missingness_by_variable.*, figures/fig_constraint_residuals.*, figures/fig_los_and_onset.*
