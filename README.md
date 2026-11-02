# IronCBC

**University of Alkafeel — College of Health and Medical Technologies — Department of Medical Laboratory Techniques**

Graduation research project of students of the Department of Medical Laboratory Techniques, supervised by Asst. Lect. Hussein Ali Hussein Al-Naffakh. For educational and research purposes only.

Estimate the probability of iron deficiency (serum ferritin ≤30 ng/mL, ASH 2026) from a routine complete blood count, age and sex, and calibrate the decision threshold to your own laboratory.

**Web app:** https://ironcbc.netlify.app (source: [`web/index.html`](IronCBC_GitHub_files/web/index.html)). Everything runs in the browser; no data leave the device.

**All code and data files are in the folder [`IronCBC_GitHub_files`](IronCBC_GitHub_files).**

## Main results (ferritin ≤30 ng/mL)

| | MIMIC-IV hospital (n = 67,471) | Najaf outpatient lab (n = 286) |
|---|---|---|
| IronCBC AUC (95% CI) | 0.789 (0.784–0.794) | 0.833 (0.779–0.878) |
| Hb alone, AUC | 0.562 | 0.790 |
| Best classic index, AUC | 0.614 (RDW) | 0.793 (Ricerca) |
| Ferritin tests per 100 patients, clinical rule (low Hb or MCV <80) | 66.7 | 56.6 |
| Ferritin tests per 100, IronCBC after local recalibration, same sensitivity | 36.7 | 51.4 |

Recalibration and thresholds were evaluated with nested 5-fold cross-validation within each setting.

## What is in this repository

| Path | Content |
|---|---|
| `IronCBC_GitHub_files/web/index.html` | Bilingual (Arabic/English) web app: single patient, batch file, calibrate to your lab |
| `IronCBC_GitHub_files/model/ironcbc_model.json` | The exported random forest (100 trees) and its calibrations |
| `IronCBC_GitHub_files/data/nhanes_train.csv` | NHANES training table (public data, derived) |
| `IronCBC_GitHub_files/analysis/prepare_nhanes.py` | Builds `nhanes_train.csv` from NHANES XPT files |
| `IronCBC_GitHub_files/analysis/train_export.py` | Trains the model and exports it to JSON |
| `IronCBC_GitHub_files/analysis/validate_external.py` | External validation on a local file (AUC, calibration, clinical rule, classic indices) |
| `IronCBC_GitHub_files/analysis/ironcbc.py` | Scores rows with the exported model exactly as the web app does |
| `IronCBC_GitHub_files/analysis/build_web.py` | Embeds the model into `web/index.html` |
| `IronCBC_GitHub_files/mimic/run_mimic.py` | MIMIC-IV analysis, to be run locally by a credentialed user |
| `IronCBC_GitHub_files/mimic/calibrate_mimic.py` | Fits the MIMIC-IV hospital calibration (aggregate output only) |

## Model

- **Training data:** NHANES 2015–2016, 2017–March 2020 and 2021–2023; age ≥12 years; non-pregnant; complete CBC and ferritin (n = 13,047).
- **Inputs (11):** age, sex, WBC, RBC, Hb, HCT, MCV, MCH, MCHC, RDW-CV, platelets.
- **Outcome:** ferritin ≤30 ng/mL.
- **Algorithm:** scikit-learn `RandomForestClassifier(n_estimators=100, min_samples_leaf=40, random_state=0)`.
- **Calibrations shipped:**
  - `najaf_outpatient_lab`: logistic recalibration on 286 patients from an outpatient laboratory in Najaf, Iraq. The threshold matches the sensitivity of the clinical rule.
  - `mimic_hospital`: logistic recalibration on 67,471 MIMIC-IV hospital patients. The threshold matches the sensitivity of the clinical rule.
  - `nhanes_uncalibrated`: no recalibration. The threshold gives 80% sensitivity in NHANES cross-validation (comparison only).

The model ranks patients well across settings, but its probabilities and threshold do not transfer between laboratories. Recalibrate locally before use; the app's **Calibrate to your lab** tab does this from a file of paired CBC and ferritin results.

## Reproduce

```bash
cd IronCBC_GitHub_files
pip install -r requirements.txt
python analysis/prepare_nhanes.py <folder_with_NHANES_XPT_files>   # optional; data/nhanes_train.csv is included
python analysis/train_export.py
python analysis/validate_external.py <local.csv> --write-calibration my_lab
python analysis/build_web.py
```

The local CSV needs these columns: `age, female (1/0), WBC, RBC, HGB, HCT, MCV, MCH, MCHC, RDW_CV, PLT, ferritin`.

## MIMIC-IV

MIMIC-IV data must not be shared or uploaded to online services (PhysioNet Credentialed Data Use Agreement). `mimic/run_mimic.py` reads `hosp/labevents`, `hosp/d_labitems` and `hosp/patients` locally, and writes aggregate results only:

```bash
cd IronCBC_GitHub_files/mimic
python run_mimic.py "<path to the MIMIC-IV folder that contains hosp>"
```

## Data availability

- **NHANES** data are public.
- **Najaf** patient data are not included. They are available from the author on reasonable request, subject to ethics approval.
- **MIMIC-IV** is available to credentialed users from PhysioNet.

## Disclaimer

IronCBC is a research tool, not a diagnostic device. It does not reliably separate iron deficiency from thalassaemia trait, and inflammation can mask iron deficiency.

## Citation

Al-Naffakh HAH. IronCBC: external validation and local calibration of a complete-blood-count model for iron deficiency (manuscript in preparation), 2026.

## License

MIT
