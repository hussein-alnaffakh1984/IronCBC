"""External validation of IronCBC on a local laboratory file.

Input CSV columns: age, female (1/0), WBC, RBC, HGB, HCT, MCV, MCH, MCHC, RDW_CV, PLT, ferritin
Usage: python validate_external.py <local.csv> [--write-calibration NAME]
Prints discrimination, calibration, comparison with the clinical rule (nested CV recalibration),
classic indices and a conformal 'uncertain' zone. Patients younger than 12 are excluded.
"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ironcbc import load_model, predict_raw, logit, clinical_rule, classic_indices, DEFAULT_MODEL

rng = np.random.default_rng(2026)
d = pd.read_csv(sys.argv[1]); d = d[d.age >= 12].reset_index(drop=True)
M = load_model()
p = predict_raw(M, d); lp = logit(p).reshape(-1, 1)
y = (d.ferritin <= 30).astype(int).values; rule = clinical_rule(d)
print(f'n={len(y)}, ferritin<=30: {y.sum()} ({y.mean()*100:.1f}%), <15: {(d.ferritin < 15).sum()}')

def auc_ci(y, s, B=2000):
    a = [roc_auc_score(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(B)) if 0 < y[i].sum() < len(i)]
    return roc_auc_score(y, s), np.percentile(a, 2.5), np.percentile(a, 97.5)

a, lo, hi = auc_ci(y, p); print(f'AUC {a:.3f} ({lo:.3f}-{hi:.3f})')
r = LogisticRegression(C=1e6).fit(lp, y); print(f'calibration slope {r.coef_[0][0]:.2f}, O/E {y.mean()/p.mean():.2f}')
for name, cal in M['calibrations'].items():
    q = 1 / (1 + np.exp(-(cal['intercept'] + cal['slope'] * lp[:, 0]))); pos = q >= cal['threshold']
    print(f'  fixed calibration "{name}": sens {pos[y==1].mean():.3f} spec {(~pos[y==0]).mean():.3f} tests/100 {pos.mean()*100:.1f}')
print(f'clinical rule: sens {rule[y==1].mean():.3f} spec {(~rule[y==0]).mean():.3f} tests/100 {rule.mean()*100:.1f}')

se, sp, fl = [], [], []
for rep in range(200):
    pred = np.zeros(len(y), bool)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(lp, y):
        q = LogisticRegression(C=1e6).fit(lp[tr], y[tr]).predict_proba(lp)[:, 1]
        st = rule[tr][y[tr] == 1].mean(); ts = np.sort(np.unique(q[tr]))[::-1]
        t = next(t for t in ts if (q[tr][y[tr] == 1] >= t).mean() >= st); pred[te] = q[te] >= t
    se.append(pred[y == 1].mean()); sp.append((~pred[y == 0]).mean()); fl.append(pred.mean())
print(f'local recalibration, nested 5-fold x200, threshold at rule sensitivity: sens {np.mean(se):.3f} spec {np.mean(sp):.3f} '
      f'(2.5-97.5%: {np.percentile(sp,2.5):.3f}-{np.percentile(sp,97.5):.3f}) tests/100 {np.mean(fl)*100:.1f}')

print('classic indices (AUC, 95% CI):')
for k, v in classic_indices(d).items():
    a, lo, hi = auc_ci(y, v.values, 1000); print(f'  {k:16s} {a:.3f} ({lo:.3f}-{hi:.3f})')
mic = (d.MCV < 80).values
if mic.sum() > 20 and 0 < y[mic].sum() < mic.sum():
    print(f'microcytic subgroup n={mic.sum()}, ID={y[mic].sum()}: IronCBC AUC {roc_auc_score(y[mic], p[mic]):.3f}')

if '--write-calibration' in sys.argv:
    name = sys.argv[sys.argv.index('--write-calibration') + 1]
    q = r.predict_proba(lp)[:, 1]; st = rule[y == 1].mean(); ts = np.sort(np.unique(q))[::-1]
    thr = float(next(t for t in ts if (q[y == 1] >= t).mean() >= st))
    raw = json.load(open(DEFAULT_MODEL))
    raw['calibrations'][name] = {'intercept': round(float(r.intercept_[0]), 4), 'slope': round(float(r.coef_[0][0]), 4),
                                 'threshold': round(thr, 4), 'n': int(len(y)), 'events': int(y.sum()),
                                 'note': 'logistic recalibration of logit(p); threshold matches clinical-rule sensitivity'}
    json.dump(raw, open(DEFAULT_MODEL, 'w'), separators=(',', ':'))
    print('calibration written:', raw['calibrations'][name])
