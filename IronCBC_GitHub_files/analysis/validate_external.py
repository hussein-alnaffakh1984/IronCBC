"""External validation of IronCBC on a local laboratory file (reproduces the Najaf results of the paper).

Input CSV columns: age, female (1/0), WBC, RBC, HGB, HCT, MCV, MCH, MCHC, RDW_CV, PLT, ferritin
Usage: python validate_external.py <local.csv> [out.json] [--write-calibration NAME]
Patients younger than 12 are excluded. Outcome: ferritin <= 30 ng/mL.
Settings: AUC bootstrap B = 1000; local recalibration with 5-fold CV repeated 50 times;
paired bootstrap (B = 500) of the whole recalibration procedure for model-minus-rule differences. Seed 2026.
"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ironcbc import load_model, predict_raw, logit, clinical_rule, classic_indices, DEFAULT_MODEL

SEED, B_AUC, B_IDX, REPS, B_DIFF = 2026, 1000, 1000, 50, 500
def boot(n, B):
    """Bootstrap index sets; the same seed gives the same resamples in every analysis (paired, order-independent)."""
    r = np.random.default_rng(SEED)
    return [r.integers(0, n, n) for _ in range(B)]
d = pd.read_csv(sys.argv[1]); d = d[d.age >= 12].reset_index(drop=True)
out_json = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
M = load_model()
p = predict_raw(M, d); lp = logit(p)
y = (d.ferritin <= 30).astype(int).values; rule = clinical_rule(d)
R = {'n': int(len(y)), 'events': int(y.sum())}

def auc_ci(y, s, B=B_AUC):
    a = []
    for i in boot(len(y), B):
        if 0 < y[i].sum() < len(i): a.append(roc_auc_score(y[i], s[i]))
    return [round(roc_auc_score(y, s), 3), round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]

def calib(y, p):
    l = logit(p)
    slope = LogisticRegression(C=1e6).fit(l.reshape(-1, 1), y).coef_[0][0]
    citl = brentq(lambda a: (y - 1 / (1 + np.exp(-(a + l)))).sum(), -10, 10)
    return slope, citl, y.mean() / p.mean(), brier_score_loss(y, p)

def ci(vals): return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]

# discrimination and calibration with bootstrap CIs
R['auc'] = auc_ci(y, p)
s0 = calib(y, p); bs = []
for i in boot(len(y), B_AUC):
    bs.append(calib(y[i], p[i]))
bs = np.array(bs)
R['calibration'] = {k: [round(float(v), 3)] + ci(bs[:, j]) for j, (k, v) in enumerate(zip(['slope', 'citl', 'oe', 'brier'], s0))}

# fixed calibrations shipped with the model
R['fixed'] = {}
for name, cal in M['calibrations'].items():
    q = 1 / (1 + np.exp(-(cal['intercept'] + cal['slope'] * lp))); pos = q >= cal['threshold']
    R['fixed'][name] = {'sens': round(float(pos[y == 1].mean()), 3), 'spec': round(float((~pos[y == 0]).mean()), 3), 'tests_per100': round(float(pos.mean()) * 100, 1)}

def stats(pred, y):
    tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum(); tn = (~pred & (y == 0)).sum()
    return dict(sens=tp / (tp + fn), spec=tn / (tn + fp), tests=100 * pred.mean(), ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1), missed=100 * fn / len(y))

def cv_predict(score, y, rule, target=None, reps=REPS, seed0=0):
    """Out-of-fold flags. Score is recalibrated by logistic regression in training folds; threshold set in training folds
    to the rule's sensitivity (target=None) or to a fixed target sensitivity."""
    flags = np.zeros((reps, len(y)), bool); X = score.reshape(-1, 1)
    for rep in range(reps):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed0 + rep).split(X, y):
            q = LogisticRegression(C=1e6).fit(X[a], y[a]).predict_proba(X)[:, 1]
            st = rule[a][y[a] == 1].mean() if target is None else target
            qs = np.sort(q[a][y[a] == 1])[::-1]; t = qs[int(np.ceil(st * len(qs))) - 1]
            flags[rep, b] = q[b] >= t
    return flags

def summarise(flags, y):
    S = [stats(f, y) for f in flags]
    return {k: round(float(np.mean([s[k] for s in S])), 4) for k in S[0]}

R['rule'] = {k: round(float(v), 4) for k, v in stats(rule, y).items()}
R['model_at_rule_sens'] = summarise(cv_predict(lp, y, rule), y)
R['hb_at_rule_sens'] = summarise(cv_predict(-d.HGB.values, y, rule), y)
for tgt in (0.90, 0.95):
    R[f'model_at_sens_{int(tgt*100)}'] = summarise(cv_predict(lp, y, rule, target=tgt), y)

# paired bootstrap of the whole procedure: model minus rule
diffs = []
for b, i in enumerate(boot(len(y), B_DIFF)):
    yy, ll, rr = y[i], lp[i], rule[i]
    if yy.sum() < 10: continue
    f = cv_predict(ll, yy, rr, reps=1, seed0=b)[0]
    sm, sr = stats(f, yy), stats(rr, yy)
    diffs.append([sm['tests'] - sr['tests'], sm['spec'] - sr['spec'], sm['sens'] - sr['sens']])
diffs = np.array(diffs)
R['model_minus_rule'] = {'tests_per100': [round(R['model_at_rule_sens']['tests'] - R['rule']['tests'], 1)] + [round(float(x), 1) for x in np.percentile(diffs[:, 0], [2.5, 97.5])],
                         'spec': [round(R['model_at_rule_sens']['spec'] - R['rule']['spec'], 3)] + ci(diffs[:, 1]),
                         'sens': [round(R['model_at_rule_sens']['sens'] - R['rule']['sens'], 3)] + ci(diffs[:, 2])}

# comparators: indices, NHANES-trained logistic regression, locally fitted logistic regression
R['indices_auc'] = {k: auc_ci(y, v.values, B_IDX) for k, v in classic_indices(d).items()}
F = M['features']
nh = pd.read_csv(os.path.join(os.path.dirname(DEFAULT_MODEL), '..', 'data', 'nhanes_train.csv')).dropna(subset=F + ['ferritin'])
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(nh[F], (nh.ferritin <= 30).astype(int))
R['nhanes_logistic_auc'] = auc_ci(y, lr.predict_proba(d[F])[:, 1])
oof = np.zeros(len(y))
for rep in range(REPS):
    for a, b in StratifiedKFold(5, shuffle=True, random_state=rep).split(d, y):
        oof[b] += make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(d.loc[a, F], y[a]).predict_proba(d.loc[b, F])[:, 1] / REPS
R['local_logistic_cv_auc'] = round(roc_auc_score(y, oof), 3)

# paired bootstrap of AUC differences: model minus comparator
comp = {'Hb': -d.HGB.values, 'Ricerca': (d.RDW_CV / d.RBC).values, 'NHANES logistic': lr.predict_proba(d[F])[:, 1]}
R['auc_difference'] = {}
for k, s in comp.items():
    dd = []
    for i in boot(len(y), B_AUC):
        dd.append(roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], s[i]))
    R['auc_difference'][k] = [round(roc_auc_score(y, p) - roc_auc_score(y, s), 3)] + ci(dd)

# baseline demographic comparators: age and sex only
D = ['age', 'female']
R['sex_only_auc'] = auc_ci(y, d.female.values)
R['age_sex_nhanes_logistic_auc'] = auc_ci(y, LogisticRegression(max_iter=2000).fit(nh[D], (nh.ferritin <= 30).astype(int)).predict_proba(d[D])[:, 1])
oof2 = np.zeros(len(y))
for rep in range(REPS):
    for a, b in StratifiedKFold(5, shuffle=True, random_state=rep).split(d, y):
        oof2[b] += LogisticRegression(max_iter=2000).fit(d.loc[a, D], y[a]).predict_proba(d.loc[b, D])[:, 1] / REPS
R['age_sex_local_cv_auc'] = round(roc_auc_score(y, oof2), 3)

# women aged 12-49: model vs haemoglobin
m = ((d.female == 1) & (d.age <= 49)).values
dd = []
for j in boot(int(m.sum()), B_AUC):
    i = np.where(m)[0][j]
    dd.append(roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], -d.HGB.values[i]))
R['women_12_49'] = {'n': int(m.sum()), 'events': int(y[m].sum()), 'auc_model': auc_ci(y[m], p[m]), 'auc_hb': auc_ci(y[m], -d.HGB.values[m]),
                    'diff': [round(roc_auc_score(y[m], p[m]) - roc_auc_score(y[m], -d.HGB.values[m]), 3)] + ci(dd)}

# decision curve: cross-validated recalibrated probabilities (model, haemoglobin), clinical rule, test all
def oof_recal(score):
    o = np.zeros(len(y)); X = score.reshape(-1, 1)
    for rep in range(REPS):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            o[b] += LogisticRegression(C=1e6).fit(X[a], y[a]).predict_proba(X[b])[:, 1] / REPS
    return o
pm, ph = oof_recal(lp), oof_recal(-d.HGB.values)
def nb(flag, t): n = len(y); return ((flag & (y == 1)).sum() - (flag & (y == 0)).sum() * t / (1 - t)) / n
R['dca'] = [{'t': round(t, 2), 'model': round(nb(pm >= t, t), 4), 'hb': round(nb(ph >= t, t), 4), 'rule': round(nb(rule, t), 4),
             'all': round(nb(np.ones(len(y), bool), t), 4)} for t in np.arange(0.05, 0.71, 0.05)]
pd.DataFrame({'y': y, 'p_model_cv': pm, 'p_hb_cv': ph, 'rule': rule.astype(int)}).to_csv(os.path.splitext(out_json or 'out.json')[0] + '_dca.csv', index=False) if out_json else None

# bootstrap CIs for net-benefit differences (predictions fixed) at selected thresholds
R['dca_diff_ci'] = {}
for t in (0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
    dl = []
    for i in boot(len(y), B_AUC):
        yy = y[i]
        nbi = lambda fl: ((fl & (yy == 1)).sum() - (fl & (yy == 0)).sum() * t / (1 - t)) / len(yy)
        dl.append([nbi(pm[i] >= t) - nbi(np.ones(len(yy), bool)), nbi(pm[i] >= t) - nbi(ph[i] >= t), nbi(pm[i] >= t) - nbi(rule[i])])
    dl = np.array(dl)
    R['dca_diff_ci'][f'{t:.2f}'] = {'model_minus_all': [round(nb(pm >= t, t) - nb(np.ones(len(y), bool), t), 4)] + ci(dl[:, 0]),
                                   'model_minus_hb': [round(nb(pm >= t, t) - nb(ph >= t, t), 4)] + ci(dl[:, 1]),
                                   'model_minus_rule': [round(nb(pm >= t, t) - nb(rule, t), 4)] + ci(dl[:, 2])}

# women aged 12-49: flagged per 100 at the rule's sensitivity (flags from the whole-cohort cross-validation)
fl = cv_predict(lp, y, rule)
w = ((d.female == 1) & (d.age <= 49)).values
R['women_12_49']['rule'] = {k: round(float(v), 4) for k, v in stats(rule[w], y[w]).items()}
R['women_12_49']['model_at_rule_sens'] = summarise(fl[:, w], y[w])

# precision of O/E (Riley et al. 2021): SE(ln O/E) = sqrt((1 - phi) / (n * phi))
phi = y.mean(); se = np.sqrt((1 - phi) / (len(y) * phi))
R['riley_oe'] = {'se_ln_oe': round(float(se), 4), 'ci_ratio_width_at_oe1': round(float(np.exp(1.96 * se) - np.exp(-1.96 * se)), 3),
                 'n_for_width_0.2': int(np.ceil((1 - phi) / (phi * (np.arcsinh(0.1) / 1.96) ** 2)))}

# subgroups
for name, m in [('women', d.female == 1), ('age_18_plus', d.age >= 18), ('microcytic_MCV_lt80', d.MCV < 80)]:
    m = m.values
    R[f'sub_{name}'] = {'n': int(m.sum()), 'events': int(y[m].sum()), 'auc': auc_ci(y[m], p[m]),
                        'mentzer_auc': auc_ci(y[m], (d.MCV / d.RBC).values[m], B_IDX)}

print(json.dumps(R, indent=1))
if out_json: json.dump(R, open(out_json, 'w'), indent=1)

if '--write-calibration' in sys.argv:
    name = sys.argv[sys.argv.index('--write-calibration') + 1]
    r = LogisticRegression(C=1e6).fit(lp.reshape(-1, 1), y); q = r.predict_proba(lp.reshape(-1, 1))[:, 1]
    st = rule[y == 1].mean(); qs = np.sort(q[y == 1])[::-1]; thr = float(qs[int(np.ceil(st * len(qs))) - 1])
    raw = json.load(open(DEFAULT_MODEL))
    raw['calibrations'][name] = {'intercept': round(float(r.intercept_[0]), 4), 'slope': round(float(r.coef_[0][0]), 4),
                                 'threshold': round(thr, 4), 'n': int(len(y)), 'events': int(y.sum()),
                                 'note': 'logistic recalibration of logit(p); threshold matches clinical-rule sensitivity'}
    json.dump(raw, open(DEFAULT_MODEL, 'w'), separators=(',', ':'))
    print('calibration written:', raw['calibrations'][name])
