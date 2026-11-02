"""
CBC -> iron deficiency: ADDITIONAL MIMIC-IV analyses requested by reviewers
(run locally; MIMIC rows never leave this computer).

Usage (Windows CMD), from this folder:
    python run_mimic_extra.py "C:\\path\\to\\mimic-iv-3.1"
The path is the folder that contains the "hosp" folder.
Output: results_mimic_extra.txt and results_mimic_extra.json (aggregate numbers only).
"""
import os, sys, json, time, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON = os.path.join(HERE, '..', 'model', 'ironcbc_model.json')
NHANES_CSV = os.path.join(HERE, '..', 'data', 'nhanes_train.csv')
LOCAL_CSV = os.path.join(HERE, '..', 'data', 'local_lab.csv')   # optional, not distributed
OUT_TXT = os.path.join(HERE, 'results_mimic_extra.txt')
LOG = []
RES = {}
T0 = time.time()
CHUNK = int(os.environ.get('CHUNK', '5000000'))

def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

def find(root, name):
    hits = glob.glob(os.path.join(root, '**', name), recursive=True)
    if not hits:
        sys.exit(f'ERROR: {name} not found under {root}')
    return hits[0]

import traceback, atexit
def _crash(t, v, tb):
    msg = ''.join(traceback.format_exception(t, v, tb)); print(msg, flush=True)
    open(os.path.join(HERE, 'results_mimic_extra_ERROR.txt'), 'w', encoding='utf-8').write('\n'.join(LOG) + '\n' + msg)
sys.excepthook = _crash
# ------------------------------------------------------------------ 1. locate files
if len(sys.argv) < 2:
    sys.exit('Give the MIMIC-IV folder path, e.g.  python run_mimic.py "D:\\mimic-iv-3.1"')
ROOT = sys.argv[1]
F_ITEMS = find(ROOT, 'd_labitems.csv*')
F_LAB = find(ROOT, 'labevents.csv*')
F_PAT = find(ROOT, 'patients.csv*')
say('d_labitems:', F_ITEMS); say('labevents :', F_LAB); say('patients  :', F_PAT)

# ------------------------------------------------------------------ 2. item ids by label
items = pd.read_csv(F_ITEMS)
items['label_l'] = items['label'].astype(str).str.strip().str.lower()
items['fluid_l'] = items['fluid'].astype(str).str.strip().str.lower()
items['cat_l'] = items['category'].astype(str).str.strip().str.lower()
WANT = {
    'HGB': (['hemoglobin'], 'hematology'),
    'HCT': (['hematocrit'], 'hematology'),
    'RBC': (['red blood cells'], 'hematology'),
    'WBC': (['white blood cells', 'wbc count'], 'hematology'),
    'MCV': (['mcv'], 'hematology'),
    'MCH': (['mch'], 'hematology'),
    'MCHC': (['mchc'], 'hematology'),
    'RDW_CV': (['rdw'], 'hematology'),
    'PLT': (['platelet count'], 'hematology'),
    'MPV': (['mpv', 'mean platelet volume'], 'hematology'),
    'ferritin': (['ferritin'], 'chemistry'),
    'CRP': (['c-reactive protein'], 'chemistry'),
    'IRON': (['iron'], 'chemistry'),
    'TIBC': (['iron binding capacity, total'], 'chemistry'),
}
ID2VAR = {}
for var, (labels, cat) in WANT.items():
    m = items[items.label_l.isin(labels) & (items.fluid_l == 'blood') & (items.cat_l == cat)]
    for _, r in m.iterrows():
        ID2VAR[int(r.itemid)] = var
    say(f'item {var:8s}: {m.itemid.tolist()} {m.label.tolist()}')
RES['itemids'] = {v: [k for k, x in ID2VAR.items() if x == v] for v in WANT}
for must in ['HGB', 'HCT', 'RBC', 'MCV', 'MCH', 'MCHC', 'RDW_CV', 'PLT', 'WBC', 'ferritin']:
    if not RES['itemids'][must]:
        sys.exit(f'ERROR: no itemid found for {must}; send results_mimic.txt so the label list can be fixed')

# ------------------------------------------------------------------ 3. read labevents in chunks
say('Reading labevents (this can take 10-40 minutes)...')
keep = []; ferh = []
n = 0
FER_IDS = [k for k, v in ID2VAR.items() if v == 'ferritin']
for ch in pd.read_csv(F_LAB, usecols=['subject_id', 'hadm_id', 'specimen_id', 'itemid', 'charttime', 'valuenum'],
                      chunksize=CHUNK):
    ch = ch[ch.itemid.isin(ID2VAR.keys()) & ch.valuenum.notna()]
    ferh.append(ch.loc[ch.itemid.isin(FER_IDS), ['subject_id', 'charttime', 'hadm_id']])   # admission id kept for ferritin rows only
    ch = ch.drop(columns='hadm_id')
    ch = ch.astype({'subject_id': 'int32', 'specimen_id': 'int64', 'itemid': 'int32', 'valuenum': 'float32'})
    ch['charttime'] = pd.to_datetime(ch.charttime)
    keep.append(ch)
    n += 1
    say(f'  chunk {n} done, kept so far {sum(len(k) for k in keep):,} rows, {time.time()-T0:.0f}s')
lab = pd.concat(keep, ignore_index=True); del keep
lab['var'] = lab.itemid.map(ID2VAR).astype('category')
say('Kept rows by variable:', lab['var'].value_counts().to_dict())

# ------------------------------------------------------------------ 4. CBC panels and ferritin pairs (memory-lean)
cbcv = ['HGB', 'HCT', 'RBC', 'WBC', 'MCV', 'MCH', 'MCHC', 'RDW_CV', 'PLT', 'MPV']
core = ['HGB', 'HCT', 'RBC', 'WBC', 'MCV', 'MCH', 'MCHC', 'RDW_CV', 'PLT']
cb = lab.loc[lab['var'].isin(cbcv), ['subject_id', 'specimen_id', 'charttime', 'var', 'valuenum']]
cb = cb.drop_duplicates(['specimen_id', 'var'])
cb['var'] = cb['var'].astype(str)
say('  building CBC panels...')
wide = cb.pivot(index='specimen_id', columns='var', values='valuenum')
meta = cb.drop_duplicates('specimen_id').set_index('specimen_id')[['subject_id', 'charttime']]
del cb
wide = wide.dropna(subset=core)
cbc = meta.loc[wide.index].join(wide).reset_index()
del wide, meta
cbc['subject_id'] = cbc['subject_id'].astype('int64')
HAS_MPV = 'MPV' in cbc.columns and cbc['MPV'].notna().mean() > 0.8
say(f'Complete CBC panels: {len(cbc):,}; MPV available: {HAS_MPV}')
fer = lab.loc[lab['var'] == 'ferritin', ['subject_id', 'charttime', 'valuenum']].rename(columns={'valuenum': 'ferritin'})
fer['subject_id'] = fer['subject_id'].astype('int64')
fh = pd.concat(ferh, ignore_index=True); fh['charttime'] = pd.to_datetime(fh.charttime)
fer = fer.merge(fh.drop_duplicates(['subject_id', 'charttime']), on=['subject_id', 'charttime'], how='left')
FLOW = {'ferritin_results': int(len(fer)), 'ferritin_patients': int(fer.subject_id.nunique()), 'complete_cbc_panels': int(len(cbc))}
crp = lab.loc[lab['var'] == 'CRP', ['subject_id', 'charttime', 'valuenum']].rename(columns={'valuenum': 'CRP'})
crp['subject_id'] = crp['subject_id'].astype('int64')

def nearest(left, right, cols, tol='24h'):
    left = left.sort_values('charttime'); right = right.sort_values('charttime')
    out = pd.merge_asof(left, right[['subject_id', 'charttime'] + cols].rename(columns={'charttime': 't_r'}),
                        left_on='charttime', right_on='t_r', by='subject_id', direction='nearest',
                        tolerance=pd.Timedelta(tol))
    return out

pairs = nearest(fer, cbc, core + (['MPV'] if HAS_MPV else []))
pairs = pairs.dropna(subset=core)
FLOW['ferritin_results_with_cbc_24h'] = int(len(pairs))
pairs['gap_h'] = (pairs.t_r - pairs.charttime).abs().dt.total_seconds() / 3600
if len(crp):
    pairs = nearest(pairs.drop(columns='t_r'), crp, ['CRP'])
for v in ['IRON', 'TIBC']:
    x = lab.loc[lab['var'] == v, ['subject_id', 'charttime', 'valuenum']].rename(columns={'valuenum': v})
    x['subject_id'] = x['subject_id'].astype('int64')
    if len(x):
        pairs = nearest(pairs.drop(columns=[c for c in ['t_r'] if c in pairs]), x, [v])
    else:
        pairs[v] = np.nan
pairs = pairs.sort_values('charttime').drop_duplicates('subject_id')   # first pair per patient
FLOW['patients_first_pair'] = int(len(pairs))

pat = pd.read_csv(F_PAT)
pairs = pairs.merge(pat[[c for c in ['subject_id', 'gender', 'anchor_age', 'anchor_year', 'anchor_year_group'] if c in pat]], on='subject_id')
pairs['age'] = pairs.anchor_age + (pairs.charttime.dt.year - pairs.anchor_year)
pairs['female'] = (pairs.gender == 'F').astype(int)
pairs = pairs[(pairs.age >= 18) & (pairs.ferritin > 0)]
FLOW['analysed_age18_plus'] = int(len(pairs))
say(f'MIMIC CBC+ferritin pairs (one per patient, |gap|<=24 h): {len(pairs):,}')
say('FLOW', FLOW)

# ------------------------------------------------------------------ 5. data and model
F = ['age', 'female', 'WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW_CV', 'PLT']
nh = pd.read_csv(NHANES_CSV).dropna(subset=F + ['ferritin'])
sys.path.insert(0, os.path.join(HERE, '..', 'analysis'))
from ironcbc import load_model, predict_raw
from scipy.optimize import brentq
M = load_model(MODEL_JSON)
mi = pairs.reset_index(drop=True)
rng = np.random.default_rng(2026)
RES['flow'] = FLOW

def q(x, r=1): x = pd.Series(x).dropna(); return [round(float(x.median()), r), round(float(x.quantile(.25)), r), round(float(x.quantile(.75)), r)]
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def auc_ci(y, s, B=1000):
    a = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() < y[i].max(): a.append(roc_auc_score(y[i], s[i]))
    return [round(roc_auc_score(y, s), 3), round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
def calib(y, p):
    l = logit(p); slope = LogisticRegression(C=1e6).fit(l.reshape(-1, 1), y).coef_[0][0]
    citl = brentq(lambda a: (y - 1 / (1 + np.exp(-(a + l)))).sum(), -10, 10)   # intercept with slope fixed at 1 (offset)
    return [float(slope), float(citl), float(y.mean() / p.mean()), float(brier_score_loss(y, p))]
def calib_ci(y, p, B=200):
    s0 = calib(y, p); bs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y)); bs.append(calib(y[i], p[i]))
    bs = np.array(bs)
    return {k: [round(s0[j], 3), round(float(np.percentile(bs[:, j], 2.5)), 3), round(float(np.percentile(bs[:, j], 97.5)), 3)] for j, k in enumerate(['slope', 'citl', 'oe', 'brier'])}
def rule(d):
    low = d.HGB < np.where((d.female == 1) | (d.age < 15), 12.0, 13.0)
    return (low | (d.MCV < 80)).values
def stats(pred, y):
    tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum(); tn = (~pred & (y == 0)).sum()
    return dict(sens=tp / (tp + fn), spec=tn / (tn + fp), tests=100 * pred.mean(), ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1), missed=100 * fn / len(y))
def cv_flags(score, y, r, target=None, reps=50, seed0=0):
    X = score.reshape(-1, 1); F_ = np.zeros((reps, len(y)), bool)
    for rep in range(reps):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed0 + rep).split(X, y):
            qq = LogisticRegression(C=1e6).fit(X[a], y[a]).predict_proba(X)[:, 1]
            st = r[a][y[a] == 1].mean() if target is None else target
            qs = np.sort(qq[a][y[a] == 1])[::-1]; t = qs[int(np.ceil(st * len(qs))) - 1]
            F_[rep, b] = qq[b] >= t
    return F_
def summ(F_, y):
    S = [stats(f, y) for f in F_]; return {k: round(float(np.mean([s[k] for s in S])), 4) for k in S[0]}
def oof_recal(score, y, reps=10):
    o = np.zeros(len(y)); X = score.reshape(-1, 1)
    for rep in range(reps):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            o[b] += LogisticRegression(C=1e6).fit(X[a], y[a]).predict_proba(X[b])[:, 1] / reps
    return o
def indices(d):
    return {'Mentzer': d.MCV / d.RBC, 'England-Fraser': d.MCV - d.RBC - 5 * d.HGB - 3.4, 'Srivastava': d.MCH / d.RBC,
            'Shine-Lal': d.MCV ** 2 * d.MCH / 100, 'Green-King': d.MCV ** 2 * d.RDW_CV / (100 * d.HGB),
            'RDWI': d.MCV * d.RDW_CV / d.RBC, 'Sirdah': d.MCV - d.RBC - 3 * d.HGB, 'Ehsani': d.MCV - 10 * d.RBC,
            'Ricerca': d.RDW_CV / d.RBC, 'Bessman RDW': d.RDW_CV, 'Hb (lower=ID)': -d.HGB}

def full(name, d, y, quick=False):
    """All performance measures of IronCBC in cohort d with outcome y."""
    p = predict_raw(M, d); lp = logit(p); r = rule(d); o = {'n': int(len(y)), 'events': int(y.sum())}
    o['auc'] = auc_ci(y, p, 300 if quick else 1000); o['calibration'] = calib_ci(y, p, 100 if quick else 200)
    o['rule'] = {k: round(float(v), 4) for k, v in stats(r, y).items()}
    o['model_at_rule_sens'] = summ(cv_flags(lp, y, r, reps=10 if quick else 50), y)
    if not quick:
        o['hb_at_rule_sens'] = summ(cv_flags(-d.HGB.values, y, r), y)
        for t in (0.90, 0.95): o[f'model_at_sens_{int(t*100)}'] = summ(cv_flags(lp, y, r, target=t), y)
        diffs = []
        for b in range(200):
            i = rng.integers(0, len(y), len(y)); f = cv_flags(lp[i], y[i], r[i], reps=1, seed0=b)[0]
            sm, sr = stats(f, y[i]), stats(r[i], y[i]); diffs.append([sm['tests'] - sr['tests'], sm['spec'] - sr['spec']])
        diffs = np.array(diffs)
        o['model_minus_rule_CI'] = {'tests_per100': [round(float(x), 1) for x in np.percentile(diffs[:, 0], [2.5, 97.5])],
                                    'spec': [round(float(x), 3) for x in np.percentile(diffs[:, 1], [2.5, 97.5])]}
        gq = pd.qcut(p, 10, labels=False, duplicates='drop')
        o['calibration_deciles'] = pd.DataFrame({'pred': p, 'obs': y}).groupby(gq).agg(pred=('pred', 'mean'), obs=('obs', 'mean'), n=('obs', 'size')).round(4).to_dict('list')
        pm, ph = oof_recal(lp, y), oof_recal(-d.HGB.values, y)
        gq2 = pd.qcut(pm, 10, labels=False, duplicates='drop')
        o['recal_calibration_deciles'] = pd.DataFrame({'pred': pm, 'obs': y}).groupby(gq2).agg(pred=('pred', 'mean'), obs=('obs', 'mean'), n=('obs', 'size')).round(4).to_dict('list')
        nb = lambda fl, t: float(((fl & (y == 1)).sum() - (fl & (y == 0)).sum() * t / (1 - t)) / len(y))
        o['dca'] = [{'t': round(t, 2), 'model': round(nb(pm >= t, t), 4), 'hb': round(nb(ph >= t, t), 4), 'rule': round(nb(r, t), 4), 'all': round(nb(np.ones(len(y), bool), t), 4)} for t in np.arange(0.05, 0.71, 0.05)]
    say(f'[{name}] n={o["n"]} events={o["events"]} AUC={o["auc"]} cal={o["calibration"]} rule={o["rule"]} model={o["model_at_rule_sens"]}')
    return o

# ------------------------------------------------------------------ 6. descriptives (Table 1)
say('\n=== Descriptives ===')
an = mi.HGB < np.where(mi.female == 1, 12.0, 13.0)
D = {'n': int(len(mi)), 'age': q(mi.age, 0), 'women_n': int(mi.female.sum()), 'women_pct': round(float(mi.female.mean() * 100), 1),
     'ferritin': q(mi.ferritin), 'fer_le30_n': int((mi.ferritin <= 30).sum()), 'fer_lt15_n': int((mi.ferritin < 15).sum()),
     'anaemia_n': int(an.sum()), 'anaemia_pct': round(float(an.mean() * 100), 1), 'mcv_lt80_n': int((mi.MCV < 80).sum()), 'mcv_lt80_pct': round(float((mi.MCV < 80).mean() * 100), 1),
     'HGB': q(mi.HGB), 'MCV': q(mi.MCV), 'MCH': q(mi.MCH), 'RDW_CV': q(mi.RDW_CV), 'RBC': q(mi.RBC, 2), 'PLT': q(mi.PLT, 0), 'WBC': q(mi.WBC),
     'gap_h': q(mi.gap_h), 'crp_available_n': int(mi.CRP.notna().sum()) if 'CRP' in mi else 0,
     'crp_gt5_n': int((mi.CRP > 5).sum()) if 'CRP' in mi else 0,
     'tsat_available_n': int((mi.IRON.notna() & mi.TIBC.notna()).sum()),
     'inpatient_n': int(mi.hadm_id.notna().sum()), 'outpatient_or_ed_n': int(mi.hadm_id.isna().sum())}
if 'anchor_year_group' in mi: D['anchor_year_group'] = mi.anchor_year_group.value_counts().sort_index().to_dict()
RES['descriptives'] = D; say(D)

# ------------------------------------------------------------------ 7. primary outcome, full measures
y30 = (mi.ferritin <= 30).values.astype(int)
say('\n=== A. IronCBC -> MIMIC, ferritin <=30 (full) ===')
RES['A_full'] = full('MIMIC <=30', mi, y30)
p = predict_raw(M, mi)
RES['A_full']['indices_auc_B1000'] = {k: auc_ci(y30, v.values) for k, v in indices(mi).items()}

say('\n=== B. Comparator models ===')
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(nh[F], (nh.ferritin <= 30).astype(int))
RES['B_nhanes_logistic_auc'] = auc_ci(y30, lr.predict_proba(mi[F])[:, 1])
D2 = ['age', 'female']
RES['B_age_sex_nhanes_logistic_auc'] = auc_ci(y30, LogisticRegression(max_iter=3000).fit(nh[D2], (nh.ferritin <= 30).astype(int)).predict_proba(mi[D2])[:, 1])
oof = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), mi[F], y30, cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
RES['B_local_logistic_cv_auc'] = round(float(roc_auc_score(y30, oof)), 3)
dd = []
for _ in range(1000):
    i = rng.integers(0, len(y30), len(y30)); dd.append(roc_auc_score(y30[i], p[i]) - roc_auc_score(y30[i], -mi.HGB.values[i]))
RES['B_auc_diff_model_minus_hb'] = [round(float(roc_auc_score(y30, p) - roc_auc_score(y30, -mi.HGB.values)), 3)] + [round(float(x), 3) for x in np.percentile(dd, [2.5, 97.5])]
say(RES['B_nhanes_logistic_auc'], RES['B_age_sex_nhanes_logistic_auc'], RES['B_local_logistic_cv_auc'], RES['B_auc_diff_model_minus_hb'])

say('\n=== C. Subgroups (AUC) ===')
sub = {'women': mi.female == 1, 'men': mi.female == 0, 'age_18_49': mi.age < 50, 'age_50_69': (mi.age >= 50) & (mi.age < 70), 'age_70_plus': mi.age >= 70,
       'inpatient': mi.hadm_id.notna(), 'outpatient_or_ed': mi.hadm_id.isna(), 'anaemic': an, 'non_anaemic': ~an, 'mcv_lt80': mi.MCV < 80}
RES['C_subgroups'] = {}
for k, m in sub.items():
    m = m.values
    if m.sum() > 200 and 20 < y30[m].sum() < m.sum() - 20:
        RES['C_subgroups'][k] = {'n': int(m.sum()), 'events': int(y30[m].sum()), 'auc': auc_ci(y30[m], p[m], 500),
                                 'oe': round(float(y30[m].mean() / p[m].mean()), 2), 'hb_auc': auc_ci(y30[m], -mi.HGB.values[m], 500)}
        say(k, RES['C_subgroups'][k])

say('\n=== D. Inflammation-aware reference standards (ASH 2026) ===')
tsat = 100 * mi.IRON / mi.TIBC
mT = (tsat.notna() & (tsat > 0) & (tsat <= 100)).values
if mT.sum() > 300:
    d = mi[mT].reset_index(drop=True); ts = tsat[mT].values
    yI = ((d.ferritin <= 30) | ((d.ferritin < 100) & (ts < 20))).values.astype(int)
    RES['D1_tsat_subset_ferritin_le30'] = full('TSAT subset, ferritin<=30', d, (d.ferritin <= 30).values.astype(int), quick=True)
    RES['D1_tsat_subset_ash_inflammation'] = full('TSAT subset, ferritin<=30 or (ferritin<100 and TSAT<20%)', d, yI, quick=True)
if 'CRP' in mi:
    mC = (mi.CRP > 5).values
    if mC.sum() > 300:
        d = mi[mC].reset_index(drop=True)
        RES['D2_crp_gt5_ferritin_lt100'] = full('CRP>5, ferritin<100', d, (d.ferritin < 100).values.astype(int), quick=True)
        RES['D2_crp_gt5_ferritin_le30'] = full('CRP>5, ferritin<=30', d, (d.ferritin <= 30).values.astype(int), quick=True)
    mN = (mi.CRP <= 5).values
    if mN.sum() > 300:
        d = mi[mN].reset_index(drop=True)
        RES['D3_crp_le5_full'] = full('CRP<=5, ferritin<=30', d, (d.ferritin <= 30).values.astype(int), quick=True)


say('\n=== E. Reviewer round 3 ===')
def diff_ci(d, y, B=200):
    """Bootstrap CI of model-minus-rule difference in flagged per 100 (whole procedure, one CV repeat per resample)."""
    p_ = predict_raw(M, d); lp_ = logit(p_); r_ = rule(d); out = []
    for b in range(B):
        i = rng.integers(0, len(y), len(y)); f = cv_flags(lp_[i], y[i], r_[i], reps=1, seed0=b)[0]
        out.append(stats(f, y[i])['tests'] - stats(r_[i], y[i])['tests'])
    return [round(float(x), 1) for x in np.percentile(out, [2.5, 97.5])]
tsat_all = 100 * mi.IRON / mi.TIBC
mT = (tsat_all.notna() & (tsat_all > 0) & (tsat_all <= 100)).values
if mT.sum() > 300:
    d = mi[mT].reset_index(drop=True); ts = tsat_all[mT].values
    yI = ((d.ferritin <= 30) | ((d.ferritin < 100) & (ts < 20))).values.astype(int)
    RES['E1_tsat_ash_diff_CI'] = diff_ci(d, yI); say('E1', RES['E1_tsat_ash_diff_CI'])
if 'CRP' in mi:
    mC = (mi.CRP > 5).values
    d = mi[mC].reset_index(drop=True)
    RES['E2_crp_gt5_fer100_diff_CI'] = diff_ci(d, (d.ferritin < 100).values.astype(int)); say('E2', RES['E2_crp_gt5_fer100_diff_CI'])
    mCT = mC & mT
    if mCT.sum() > 300:
        d = mi[mCT].reset_index(drop=True); ts = tsat_all[mCT].values
        yL = ((d.ferritin < 100) | (ts < 20)).values.astype(int)   # literal ASH definition in inflammation
        RES['E3_crp_gt5_literal_ash'] = full('CRP>5 with TSAT, ferritin<100 OR TSAT<20%', d, yL, quick=True)
        RES['E3_crp_gt5_literal_ash']['diff_CI'] = diff_ci(d, yL)
        say('E3 prevalence', round(float(yL.mean()), 3), RES['E3_crp_gt5_literal_ash']['diff_CI'])
# calibration after recalibration by care setting: one hospital-wide recalibration vs setting-specific
lp_all = logit(predict_raw(M, mi)); pm_all = oof_recal(lp_all, y30)
RES['E4_setting_recal'] = {}
for k, m in {'inpatient': mi.hadm_id.notna().values, 'outpatient_or_ed': mi.hadm_id.isna().values}.items():
    ps = oof_recal(lp_all[m], y30[m])
    RES['E4_setting_recal'][k] = {'oe_raw': round(float(y30[m].mean() / predict_raw(M, mi[m]).mean()), 3),
                                  'oe_after_hospital_wide_recal': round(float(y30[m].mean() / pm_all[m].mean()), 3),
                                  'oe_after_setting_recal': round(float(y30[m].mean() / ps.mean()), 3),
                                  'prevalence': round(float(y30[m].mean()), 3)}
say('E4', RES['E4_setting_recal'])
RES['runtime_min'] = round((time.time() - T0) / 60, 1)
with open(os.path.join(HERE, 'results_mimic_extra.json'), 'w') as fh: json.dump(RES, fh, indent=1, default=float)
with open(OUT_TXT, 'w', encoding='utf-8') as fh: fh.write('\n'.join(LOG))
say(f'\nDONE in {RES["runtime_min"]} min. Send results_mimic_extra.json (aggregate numbers only).')
