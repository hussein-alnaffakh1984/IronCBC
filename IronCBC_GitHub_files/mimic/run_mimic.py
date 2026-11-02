"""
CBC -> iron deficiency: MIMIC-IV analysis (run locally; MIMIC rows never leave this computer).

Usage (Windows CMD), from this folder:
    python run_mimic.py "C:\\path\\to\\mimic-iv-3.1"
The path is the folder that contains the "hosp" folder.
Output: results_mimic.txt and results_mimic.json (aggregate numbers only).
"""
import os, sys, json, time, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON = os.path.join(HERE, '..', 'model', 'ironcbc_model.json')
NHANES_CSV = os.path.join(HERE, '..', 'data', 'nhanes_train.csv')
LOCAL_CSV = os.path.join(HERE, '..', 'data', 'local_lab.csv')   # optional, not distributed
OUT_TXT = os.path.join(HERE, 'results_mimic.txt')
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
keep = []
n = 0
for ch in pd.read_csv(F_LAB, usecols=['subject_id', 'specimen_id', 'itemid', 'charttime', 'valuenum'],
                      chunksize=CHUNK):
    ch = ch[ch.itemid.isin(ID2VAR.keys()) & ch.valuenum.notna()]
    keep.append(ch)
    n += 1
    say(f'  chunk {n} done, kept so far {sum(len(k) for k in keep):,} rows, {time.time()-T0:.0f}s')
lab = pd.concat(keep, ignore_index=True)
lab['var'] = lab.itemid.map(ID2VAR)
lab['charttime'] = pd.to_datetime(lab.charttime)
say('Kept rows by variable:', lab['var'].value_counts().to_dict())

# ------------------------------------------------------------------ 4. CBC panels and ferritin pairs
cbcv = ['HGB', 'HCT', 'RBC', 'WBC', 'MCV', 'MCH', 'MCHC', 'RDW_CV', 'PLT', 'MPV']
cb = lab[lab['var'].isin(cbcv)]
cbc = cb.pivot_table(index=['subject_id', 'specimen_id', 'charttime'], columns='var', values='valuenum', aggfunc='first').reset_index()
core = ['HGB', 'HCT', 'RBC', 'WBC', 'MCV', 'MCH', 'MCHC', 'RDW_CV', 'PLT']
cbc = cbc.dropna(subset=core)
HAS_MPV = 'MPV' in cbc.columns and cbc['MPV'].notna().mean() > 0.8
say(f'Complete CBC panels: {len(cbc):,}; MPV available: {HAS_MPV}')
fer = lab[lab['var'] == 'ferritin'][['subject_id', 'charttime', 'valuenum']].rename(columns={'valuenum': 'ferritin'})
crp = lab[lab['var'] == 'CRP'][['subject_id', 'charttime', 'valuenum']].rename(columns={'valuenum': 'CRP'})

def nearest(left, right, cols, tol='24h'):
    left = left.sort_values('charttime'); right = right.sort_values('charttime')
    out = pd.merge_asof(left, right[['subject_id', 'charttime'] + cols].rename(columns={'charttime': 't_r'}),
                        left_on='charttime', right_on='t_r', by='subject_id', direction='nearest',
                        tolerance=pd.Timedelta(tol))
    return out

pairs = nearest(fer, cbc, core + (['MPV'] if HAS_MPV else []))
pairs = pairs.dropna(subset=core)
pairs['gap_h'] = (pairs.t_r - pairs.charttime).abs().dt.total_seconds() / 3600
if len(crp):
    pairs = nearest(pairs.drop(columns='t_r'), crp, ['CRP'])
pairs = pairs.sort_values('charttime').drop_duplicates('subject_id')   # first pair per patient

pat = pd.read_csv(F_PAT)
pairs = pairs.merge(pat[['subject_id', 'gender', 'anchor_age', 'anchor_year']], on='subject_id')
pairs['age'] = pairs.anchor_age + (pairs.charttime.dt.year - pairs.anchor_year)
pairs['female'] = (pairs.gender == 'F').astype(int)
pairs = pairs[(pairs.age >= 18) & (pairs.ferritin > 0)]
say(f'MIMIC CBC+ferritin pairs (one per patient, |gap|<=24 h): {len(pairs):,}')

# ------------------------------------------------------------------ 5. common features and data
F = ['age', 'female', 'WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW_CV', 'PLT'] + (['MPV'] if HAS_MPV else [])
RES['features'] = F
nh = pd.read_csv(NHANES_CSV).dropna(subset=F + ['ferritin'])
sys.path.insert(0, os.path.join(HERE, '..', 'analysis'))
from ironcbc import load_model, predict_raw
IRONCBC = load_model(MODEL_JSON)
F = IRONCBC['features']  # final model: 11 features, no MPV
nj = pd.read_csv(LOCAL_CSV) if os.path.exists(LOCAL_CSV) else None
mi = pairs
rng = np.random.default_rng(2026)

def lab30(d): return (d.ferritin <= 30).values.astype(int)
def lab15(d): return (d.ferritin < 15).values.astype(int)
def rf(): return RandomForestClassifier(n_estimators=100, min_samples_leaf=40, n_jobs=-1, random_state=0)
def auc_ci(y, s, B=1000):
    a = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() < y[i].max(): a.append(roc_auc_score(y[i], s[i]))
    return [round(roc_auc_score(y, s), 3), round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
def slope_oe(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4); lp = np.log(p / (1 - p)).reshape(-1, 1)
    s = LogisticRegression(C=1e6).fit(lp, y).coef_[0][0]
    return round(float(s), 2), round(float(y.mean() / p.mean()), 2)
def rule(d):
    low = d.HGB < np.where((d.female == 1) | (d.age < 15), 12.0, 13.0)  # WHO anaemia
    return (low | (d.MCV < 80)).values
def recal(y, p, r, reps=50):
    p = np.clip(p, 1e-4, 1 - 1e-4); lp = np.log(p / (1 - p)).reshape(-1, 1); se = []; sp = []; fl = []
    for rep in range(reps):
        pred = np.zeros(len(y), bool)
        for a, b in StratifiedKFold(5, shuffle=True, random_state=rep).split(lp, y):
            q = LogisticRegression(C=1e6).fit(lp[a], y[a]).predict_proba(lp)[:, 1]
            st = r[a][y[a] == 1].mean(); ts = np.sort(np.unique(q[a]))[::-1]
            t = next(t for t in ts if (q[a][y[a] == 1] >= t).mean() >= st); pred[b] = q[b] >= t
        se.append(pred[y == 1].mean()); sp.append((~pred[y == 0]).mean()); fl.append(pred.mean())
    return dict(sens=round(float(np.mean(se)), 3), spec=round(float(np.mean(sp)), 3), tests_per100=round(float(np.mean(fl)) * 100, 1))
def indices(d):
    return {'Mentzer': d.MCV / d.RBC, 'England-Fraser': d.MCV - d.RBC - 5 * d.HGB - 3.4, 'Srivastava': d.MCH / d.RBC,
            'Shine-Lal': d.MCV ** 2 * d.MCH / 100, 'Green-King': d.MCV ** 2 * d.RDW_CV / (100 * d.HGB),
            'RDWI': d.MCV * d.RDW_CV / d.RBC, 'Sirdah': d.MCV - d.RBC - 3 * d.HGB, 'Ehsani': d.MCV - 10 * d.RBC,
            'Ricerca': d.RDW_CV / d.RBC, 'Bessman RDW': d.RDW_CV, 'Hb (lower=ID)': -d.HGB}

def evaluate(name, d, p):
    out = {}
    for lab, f in [('fer<=30', lab30), ('fer<15', lab15)]:
        y = f(d); r = rule(d)
        o = dict(n=int(len(y)), events=int(y.sum()), auc=auc_ci(y, p[lab]))
        o['calib_slope'], o['O_over_E'] = slope_oe(y, p[lab])
        o['brier'] = round(float(brier_score_loss(y, p[lab])), 3)
        o['rule'] = dict(sens=round(float(r[y == 1].mean()), 3), spec=round(float((~r[y == 0]).mean()), 3), tests_per100=round(float(r.mean()) * 100, 1))
        o['recalibrated_model_at_rule_sens'] = recal(y, p[lab], r)
        if lab == 'fer<=30':
            o['indices_auc'] = {k: auc_ci(y, v.values, B=300) for k, v in indices(d).items()}
        out[lab] = o
        say(f'[{name}] {lab}: n={o["n"]} events={o["events"]} AUC={o["auc"]} slope={o["calib_slope"]} O/E={o["O_over_E"]} | rule {o["rule"]} | recal {o["recalibrated_model_at_rule_sens"]}')
    return out

def fit_pred(train, tests, use_final=False):
    P = {t: {} for t in tests}
    for lab, f in [('fer<=30', lab30), ('fer<15', lab15)]:
        if use_final and lab == 'fer<=30':
            for t, d in tests.items(): P[t][lab] = predict_raw(IRONCBC, d)
            continue
        m = rf().fit(train[F], f(train))
        for t, d in tests.items(): P[t][lab] = m.predict_proba(d[F])[:, 1]
    return P

# ------------------------------------------------------------------ 6. analyses
say('\n=== Cohort ===')
RES['cohort'] = dict(n=int(len(mi)), female_pct=round(float(mi.female.mean()) * 100, 1), age_median=float(mi.age.median()),
                     fer_le30_pct=round(float((mi.ferritin <= 30).mean()) * 100, 1), fer_lt15_pct=round(float((mi.ferritin < 15).mean()) * 100, 1),
                     crp_available_pct=round(float(mi['CRP'].notna().mean()) * 100, 1) if 'CRP' in mi else 0.0,
                     gap_h_median=round(float(mi.gap_h.median()), 1))
say(RES['cohort'])

say('\n=== A. IronCBC (final exported model) -> MIMIC ===')
tests = {'mimic': mi} if nj is None else {'mimic': mi, 'najaf': nj}
P = fit_pred(nh, tests, use_final=True)
RES['A_ironcbc_to_mimic'] = evaluate('IronCBC->MIMIC', mi, P['mimic'])
if nj is not None: RES['A2_ironcbc_to_local'] = evaluate('IronCBC->local', nj, P['najaf'])

say('\n=== B. MIMIC internal (5-fold CV) ===')
RES['B_mimic_internal_cv_auc'] = {}
for lab, f in [('fer<=30', lab30), ('fer<15', lab15)]:
    y = f(mi); cv = cross_val_predict(rf(), mi[F], y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
    RES['B_mimic_internal_cv_auc'][lab] = round(float(roc_auc_score(y, cv)), 3); say(lab, RES['B_mimic_internal_cv_auc'][lab])

if nj is not None:
    say('\n=== C. MIMIC-trained model -> local lab ===')
    P2 = fit_pred(mi, {'najaf': nj})
    RES['C_mimic_to_local'] = evaluate('MIMIC->local', nj, P2['najaf'])

if 'CRP' in mi and mi['CRP'].notna().sum() > 500:
    say('\n=== D. Sensitivity: MIMIC patients with CRP <= 5 mg/L only ===')
    sub = mi[mi.CRP.notna() & (mi.CRP <= 5)]
    if len(sub) > 300 and lab30(sub).sum() > 50:
        P3 = fit_pred(nh, {'sub': sub}, use_final=True)
        RES['D_ironcbc_to_mimic_crp_le5'] = evaluate('IronCBC->MIMIC CRP<=5', sub, P3['sub'])

RES['runtime_min'] = round((time.time() - T0) / 60, 1)
with open(os.path.join(HERE, 'results_mimic.json'), 'w') as fh: json.dump(RES, fh, indent=1)
with open(OUT_TXT, 'w', encoding='utf-8') as fh: fh.write('\n'.join(LOG))
say(f'\nDONE in {RES["runtime_min"]} min. Send results_mimic.json and results_mimic.txt (aggregate numbers only).')
