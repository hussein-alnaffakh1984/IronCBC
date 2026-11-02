"""
IronCBC: fit the hospital (MIMIC-IV) calibration only. Faster than run_mimic.py.
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
OUT_TXT = os.path.join(HERE, 'mimic_calibration.txt')
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


# ------------------------------------------------------------------ 5. calibration
sys.path.insert(0, os.path.join(HERE, '..', 'analysis'))
from ironcbc import load_model, predict_raw, logit, clinical_rule
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
M = load_model(MODEL_JSON)
mi = pairs
p = predict_raw(M, mi); lp = logit(p).reshape(-1, 1)
y = (mi.ferritin <= 30).astype(int).values; rule = clinical_rule(mi)
r = LogisticRegression(C=1e6).fit(lp, y); q = r.predict_proba(lp)[:, 1]
st = rule[y == 1].mean(); qs = np.sort(q[y == 1])[::-1]
thr = float(qs[int(np.ceil(st * len(qs))) - 1])
pos = q >= thr
cal = {'intercept': round(float(r.intercept_[0]), 4), 'slope': round(float(r.coef_[0][0]), 4), 'threshold': round(thr, 4),
       'n': int(len(y)), 'events': int(y.sum()), 'auc': round(float(roc_auc_score(y, p)), 3),
       'sens': round(float(pos[y == 1].mean()), 3), 'spec': round(float((~pos[y == 0]).mean()), 3),
       'tests_per100': round(float(pos.mean()) * 100, 1),
       'rule': {'sens': round(float(st), 3), 'spec': round(float((~rule[y == 0]).mean()), 3), 'tests_per100': round(float(rule.mean()) * 100, 1)},
       'note': 'logistic recalibration of logit(p) on MIMIC-IV hospital patients; threshold matches clinical-rule sensitivity'}
say('MIMIC calibration:', cal)
with open(os.path.join(HERE, 'mimic_calibration.json'), 'w') as fh: json.dump(cal, fh, indent=1)
with open(OUT_TXT, 'w', encoding='utf-8') as fh: fh.write('\n'.join(LOG))
say('DONE. Send mimic_calibration.json (aggregate numbers only).')
