"""Train the IronCBC random forest on NHANES and export it as JSON for the web app.
Usage: python train_export.py [nhanes_csv] [out_json]
The local (Najaf) calibration block is copied from an existing model file if present;
recompute it with validate_external.py --write-calibration.
"""
import sys, os, json
import numpy as np, pandas as pd, sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'data', 'nhanes_train.csv')
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, '..', 'model', 'ironcbc_model.json')
F = ['age', 'female', 'WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW_CV', 'PLT']
PARAMS = dict(n_estimators=100, min_samples_leaf=40, random_state=0)

d = pd.read_csv(src).dropna(subset=F + ['ferritin'])
y = (d.ferritin <= 30).astype(int).values
m = RandomForestClassifier(n_jobs=-1, **PARAMS).fit(d[F], y)
cv = cross_val_predict(RandomForestClassifier(n_jobs=-1, **PARAMS), d[F], y,
                       cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
thr_nhanes = float(np.quantile(cv[y == 1], 0.20))  # 80% sensitivity in NHANES CV

trees = []
for e in m.estimators_:
    t = e.tree_; v = t.value[:, 0, :]
    trees.append({'f': t.feature.tolist(), 't': [float(x) for x in t.threshold], 'l': t.children_left.tolist(),
                  'r': t.children_right.tolist(), 'p': [round(float(x), 6) for x in v[:, 1] / v.sum(1)]})
cals = {'nhanes_uncalibrated': {'intercept': 0.0, 'slope': 1.0, 'threshold': round(thr_nhanes, 4),
                                'note': 'no recalibration; threshold gives 80% sensitivity in NHANES 5-fold cross-validation'}}
if os.path.exists(out):
    old = json.load(open(out)).get('calibrations', {})
    if 'najaf_outpatient_lab' in old: cals['najaf_outpatient_lab'] = old['najaf_outpatient_lab']
model = {'name': 'IronCBC random forest', 'version': '1.0',
         'trained_on': 'NHANES 2015-2016, 2017-March 2020, 2021-2023; age>=12, non-pregnant; n=%d' % len(d),
         'outcome': 'serum ferritin <= 30 ng/mL (ASH 2026)', 'features': F,
         'units': {'age': 'years', 'female': '1 female, 0 male', 'WBC': '10^9/L', 'RBC': '10^12/L', 'HGB': 'g/dL', 'HCT': '%',
                   'MCV': 'fL', 'MCH': 'pg', 'MCHC': 'g/dL', 'RDW_CV': '%', 'PLT': '10^9/L'},
         'sklearn': sklearn.__version__, 'params': PARAMS, 'calibrations': cals, 'trees': trees}
json.dump(model, open(out, 'w'), separators=(',', ':'))
print('NHANES n=%d, prevalence %.3f, CV threshold %.4f -> %s' % (len(d), y.mean(), thr_nhanes, out))
