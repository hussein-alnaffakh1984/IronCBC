"""Shared helpers: load the exported IronCBC model and score CBC rows exactly as the web app does."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, '..', 'model', 'ironcbc_model.json')


def load_model(path=DEFAULT_MODEL):
    with open(path) as fh:
        m = json.load(fh)
    for t in m['trees']:
        for k in ('f', 'l', 'r'):
            t[k] = np.asarray(t[k], dtype=np.int64)
        t['t'] = np.asarray(t['t'], dtype=np.float64)
        t['p'] = np.asarray(t['p'], dtype=np.float64)
    return m


def predict_raw(model, df):
    """Uncalibrated probability of ferritin <= 30 ng/mL. df needs model['features'] columns."""
    X = df[model['features']].to_numpy(dtype=np.float32).astype(np.float64)  # same rounding as scikit-learn / JS Math.fround
    out = np.zeros(len(X))
    rows = np.arange(len(X))
    for t in model['trees']:
        node = np.zeros(len(X), dtype=np.int64)
        while True:
            internal = t['l'][node] != -1
            if not internal.any():
                break
            n = node[internal]
            go_left = X[rows[internal], t['f'][n]] <= t['t'][n]
            node[internal] = np.where(go_left, t['l'][n], t['r'][n])
        out += t['p'][node]
    return out / len(model['trees'])


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def apply_calibration(p_raw, cal):
    return 1 / (1 + np.exp(-(cal['intercept'] + cal['slope'] * logit(p_raw))))


def clinical_rule(df):
    """Low Hb (women <12, men <13 g/dL) or MCV < 80 fL."""
    low = np.where(df['female'] == 1, df['HGB'] < 12, df['HGB'] < 13)
    return np.asarray(low | (df['MCV'] < 80))


def classic_indices(d):
    return {'Mentzer': d.MCV / d.RBC, 'England-Fraser': d.MCV - d.RBC - 5 * d.HGB - 3.4, 'Srivastava': d.MCH / d.RBC,
            'Shine-Lal': d.MCV ** 2 * d.MCH / 100, 'Green-King': d.MCV ** 2 * d.RDW_CV / (100 * d.HGB),
            'RDWI': d.MCV * d.RDW_CV / d.RBC, 'Sirdah': d.MCV - d.RBC - 3 * d.HGB, 'Ehsani': d.MCV - 10 * d.RBC,
            'Ricerca': d.RDW_CV / d.RBC, 'Bessman RDW': d.RDW_CV, 'Hb (lower = ID)': -d.HGB}
