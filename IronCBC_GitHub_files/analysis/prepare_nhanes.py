"""Build the NHANES training table from the public XPT files.

Download from https://wwwn.cdc.gov/nchs/nhanes/ into one folder:
  2015-2016: DEMO_I, CBC_I, FERTIN_I
  2017-March 2020: P_DEMO, P_CBC, P_FERTIN, P_HSCRP
  2021-2023: DEMO_L, CBC_L, FERTIN_L, HSCRP_L
Usage: python prepare_nhanes.py <xpt_folder> [out_csv]
"""
import sys, os, glob
import numpy as np, pandas as pd

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), '..', 'data', 'nhanes_train.csv')

def rd(name):
    hits = [h for h in glob.glob(os.path.join(src, '*.xpt')) + glob.glob(os.path.join(src, '*.XPT'))
            if os.path.basename(h).upper().endswith(name.upper() + '.XPT')]
    return pd.read_sas(hits[0], format='xport') if hits else None

cycles = [('2015-16', 'DEMO_I', 'CBC_I', 'FERTIN_I', None),
          ('2017-20', 'P_DEMO', 'P_CBC', 'P_FERTIN', 'P_HSCRP'),
          ('2021-23', 'DEMO_L', 'CBC_L', 'FERTIN_L', 'HSCRP_L')]
parts = []
for c, de, cb, fe, cr in cycles:
    d = rd(de)[['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDEXPRG']].merge(rd(cb), on='SEQN').merge(rd(fe)[['SEQN', 'LBXFER']], on='SEQN')
    crp = rd(cr) if cr else None
    d['CRP'] = crp.set_index('SEQN').LBXHSCRP.reindex(d.SEQN).values if crp is not None else np.nan
    d['cycle'] = c
    parts.append(d)
d = pd.concat(parts, ignore_index=True).rename(columns={
    'RIDAGEYR': 'age', 'LBXWBCSI': 'WBC', 'LBXRBCSI': 'RBC', 'LBXHGB': 'HGB', 'LBXHCT': 'HCT', 'LBXMCVSI': 'MCV',
    'LBXMCHSI': 'MCH', 'LBXMC': 'MCHC', 'LBXRDW': 'RDW_CV', 'LBXPLTSI': 'PLT', 'LBXMPSI': 'MPV', 'LBXFER': 'ferritin'})
d['female'] = (d.RIAGENDR == 2).astype(int)
F = ['age', 'female', 'WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW_CV', 'PLT']
d = d[(d.age >= 12) & (d.RIDEXPRG != 1)].dropna(subset=F + ['ferritin'])
d[F + ['MPV', 'ferritin', 'CRP']].to_csv(out, index=False)
print('NHANES rows:', len(d), '->', out)
