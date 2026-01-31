"""Threshold analysis for new model"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')
from sklearn.metrics import f1_score
from src.models.mpbpk_ml import mPBPKClassifier

model_path = Path('models/mpbpk_classifier_20260127_140004.pkl')
classifier = mPBPKClassifier.load(model_path)

df = pd.read_csv('data/processed/drugbank_pk_parameters.csv')
df['is_mab'] = df['name'].str.lower().str.endswith('mab')
antibodies = df[df['is_mab']].copy()

def get_label(g):
    if pd.isna(g): return None
    if 'withdrawn' in g.lower(): return 0
    if 'approved' in g.lower(): return 1
    return None

antibodies['label'] = antibodies['groups'].apply(get_label)
ab_labeled = antibodies[antibodies['label'].notna() & antibodies['half_life_value'].notna() & antibodies['average_dose_mg'].notna()].copy()

results = []
for _, row in ab_labeled.iterrows():
    hl = row['half_life_value']
    if hl <= 0 or hl > 20000: continue
    dose_val = row['average_dose_mg'] / 70.0
    
    X = []
    for fn in classifier.feature_names:
        if fn == 'log_halflife':
            X.append(np.log10(max(hl, 0.1)))
        elif fn == 'log_dose':
            X.append(np.log10(max(dose_val, 0.01)))
        elif fn == 'log_KD':
            X.append(0)
        elif fn == 'log_MW':
            X.append(np.log10(150))
        elif fn == 'log_T0':
            X.append(np.log10(5))
        elif fn.startswith('pop_EUR'):
            X.append(1)
        elif fn.startswith('pop_'):
            X.append(0)
        elif fn == 'phenotype_encoded':
            X.append(2)
        else:
            X.append(0)
    
    X = np.array(X).reshape(1, -1)
    prob = classifier.rf_model.predict_proba(X)[0][1]
    results.append({'label': int(row['label']), 'prob': prob})

res_df = pd.DataFrame(results)

print("Threshold | Approved | Withdrawn | Accuracy | F1")
print("-" * 55)

for thresh in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    res_df['pred'] = (res_df['prob'] >= thresh).astype(int)
    
    ap_mask = res_df['label'] == 1
    wd_mask = res_df['label'] == 0
    
    ap_rate = (res_df.loc[ap_mask, 'pred'] == 1).mean()
    wd_detect = (res_df.loc[wd_mask, 'pred'] == 0).mean()
    acc = (res_df['pred'] == res_df['label']).mean()
    f1 = f1_score(res_df['label'], res_df['pred'], zero_division=0)
    
    marker = ""
    if thresh == 0.30:
        marker = " <-- Balanced"
    elif thresh == 0.50:
        marker = " <-- Current"
    
    print(f"  {thresh:.2f}   |  {ap_rate:5.1%}  |   {wd_detect:5.1%}   |  {acc:5.1%}  | {f1:.3f}{marker}")
