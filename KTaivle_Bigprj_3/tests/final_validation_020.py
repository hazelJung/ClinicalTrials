"""Final validation with threshold 0.20"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from src.models.mpbpk_ml import mPBPKClassifier

THRESHOLD = 0.20

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
    pred = 1 if prob >= THRESHOLD else 0
    
    results.append({
        'name': row['name'],
        'label': int(row['label']),
        'pred': pred,
        'prob': prob,
        'dose': dose_val,
        'halflife': hl
    })

res_df = pd.DataFrame(results)

y_true = res_df['label']
y_pred = res_df['pred']

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, zero_division=0)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

print("=== FINAL VALIDATION (Threshold=0.20, True Antibodies) ===")
print(f"Model: {model_path.name}")
print(f"Decision Threshold: {THRESHOLD}")
print(f"Samples: {len(res_df)}")
print(f"Approved: {(y_true == 1).sum()}, Withdrawn: {(y_true == 0).sum()}")
print()
print("Performance Metrics:")
print(f"  Accuracy:  {acc:.1%}")
print(f"  Precision: {prec:.3f}")
print(f"  Recall:    {rec:.3f}")
print(f"  F1 Score:  {f1:.3f}")
print()
print("Confusion Matrix:")
print(f"             Pred FAIL   Pred SUCCESS")
print(f"True FAIL      {cm[0,0]:4d}        {cm[0,1]:4d}")
print(f"True SUCCESS   {cm[1,0]:4d}        {cm[1,1]:4d}")
print()
if cm[0,0] + cm[0,1] > 0:
    wd_recall = cm[0,0] / (cm[0,0] + cm[0,1])
    print(f"Withdrawn Detection Rate: {wd_recall:.1%} ({cm[0,0]}/{cm[0,0]+cm[0,1]})")
if cm[1,0] + cm[1,1] > 0:
    ap_recall = cm[1,1] / (cm[1,0] + cm[1,1])
    print(f"Approved Pass Rate: {ap_recall:.1%} ({cm[1,1]}/{cm[1,0]+cm[1,1]})")

print()
print("Detected Withdrawn Drugs:")
wd = res_df[(res_df['label'] == 0) & (res_df['pred'] == 0)]
for _, r in wd.iterrows():
    print(f"  {r['name']}: Dose={r['dose']:.2f} mg/kg, t1/2={r['halflife']:.1f}h, Prob={r['prob']:.2f}")

print()
print("Missed Withdrawn Drugs:")
missed = res_df[(res_df['label'] == 0) & (res_df['pred'] == 1)]
for _, r in missed.iterrows():
    print(f"  {r['name']}: Dose={r['dose']:.2f} mg/kg, t1/2={r['halflife']:.1f}h, Prob={r['prob']:.2f}")
