"""
QSAR Model Validation on ClinTox Dataset
=========================================
Validates QSAR toxicity predictions against real FDA drug data.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

print("=" * 60)
print("  QSAR Model Validation Report (ClinTox Dataset)")
print("=" * 60)

# 1. Load ClinTox data
clintox_path = Path('data/raw/clintox.csv')
if not clintox_path.exists():
    print(f"ERROR: ClinTox data not found at {clintox_path}")
    sys.exit(1)

df = pd.read_csv(clintox_path)
print(f"\n[1] Dataset Summary")
print(f"    Total drugs: {len(df)}")
print(f"    FDA Approved: {df['FDA_APPROVED'].sum()} ({100*df['FDA_APPROVED'].mean():.1f}%)")
print(f"    Clinical Toxicity: {df['CT_TOX'].sum()} ({100*df['CT_TOX'].mean():.1f}%)")

# 2. Load QSAR Predictor
print(f"\n[2] Loading QSAR Models...")
from src.models.qsar_predictor import QSARPredictor

predictor = QSARPredictor(auto_load=True)
print(f"    Models loaded: {len(predictor.models)}")
print(f"    Endpoints: {list(predictor.models.keys())}")

if len(predictor.models) == 0:
    print("    ERROR: No models loaded!")
    sys.exit(1)

# 3. Run predictions
print(f"\n[3] Running Predictions...")

sample_size = min(500, len(df))
df_sample = df.sample(n=sample_size, random_state=42)

results = []
processed = 0
failed = 0

for idx, row in df_sample.iterrows():
    smiles = row['smiles']
    
    try:
        # Get predictions for all endpoints
        preds = predictor.predict_multiple_endpoints(smiles)
        
        if not preds:
            failed += 1
            continue
        
        # Check validity
        any_valid = any(p.is_valid for p in preds.values())
        if not any_valid:
            failed += 1
            continue
        
        # Count toxic endpoints
        toxic_count = sum(1 for p in preds.values() if p.is_valid and p.prediction == 1)
        total_valid = sum(1 for p in preds.values() if p.is_valid)
        
        # Risk level determination
        toxic_ratio = toxic_count / total_valid if total_valid > 0 else 0
        if toxic_ratio >= 0.5:
            risk_level = "CRITICAL"
        elif toxic_ratio >= 0.25:
            risk_level = "HIGH"
        elif toxic_ratio > 0:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        is_toxic_pred = 1 if toxic_count >= 2 else 0
        
        results.append({
            'smiles': smiles[:50],  # Truncate for display
            'fda_approved': row['FDA_APPROVED'],
            'ct_tox': row['CT_TOX'],
            'toxic_endpoints': toxic_count,
            'valid_endpoints': total_valid,
            'pred_toxic': is_toxic_pred,
            'risk_level': risk_level
        })
        processed += 1
        
    except Exception as e:
        failed += 1
        continue

print(f"    Processed: {processed}/{sample_size}")
print(f"    Failed: {failed}")

if processed == 0:
    print("    ERROR: No successful predictions!")
    sys.exit(1)

# 4. Calculate metrics
res_df = pd.DataFrame(results)

y_true = res_df['ct_tox']
y_pred = res_df['pred_toxic']

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

print(f"\n[4] Performance Metrics (Predicting Clinical Toxicity)")
print(f"    Accuracy:  {acc:.1%}")
print(f"    Precision: {prec:.3f}")
print(f"    Recall:    {rec:.3f}")
print(f"    F1 Score:  {f1:.3f}")
print()
print(f"    Confusion Matrix:")
print(f"                 Pred Safe   Pred Toxic")
print(f"    True Safe      {cm[0,0]:4d}        {cm[0,1]:4d}")
print(f"    True Toxic     {cm[1,0]:4d}        {cm[1,1]:4d}")

# 5. Risk level distribution
print(f"\n[5] Risk Level Distribution")
for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
    count = (res_df['risk_level'] == level).sum()
    print(f"    {level}: {count} ({100*count/len(res_df):.1f}%)")

# 6. Cross-tabulation with Clinical Toxicity
print(f"\n[6] Risk Level vs Clinical Toxicity")
cross = pd.crosstab(res_df['risk_level'], res_df['ct_tox'], margins=True)
cross.columns = ['Non-Toxic', 'Toxic', 'Total']
cross.index = [i if i != 'All' else 'Total' for i in cross.index]
print(cross.to_string())

# 7. Save results
output_path = Path('data/processed/qsar_clintox_validation.csv')
res_df.to_csv(output_path, index=False)
print(f"\n    Results saved to: {output_path}")

print("\n" + "=" * 60)
print("  Validation Complete")
print("=" * 60)
