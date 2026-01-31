"""
Threshold Sensitivity Analysis for mPBPK Model
==============================================
Analyzes different decision thresholds to balance
approved drug recall vs. withdrawn drug detection.
"""

import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from src.models.mpbpk_ml import mPBPKClassifier

def main():
    # Load model
    model_path = Path('models/mpbpk_classifier_20260127_130051.pkl')
    classifier = mPBPKClassifier.load(model_path)

    # Load DrugBank data
    df = pd.read_csv('data/processed/drugbank_pk_parameters.csv')
    df = df[df['half_life_value'].notna()].copy()

    def get_label(groups):
        if pd.isna(groups): return None
        g = groups.lower()
        if 'withdrawn' in g: return 0
        elif 'approved' in g: return 1
        return None

    df['label'] = df['groups'].apply(get_label)
    df_labeled = df[df['label'].notna() & df['average_dose_mg'].notna()].copy()

    # Build predictions
    results = []
    for _, row in df_labeled.iterrows():
        hl = row['half_life_value']
        if hl <= 0 or hl > 20000: continue
        
        dose_val = row['average_dose_mg'] / 70.0
        
        X = []
        for fn in classifier.feature_names:
            if fn == 'log_halflife':
                X.append(np.log10(hl))
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
        
        results.append({
            'name': row['name'],
            'label': int(row['label']),
            'prob': prob,
            'dose': dose_val,
            'halflife': hl
        })

    res_df = pd.DataFrame(results)

    # Threshold analysis
    print("Threshold Sensitivity Analysis")
    print("=" * 70)
    print(f"{'Thresh':>7} | {'Approved':>12} | {'Withdrawn':>12} | {'Accuracy':>10} | {'F1':>6}")
    print("-" * 70)

    best_thresh = 0.5
    best_f1 = 0

    for thresh in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        res_df['pred'] = (res_df['prob'] >= thresh).astype(int)
        
        approved_mask = res_df['label'] == 1
        withdrawn_mask = res_df['label'] == 0
        
        approved_recall = (res_df.loc[approved_mask, 'pred'] == 1).mean()
        withdrawn_recall = (res_df.loc[withdrawn_mask, 'pred'] == 0).mean()
        accuracy = (res_df['pred'] == res_df['label']).mean()
        f1 = f1_score(res_df['label'], res_df['pred'], zero_division=0)
        
        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            marker = " <-- Best F1"
        
        print(f"{thresh:>7.2f} | {approved_recall:>12.1%} | {withdrawn_recall:>12.1%} | {accuracy:>10.1%} | {f1:>6.3f}{marker}")

    print()
    print(f"Optimal Threshold: {best_thresh} (F1={best_f1:.3f})")
    
    # Feature analysis
    print()
    print("Feature Analysis: Why are approved drugs flagged as FAIL?")
    print("=" * 70)

    res_df['pred_default'] = (res_df['prob'] >= 0.5).astype(int)
    fp = res_df[(res_df['label'] == 1) & (res_df['pred_default'] == 0)]
    tp = res_df[(res_df['label'] == 1) & (res_df['pred_default'] == 1)]

    print(f"False Positives (Approved but FAIL): {len(fp)}")
    print(f"True Positives (Approved and SUCCESS): {len(tp)}")
    print()
    print("Feature Comparison:")
    print(f"  Dose (mg/kg):    FP mean={fp['dose'].mean():.2f}, TP mean={tp['dose'].mean():.2f}")
    print(f"  Half-life (h):   FP mean={fp['halflife'].mean():.1f}, TP mean={tp['halflife'].mean():.1f}")
    
    # Root cause
    print()
    print("Root Cause Analysis:")
    print("-" * 70)
    print("The model was trained on simulation data where:")
    print("  - Low dose + short half-life = FAIL (efficacy failure)")
    print("  - High dose + long half-life = SUCCESS")
    print()
    print("However, many approved drugs in DrugBank have low mg/kg doses because:")
    print("  1. They are highly potent (low dose needed for efficacy)")
    print("  2. Unit conversion issues (mg vs mcg, total vs per kg)")
    print("  3. Different administration routes (IV bolus vs infusion)")
    print()
    print("SOLUTION OPTIONS:")
    print("  1. Adjust threshold: Use 0.35 instead of 0.5 for balanced performance")
    print("  2. Add potency adjustment: Consider KD/EC50 to normalize dose")
    print("  3. Retrain with real data: Fine-tune on DrugBank data")

if __name__ == "__main__":
    main()
