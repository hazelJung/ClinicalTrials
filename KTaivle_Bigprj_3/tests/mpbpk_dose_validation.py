"""
mPBPK-ML Final Validation: Real DrugBank Dose + Half-life
==========================================================
Validates the mPBPK-ML model using real clinical data extracted from DrugBank.
Uses BOTH actual Half-life and Actual Dose (where available).

Ground Truth:
- Approved -> SUCCESS
- Withdrawn -> FAIL
"""

import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

from src.models.mpbpk_ml import mPBPKClassifier, MLConfig

def run_final_validation():
    print("=" * 60)
    print("  mPBPK-ML VALIDATION (Real Dose + Half-life)")
    print("=" * 60)
    
    # 1. Load Data
    df = pd.read_csv('data/processed/drugbank_pk_parameters.csv')
    
    # Filter: Valid Half-life AND Valid Dose
    # Note: Dose is critical. If dose is missing, model defaults to 5mg/kg which might be wrong.
    # We will test two scenarios:
    # A) Only drugs with known Dose
    # B) All drugs (impute missing dose with median of withdrawal drugs for withdrawals?) -> No, keep it clean.
    
    df_valid = df[df['half_life_value'].notna()].copy()
    
    # Labeling
    def get_label(groups):
        if pd.isna(groups): return None
        g = groups.lower()
        if 'withdrawn' in g: return 0
        elif 'approved' in g: return 1
        return None
    
    df_valid['label'] = df_valid['groups'].apply(get_label)
    df_labeled = df_valid[df_valid['label'].notna()].copy()
    
    print(f"\n[1] Data Loaded")
    print(f"    Total labeled drugs: {len(df_labeled)}")
    print(f"    - With Dose info: {df_labeled['average_dose_mg'].notna().sum()}")
    
    # 2. Load Model
    # Use the trained model from previous step
    model_dir = Path('models')
    files = list(model_dir.glob('mpbpk*.pkl'))
    if not files:
        print("Error: No model found.")
        return
    latest_model = max(files, key=lambda x: x.stat().st_mtime)
    print(f"    Loading model: {latest_model.name}")
    classifier = mPBPKClassifier.load(latest_model)
    
    # 3. Validation Loop
    results = []
    
    for _, row in df_labeled.iterrows():
        # Skip if HL is wildly out of range
        hl = row['half_life_value']
        if hl <= 0 or hl > 20000: continue
        
        # Dose Logic
        dose_mg = row['average_dose_mg']
        if pd.isna(dose_mg):
            # Imputation Strategy
            # Use 'Standard' 5 mg/kg ~ 350mg for missing, 
            # BUT for this validation we rely on real signal.
            # Let's use 5.0 as fallback but mark it.
            dose_val = 5.0
            dose_source = "imputed"
        else:
            # Convert total mg to mg/kg (assume 70kg patient)
            dose_val = dose_mg / 70.0
            dose_source = "real"
            
        # Feature Vector
        feat = {
            'log_KD': np.log10(1.0),       # Default 1 nM
            'log_dose': np.log10(dose_val), 
            'charge': 0,
            'log_MW': np.log10(150),       # 150 kDa
            'log_T0': np.log10(5.0),       # 5 nM
            'log_halflife': np.log10(hl),
            'population': 'EUR',
            'phenotype': 'NM',
            'activity_score': 2.0,
            'cl_multiplier': 1.0,
        }
        
        # Build X
        X = []
        for fn in classifier.feature_names:
            if fn in feat:
                X.append(feat[fn])
            elif fn.startswith('pop_'):
                pop = fn.replace('pop_', '')
                X.append(1 if feat.get('population') == pop else 0)
            elif fn == 'phenotype_encoded':
                X.append(2)
            else:
                X.append(0)
        
        X = np.array(X).reshape(1, -1)
        prob = classifier.rf_model.predict_proba(X)[0][1]
        pred = classifier.rf_model.predict(X)[0]
        
        results.append({
            'name': row['name'],
            'label': int(row['label']),
            'pred': int(pred),
            'prob': prob,
            'half_life': hl,
            'dose_mg_kg': dose_val,
            'dose_source': dose_source
        })
        
    # 4. Analysis
    res_df = pd.DataFrame(results)
    
    print("\n[2] Overall Results")
    y_true = res_df['label']
    y_pred = res_df['pred']
    
    print(f"    Accuracy: {accuracy_score(y_true, y_pred):.1%}")
    print(f"    F1 Score: {f1_score(y_true, y_pred, zero_division=0):.3f}")
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"    Confusion Matrix:")
    print(f"                Pred FAIL   Pred SUCCESS")
    print(f"    True FAIL      {cm[0,0]:4d}        {cm[0,1]:4d}")
    print(f"    True SUCCESS   {cm[1,0]:4d}        {cm[1,1]:4d}")
    
    # 5. Focus on Real Dose Data
    print("\n[3] Results with REAL DOSE Data Only")
    res_real = res_df[res_df['dose_source'] == 'real']
    
    if len(res_real) > 0:
        y_t = res_real['label']
        y_p = res_real['pred']
        
        cm_r = confusion_matrix(y_t, y_p, labels=[0, 1])
        fail_recall = cm_r[0,0] / (cm_r[0,0] + cm_r[0,1]) if (cm_r[0,0] + cm_r[0,1]) > 0 else 0
        
        print(f"    Samples: {len(res_real)}")
        print(f"    Failure Detection Rate (Recall 0): {fail_recall:.1%} ({cm_r[0,0]}/{cm_r[0,0] + cm_r[0,1]})")
        print(f"    Approved Pass Rate (Recall 1): {recall_score(y_t, y_p):.1%} ({cm_r[1,1]}/{cm_r[1,0] + cm_r[1,1]})")
        
        # List correctly detected failures
        print("\n    [CORRECT] Detected Failures (Real Dose):")
        detected_fails = res_real[(res_real['label'] == 0) & (res_real['pred'] == 0)]
        for _, r in detected_fails.iterrows():
             print(f"      {r['name']}: Dose={r['dose_mg_kg']:.2f} mg/kg, t1/2={r['half_life']:.1f}h -> Prob {r['prob']:.2f}")

        # List missed failures
        print("\n    [MISSED] Missed Failures (Real Dose):")
        missed_fails = res_real[(res_real['label'] == 0) & (res_real['pred'] == 1)]
        for _, r in missed_fails.iterrows():
             print(f"      {r['name']}: Dose={r['dose_mg_kg']:.2f} mg/kg, t1/2={r['half_life']:.1f}h -> Prob {r['prob']:.2f}")

    else:
        print("    No samples with real dose found.")

if __name__ == "__main__":
    run_final_validation()
