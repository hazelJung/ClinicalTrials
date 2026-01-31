"""
mPBPK-ML Model Validation with Real DrugBank Antibody Data
==========================================================
DrugBank에서 추출한 실제 항체 약물 데이터로 mPBPK-ML 모델 검증

Ground Truth:
- approved → 성공 (label = 1)
- withdrawn → 실패 (label = 0)
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.models.mpbpk_ml import mPBPKClassifier, MLConfig


def run_mpbpk_validation():
    """DrugBank 항체 데이터로 mPBPK-ML 검증"""
    print("=" * 60)
    print("  mPBPK-ML VALIDATION: DrugBank Antibody Data")
    print("=" * 60)
    
    # 1. Load DrugBank biotech data
    df = pd.read_csv('data/processed/drugbank_pk_parameters.csv')
    print(f"\n[1] Loaded: {len(df)} biotech drugs")
    
    # 2. Filter drugs with PK parameters
    df_valid = df[df['half_life_value'].notna()].copy()
    print(f"    With half-life: {len(df_valid)}")
    
    # 3. Create ground truth labels
    # approved (not withdrawn) = SUCCESS (1)
    # withdrawn = FAIL (0)
    def get_label(groups):
        if pd.isna(groups):
            return None
        groups_lower = groups.lower()
        if 'withdrawn' in groups_lower:
            return 0  # FAIL
        elif 'approved' in groups_lower:
            return 1  # SUCCESS
        else:
            return None  # Unknown (investigational, experimental)
    
    df_valid['label'] = df_valid['groups'].apply(get_label)
    df_labeled = df_valid[df_valid['label'].notna()].copy()
    
    print(f"    With labels (approved/withdrawn): {len(df_labeled)}")
    print(f"    - SUCCESS (approved): {(df_labeled['label'] == 1).sum()}")
    print(f"    - FAIL (withdrawn): {(df_labeled['label'] == 0).sum()}")
    
    # 4. Prepare features for mPBPK-ML
    # Need: log_KD, log_dose, log_MW, log_halflife, log_T0, etc.
    print("\n[2] Preparing features...")
    
    # Use available data with defaults for missing
    features = []
    labels = []
    names = []
    
    for _, row in df_labeled.iterrows():
        try:
            # Half-life in hours (already parsed)
            half_life_hr = row['half_life_value']
            if half_life_hr <= 0 or half_life_hr > 10000:
                continue
            
            # Average mass in Da (convert to kDa for antibodies)
            avg_mass = row.get('average_mass', 150000)  # Default 150 kDa
            if pd.isna(avg_mass):
                avg_mass = 150000
            
            # Antibody check: MW > 100 kDa
            if avg_mass < 100000:  # Skip non-antibodies
                continue
            
            mw_kda = avg_mass / 1000  # Convert to kDa
            
            # Default values (typical for antibodies)
            features.append({
                'log_KD': np.log10(1.0),  # Default 1 nM (high affinity)
                'log_dose': np.log10(5.0),  # Default 5 mg/kg
                'charge': 0,
                'log_MW': np.log10(mw_kda),
                'log_T0': np.log10(5.0),  # Default 5 nM target
                'log_halflife': np.log10(half_life_hr),
                'population': 'EUR',
                'phenotype': 'NM',
                'activity_score': 2.0,
                'cl_multiplier': 1.0,
            })
            labels.append(int(row['label']))
            names.append(row['name'])
        except:
            continue
    
    print(f"    Valid samples: {len(features)}")
    
    if len(features) < 10:
        print("    ERROR: Too few valid samples")
        return
    
    # 5. Train mPBPK-ML classifier
    print("\n[3] Training mPBPK-ML classifier...")
    classifier = mPBPKClassifier(MLConfig(use_smote=True))
    classifier.train()
    feature_names = classifier.feature_names
    
    # 6. Create feature vectors and predict
    print("\n[4] Running predictions...")
    
    predictions = []
    probabilities = []
    
    for feat in features:
        try:
            X = []
            for fn in feature_names:
                if fn in feat:
                    X.append(feat[fn])
                elif fn.startswith('pop_'):
                    pop = fn.replace('pop_', '')
                    X.append(1 if feat.get('population') == pop else 0)
                elif fn == 'phenotype_encoded':
                    pheno_map = {'PM': 0, 'IM': 1, 'NM': 2, 'UM': 3}
                    X.append(pheno_map.get(feat.get('phenotype', 'NM'), 2))
                else:
                    X.append(0)
            
            X = np.array(X).reshape(1, -1)
            pred = classifier.rf_model.predict(X)[0]
            prob = classifier.rf_model.predict_proba(X)[0][1]
            
            predictions.append(pred)
            probabilities.append(prob)
        except Exception as e:
            predictions.append(0)
            probabilities.append(0.5)
    
    # 7. Evaluate
    print("\n[5] Evaluation Results")
    print("-" * 60)
    
    y_true = np.array(labels)
    y_pred = np.array(predictions)
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n📊 mPBPK-ML vs Real Clinical Status")
    print(f"    Samples: {len(y_true)}")
    print(f"    Accuracy: {accuracy:.3f}")
    print(f"    F1 Score: {f1:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n    Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                    FAIL   SUCCESS")
    print(f"    Actual FAIL     {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"    Actual SUCCESS  {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    # 8. Sample predictions
    print("\n📋 Sample Predictions (Withdrawn Drugs):")
    results_df = pd.DataFrame({
        'name': names,
        'label': labels,
        'prediction': predictions,
        'probability': probabilities,
    })
    
    withdrawn = results_df[results_df['label'] == 0]
    for _, r in withdrawn.head(10).iterrows():
        status = "✅" if r['label'] == r['prediction'] else "❌"
        print(f"    {status} {r['name']}: P(success)={r['probability']:.2f}, Pred={'SUCCESS' if r['prediction'] else 'FAIL'}")
    
    # 9. Save results
    output_path = Path('data/processed/mpbpk_validation_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("  VALIDATION COMPLETE")
    print("=" * 60)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'n_samples': len(y_true),
        'n_withdrawn': (y_true == 0).sum(),
        'n_approved': (y_true == 1).sum(),
    }


if __name__ == "__main__":
    run_mpbpk_validation()
