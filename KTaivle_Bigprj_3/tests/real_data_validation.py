"""
Real Data Validation Script
============================
ClinTox 실제 데이터를 사용하여 QSAR 모델 검증

ClinTox: FDA 승인/거부 및 임상 독성 라벨
- FDA_APPROVED: 1 = 승인, 0 = 거부
- CT_TOX: 1 = 독성, 0 = 안전
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# QSAR imports
from src.models.qsar_predictor import QSARPredictor


def run_clintox_validation(n_samples: int = 100):
    """ClinTox 데이터로 QSAR 모델 검증"""
    print("=" * 60)
    print("  REAL DATA VALIDATION: ClinTox Dataset")
    print("=" * 60)
    
    # 1. Load ClinTox data
    df = pd.read_csv('data/raw/clintox.csv')
    print(f"\n[1] Dataset loaded: {len(df)} drugs")
    print(f"    FDA Approved: {df['FDA_APPROVED'].sum()} ({df['FDA_APPROVED'].mean()*100:.1f}%)")
    print(f"    Clinical Toxicity: {df['CT_TOX'].sum()} ({df['CT_TOX'].mean()*100:.1f}%)")
    
    # 2. Sample for validation
    df_sample = df.sample(n=min(n_samples, len(df)), random_state=42)
    print(f"\n[2] Sampled {len(df_sample)} drugs for validation")
    
    # 3. Initialize QSAR predictor
    print("\n[3] Initializing QSAR Predictor...")
    predictor = QSARPredictor()
    
    # 4. Run predictions
    print("\n[4] Running predictions...")
    results = []
    
    for idx, row in df_sample.iterrows():
        smiles = row['smiles']
        fda_approved = row['FDA_APPROVED']
        ct_tox = row['CT_TOX']
        
        try:
            # Predict all Tox21 endpoints
            preds = predictor.predict_all(smiles)
            
            if preds:
                # Count positive toxicity predictions
                toxic_count = sum(1 for ep, r in preds.items() if r.prediction == 1)
                avg_prob = np.mean([r.probability for r in preds.values()])
                
                # Our prediction: toxic if >= 2 endpoints positive
                our_prediction = 1 if toxic_count >= 2 else 0
                
                results.append({
                    'smiles': smiles[:30] + '...' if len(smiles) > 30 else smiles,
                    'fda_approved': fda_approved,
                    'ct_tox': ct_tox,
                    'toxic_count': toxic_count,
                    'avg_prob': avg_prob,
                    'our_pred': our_prediction,
                })
        except Exception as e:
            continue
    
    print(f"    Successfully predicted: {len(results)} / {len(df_sample)}")
    
    if len(results) < 10:
        print("    WARNING: Too few successful predictions for validation")
        return
    
    # 5. Evaluate
    print("\n[5] Validation Results")
    print("-" * 60)
    
    results_df = pd.DataFrame(results)
    
    # Compare our prediction vs CT_TOX (clinical toxicity)
    y_true = results_df['ct_tox'].values
    y_pred = results_df['our_pred'].values
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    
    # Handle cases where there might be only one class
    unique_true = np.unique(y_true)
    if len(unique_true) > 1:
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, results_df['avg_prob'].values)
        except:
            auc = 0.0
    else:
        f1 = 0.0
        auc = 0.0
    
    print(f"\n  📊 QSAR vs Clinical Toxicity (CT_TOX)")
    print(f"    Accuracy:  {acc:.3f}")
    print(f"    F1 Score:  {f1:.3f}")
    print(f"    ROC-AUC:   {auc:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n    Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Safe  Toxic")
    print(f"    Actual Safe   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"    Actual Toxic  {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # 6. FDA Approval correlation
    print("\n  📊 QSAR vs FDA Approval")
    y_true_fda = results_df['fda_approved'].values
    # For FDA: our_pred=0 (safe) should correlate with fda_approved=1
    y_pred_fda = 1 - results_df['our_pred'].values  # Invert: safe -> approved
    
    acc_fda = accuracy_score(y_true_fda, y_pred_fda)
    print(f"    Accuracy:  {acc_fda:.3f}")
    
    # 7. Example predictions
    print("\n  📋 Sample Predictions:")
    for _, r in results_df.head(10).iterrows():
        status = "✅" if r['ct_tox'] == r['our_pred'] else "❌"
        print(f"    {status} {r['smiles']}: Toxic={r['ct_tox']}, Pred={r['our_pred']} ({r['toxic_count']}/12 endpoints)")
    
    # 8. Save results
    output_path = Path('data/processed/clintox_validation_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n  ✅ Results saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("  VALIDATION COMPLETE")
    print("=" * 60)
    
    return {
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': auc,
        'n_samples': len(results),
    }


if __name__ == "__main__":
    # Run with 200 samples for faster validation
    run_clintox_validation(n_samples=200)
