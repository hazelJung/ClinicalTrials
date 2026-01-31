"""
Analyze Model Sensitivity to Half-life
======================================
Investigate why the ML model predicts 'SUCCESS' even for drugs with short half-life.
"""

import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.mpbpk_ml import mPBPKClassifier, MLConfig

def analyze():
    print("=" * 60)
    print("  Model Sensitivity Analysis")
    print("=" * 60)

    # 1. Load Model
    # Find latest model
    model_dir = Path('models')
    files = list(model_dir.glob('mpbpk*.pkl'))
    if not files:
        print("Error: No model found.")
        return
    latest_model = max(files, key=lambda x: x.stat().st_mtime)
    print(f"Loading model: {latest_model.name}")
    
    classifier = mPBPKClassifier.load(latest_model)
    rf = classifier.rf_model
    
    # 2. Feature Importance
    print("\n[Feature Importance]")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    features = classifier.feature_names
    for f in range(len(features)):
        print(f"  {f+1}. {features[indices[f]]:<20} ({importances[indices[f]]:.4f})")

    # 3. Sensitivity Test
    print("\n[Half-life Sensitivity Test]")
    print("Varying half-life from 1h to 1000h. Other params fixed at 'default' (good) values.")
    
    # Defaults used in validation
    default_feat = {
        'log_KD': np.log10(1.0),      # 1 nM (Good)
        'log_dose': np.log10(5.0),    # 5 mg/kg (Good)
        'charge': 0,
        'log_MW': np.log10(150),      # 150 kDa
        'log_T0': np.log10(5.0),      # 5 nM (High target, but manageable)
        'population': 'EUR',
        'phenotype': 'NM',
        'activity_score': 2.0,
        'cl_multiplier': 1.0,
    }
    
    # Test range
    halflives = [1, 5, 10, 24, 48, 72, 96, 120, 168, 336, 500, 1000]
    
    print(f"\n{'Half-life (h)':<15} {'log_halflife':<15} {'Probability':<15} {'Prediction'}")
    print("-" * 60)
    
    scaling_impact = []
    
    for hl in halflives:
        feat = default_feat.copy()
        feat['log_halflife'] = np.log10(hl)
        
        # Build vector
        X = []
        for fn in features:
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
        prob = rf.predict_proba(X)[0][1]
        pred = rf.predict(X)[0]
        
        pred_str = "SUCCESS" if pred == 1 else "FAIL !!!"
        print(f"{hl:<15} {np.log10(hl):<15.2f} {prob:<15.4f} {pred_str}")
        
    # 4. What if Dose was lower? Or Target higher?
    print("\n[Combinatorial Check]")
    print("What if Dose is lower (0.5 mg/kg) with short half-life (10h)?")
    
    feat = default_feat.copy()
    feat['log_halflife'] = np.log10(10) # 10h (Bad)
    feat['log_dose'] = np.log10(0.5)    # 0.5 mg/kg (Low)
    
    # Build vector
    X = []
    for fn in features:
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
    prob_comb = rf.predict_proba(X)[0][1]
    print(f"Half-life=10h, Dose=0.5mg/kg -> Prob: {prob_comb:.4f} ({'SUCCESS' if prob_comb>0.5 else 'FAIL'})")


if __name__ == "__main__":
    analyze()
