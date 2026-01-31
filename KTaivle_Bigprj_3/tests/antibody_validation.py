"""
Real Antibody Drug Validation
==============================
Tests the trained mPBPK-ML model with FDA-approved antibody drugs.

These drugs have MW ~150 kDa, matching our training data domain.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from pathlib import Path
from src.models.mpbpk_ml import mPBPKClassifier, MLConfig

# =============================================================================
# Real Antibody Drug Parameters (from FDA labels, literature)
# =============================================================================

ANTIBODY_DRUGS = {
    'Adalimumab': {
        # Humira - TNF-alpha inhibitor
        'KD_nM': 0.1,           # Very high affinity to TNF-α
        'dose_mg_kg': 0.6,      # 40 mg Q2W for ~70kg = ~0.6 mg/kg
        'MW_kDa': 148,          # Full IgG1
        'halflife_hr': 336,     # ~14 days
        'T0_nM': 5.0,           # TNF-α levels (elevated in RA)
        'charge': 0,
        'clinical_status': 'FDA Approved (2002)',
        'indication': 'Rheumatoid Arthritis, Psoriasis, Crohn\'s',
        'expected': 'SUCCESS',
    },
    'Trastuzumab': {
        # Herceptin - HER2 inhibitor
        'KD_nM': 0.5,           # High affinity to HER2
        'dose_mg_kg': 6.0,      # 6 mg/kg loading, then 2 mg/kg
        'MW_kDa': 148,          # Full IgG1
        'halflife_hr': 672,     # ~28 days
        'T0_nM': 10.0,          # HER2 overexpression
        'charge': 0,
        'clinical_status': 'FDA Approved (1998)',
        'indication': 'HER2+ Breast Cancer',
        'expected': 'SUCCESS',
    },
    'Pembrolizumab': {
        # Keytruda - PD-1 inhibitor
        'KD_nM': 0.03,          # Very high affinity
        'dose_mg_kg': 2.0,      # 200 mg Q3W for ~70kg = ~2.9 mg/kg
        'MW_kDa': 149,          # IgG4
        'halflife_hr': 552,     # ~23 days
        'T0_nM': 3.0,           # PD-1 expression
        'charge': 0,
        'clinical_status': 'FDA Approved (2014)',
        'indication': 'Various Cancers (melanoma, NSCLC, etc.)',
        'expected': 'SUCCESS',
    },
    'Nivolumab': {
        # Opdivo - PD-1 inhibitor
        'KD_nM': 0.05,          # Very high affinity
        'dose_mg_kg': 3.0,      # 240 mg Q2W for ~70kg = ~3.4 mg/kg
        'MW_kDa': 146,          # IgG4
        'halflife_hr': 624,     # ~26 days
        'T0_nM': 3.0,           # PD-1 expression
        'charge': 0,
        'clinical_status': 'FDA Approved (2014)',
        'indication': 'Melanoma, RCC, NSCLC',
        'expected': 'SUCCESS',
    },
    'Rituximab': {
        # Rituxan - CD20 inhibitor
        'KD_nM': 8.0,           # Moderate affinity
        'dose_mg_kg': 5.0,      # 375 mg/m² ≈ 5 mg/kg
        'MW_kDa': 145,          # IgG1
        'halflife_hr': 504,     # ~21 days
        'T0_nM': 20.0,          # CD20 on B-cells
        'charge': 0,
        'clinical_status': 'FDA Approved (1997)',
        'indication': 'NHL, CLL, RA',
        'expected': 'SUCCESS',
    },
    'Weak_Binder_Hypothetical': {
        # Hypothetical failed drug for comparison
        'KD_nM': 500.0,         # Weak binding
        'dose_mg_kg': 1.0,      # Low dose
        'MW_kDa': 150,          
        'halflife_hr': 100,     # Short half-life
        'T0_nM': 50.0,          # High target expression
        'charge': 0,
        'clinical_status': 'Hypothetical (FAIL expected)',
        'indication': 'N/A',
        'expected': 'FAIL',
    },
}

def create_feature_vector(drug_params: dict, patient: dict, feature_names: list) -> np.ndarray:
    """Create feature vector for prediction."""
    params = {
        'log_KD': np.log10(drug_params['KD_nM']),
        'log_dose': np.log10(drug_params['dose_mg_kg']),
        'charge': drug_params['charge'],
        'log_MW': np.log10(drug_params['MW_kDa']),
        'log_T0': np.log10(drug_params['T0_nM']),
        'log_halflife': np.log10(drug_params['halflife_hr']),
        **patient
    }
    
    features = []
    for feat in feature_names:
        if feat in params:
            features.append(params[feat])
        elif feat.startswith('pop_'):
            pop = feat.replace('pop_', '')
            features.append(1 if params.get('population') == pop else 0)
        elif feat == 'phenotype_encoded':
            pheno_map = {'PM': 0, 'IM': 1, 'NM': 2, 'UM': 3}
            features.append(pheno_map.get(params.get('phenotype', 'NM'), 2))
        else:
            features.append(0)
    
    return np.array(features).reshape(1, -1)

def run_validation():
    print("=" * 70)
    print("  REAL ANTIBODY DRUG VALIDATION")
    print("=" * 70)
    
    # Train classifier
    print("\n1. Training model...")
    classifier = mPBPKClassifier(MLConfig(use_smote=True))
    classifier.train()
    feature_names = classifier.feature_names
    
    # Standard patient (EUR/NM)
    standard_patient = {
        'population': 'EUR', 
        'phenotype': 'NM', 
        'activity_score': 2.0, 
        'cl_multiplier': 1.0
    }
    
    print("\n2. Testing antibody drugs (EUR/NM patient)...")
    print("=" * 70)
    
    results = []
    correct = 0
    total = 0
    
    for drug_name, drug_params in ANTIBODY_DRUGS.items():
        X = create_feature_vector(drug_params, standard_patient, feature_names)
        prob = classifier.rf_model.predict_proba(X)[0][1]
        pred = 'SUCCESS' if classifier.rf_model.predict(X)[0] else 'FAIL'
        expected = drug_params['expected']
        
        match = "✅" if pred == expected else "❌"
        if pred == expected:
            correct += 1
        total += 1
        
        print(f"\n📊 {drug_name}")
        print(f"   Status: {drug_params['clinical_status']}")
        print(f"   KD: {drug_params['KD_nM']} nM | Dose: {drug_params['dose_mg_kg']} mg/kg | t½: {drug_params['halflife_hr']} hr")
        print(f"   P(success) = {prob:.3f} → Prediction: {pred} | Expected: {expected} {match}")
        
        results.append({
            'drug': drug_name,
            'probability': prob,
            'prediction': pred,
            'expected': expected,
            'correct': pred == expected,
        })
    
    # Summary
    accuracy = correct / total * 100
    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\n   Accuracy: {correct}/{total} ({accuracy:.0f}%)")
    print(f"\n   Results by Drug:")
    for r in results:
        match = "✅" if r['correct'] else "❌"
        print(f"   - {r['drug']}: P={r['probability']:.3f}, {r['prediction']} (expected {r['expected']}) {match}")
    
    # Save results
    output_path = Path('data/processed/antibody_validation_results.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("ANTIBODY DRUG VALIDATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy: {correct}/{total} ({accuracy:.0f}%)\n\n")
        for r in results:
            match = "CORRECT" if r['correct'] else "WRONG"
            f.write(f"{r['drug']}: P={r['probability']:.3f} → {r['prediction']} (expected {r['expected']}) [{match}]\n")
    
    print(f"\n✅ Results saved to: {output_path}")

if __name__ == "__main__":
    run_validation()
