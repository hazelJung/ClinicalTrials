"""
mPBPK Training Data Generation and Model Saving
================================================
1. Generate 50K virtual drug-patient simulations
2. Train Random Forest & Decision Tree models
3. Save trained models to disk
"""

import sys
sys.path.insert(0, '.')

from pathlib import Path
from src.models.batch_simulator import SimulationMatrix, SimulationConfig, save_training_data
from src.models.mpbpk_ml import mPBPKClassifier, MLConfig

def main():
    print("=" * 60)
    print("  mPBPK Training Pipeline")
    print("=" * 60)
    
    # Step 1: Generate virtual data
    print("\n[1] Generating virtual drug-patient simulations...")
    
    config = SimulationConfig(
        patients_per_pop=20,  # 20 patients per population (5 pops)
        # Total = 500 drugs * 20 patients * 5 pops = 50,000 samples
    )
    
    simulator = SimulationMatrix(config)
    df = simulator.run(n_drugs=500, verbose=True)
    
    print(f"    Generated: {len(df)} samples")
    print(f"    Columns: {df.columns.tolist()}")
    print(f"    Efficacy Success: {df['efficacy_success'].sum()} ({df['efficacy_success'].mean()*100:.1f}%)")
    if 'toxicity' in df.columns:
        print(f"    Toxicity: {df['toxicity'].sum()} ({df['toxicity'].mean()*100:.1f}%)")
    
    # Step 2: Save training data
    print("\n[2] Saving training data...")
    data_path = save_training_data(df)
    print(f"    Saved to: {data_path}")
    
    # Step 3: Train ML models
    print("\n[3] Training ML models...")
    classifier = mPBPKClassifier(MLConfig(use_smote=True))
    results = classifier.train(data_path=data_path)
    
    print(f"\n    Decision Tree:")
    print(f"      Accuracy: {results['dt_results']['accuracy']:.3f}")
    print(f"      F1 Score: {results['dt_results']['f1_score']:.3f}")
    print(f"      ROC-AUC:  {results['dt_results']['roc_auc']:.3f}")
    
    print(f"\n    Random Forest:")
    print(f"      Accuracy: {results['rf_results']['accuracy']:.3f}")
    print(f"      F1 Score: {results['rf_results']['f1_score']:.3f}")
    print(f"      ROC-AUC:  {results['rf_results']['roc_auc']:.3f}")
    
    # Step 4: Save models
    print("\n[4] Saving trained models...")
    model_path = classifier.save()
    print(f"    Models saved to: {model_path}")
    
    # Step 5: Feature importance
    print("\n[5] Feature Importance (Top 5):")
    importance = classifier.get_feature_importance()
    for feat, imp in list(importance.items())[:5]:
        print(f"    {feat}: {imp:.3f}")
    
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    main()
