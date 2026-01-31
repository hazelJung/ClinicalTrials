"""Train and save performance report to file"""
import sys
sys.path.insert(0, '.')

from src.models.mpbpk_ml import mPBPKClassifier, MLConfig
from pathlib import Path
from datetime import datetime

# Train with default config
config = MLConfig(use_smote=True)
classifier = mPBPKClassifier(config)

print("Training with 50K samples...")
results = classifier.train()

# Create report
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = Path(f'data/processed/model_performance_report_{timestamp}.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("  mPBPK-ML MODEL PERFORMANCE REPORT\n")
    f.write("=" * 60 + "\n\n")
    
    # Decision Tree
    f.write("📊 DECISION TREE PERFORMANCE\n")
    f.write("-" * 40 + "\n")
    dt = results['decision_tree']
    f.write(f"   Accuracy:  {dt['accuracy']:.3f}\n")
    f.write(f"   Precision: {dt['precision']:.3f}\n")
    f.write(f"   Recall:    {dt['recall']:.3f}\n")
    f.write(f"   F1 Score:  {dt['f1']:.3f}\n")
    f.write(f"   ROC-AUC:   {dt['roc_auc']:.3f}\n\n")
    
    # Random Forest
    f.write("📊 RANDOM FOREST PERFORMANCE\n")
    f.write("-" * 40 + "\n")
    rf = results['random_forest']
    f.write(f"   Accuracy:  {rf['accuracy']:.3f}\n")
    f.write(f"   Precision: {rf['precision']:.3f}\n")
    f.write(f"   Recall:    {rf['recall']:.3f}\n")
    f.write(f"   F1 Score:  {rf['f1']:.3f}\n")
    f.write(f"   ROC-AUC:   {rf['roc_auc']:.3f}\n\n")
    
    # Cross-validation
    f.write("📊 CROSS-VALIDATION (5-Fold)\n")
    f.write("-" * 40 + "\n")
    dt_cv = results['dt_cv_scores']
    rf_cv = results['rf_cv_scores']
    f.write(f"   Decision Tree F1: {dt_cv.mean():.3f} ± {dt_cv.std():.3f}\n")
    f.write(f"   Random Forest F1: {rf_cv.mean():.3f} ± {rf_cv.std():.3f}\n\n")
    
    # Feature importance
    f.write("📊 FEATURE IMPORTANCE (Top 10)\n")
    f.write("-" * 40 + "\n")
    importance_df = classifier.get_feature_importance()
    f.write(importance_df.head(10).to_string(index=False))
    f.write("\n\n")
    
    # Decision rules
    f.write("📊 DECISION TREE RULES (depth=3)\n")
    f.write("-" * 40 + "\n")
    rules = classifier.get_decision_rules(max_depth=3)
    f.write(rules)

print(f"\nReport saved to: {report_path}")
print("\n--- Report Content ---")
with open(report_path, 'r', encoding='utf-8') as f:
    print(f.read())
