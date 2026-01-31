"""Check model performance with detailed metrics"""
import sys
import os
sys.path.insert(0, '.')

import pickle
from pathlib import Path
import numpy as np

# Find latest model with absolute path
script_dir = Path(__file__).parent.parent
models_dir = script_dir / 'models'
model_files = sorted(models_dir.glob('mpbpk_classifier_*.pkl'), reverse=True)
latest_model = model_files[0]

print(f"Loading model: {latest_model}")
with open(str(latest_model), 'rb') as f:
    data = pickle.load(f)

results = data['results']
feature_names = data['feature_names']

print("\n" + "=" * 60)
print("  MODEL PERFORMANCE REPORT")
print("=" * 60)

print("\n📊 Decision Tree Performance:")
dt = results['decision_tree']
print(f"   Accuracy:  {dt['accuracy']:.3f}")
print(f"   Precision: {dt['precision']:.3f}")
print(f"   Recall:    {dt['recall']:.3f}")
print(f"   F1 Score:  {dt['f1']:.3f}")
print(f"   ROC-AUC:   {dt['roc_auc']:.3f}")

print("\n📊 Random Forest Performance:")
rf = results['random_forest']
print(f"   Accuracy:  {rf['accuracy']:.3f}")
print(f"   Precision: {rf['precision']:.3f}")
print(f"   Recall:    {rf['recall']:.3f}")
print(f"   F1 Score:  {rf['f1']:.3f}")
print(f"   ROC-AUC:   {rf['roc_auc']:.3f}")

print("\n📊 Cross-Validation (5-Fold):")
dt_cv = results['dt_cv_scores']
rf_cv = results['rf_cv_scores']
print(f"   Decision Tree F1: {dt_cv.mean():.3f} ± {dt_cv.std():.3f}")
print(f"   Random Forest F1: {rf_cv.mean():.3f} ± {rf_cv.std():.3f}")

print("\n📊 Feature Importance (Top 10):")
rf_model = data['rf_model']
importance = rf_model.feature_importances_
sorted_idx = np.argsort(importance)[::-1][:10]
for i, idx in enumerate(sorted_idx):
    print(f"   {i+1}. {feature_names[idx]}: {importance[idx]:.3f}")

print("\n" + "=" * 60)
