"""
ClinTox Direct Training and Validation
=======================================
Train Random Forest directly on ClinTox CT_TOX labels for improved clinical toxicity prediction.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from src.models.qsar_predictor import QSARPredictor

print("=" * 60)
print("  ClinTox Direct Training & Validation")
print("=" * 60)

# 1. Load ClinTox data
clintox_path = Path('data/raw/clintox.csv')
df = pd.read_csv(clintox_path)
print(f"\n[1] Loaded {len(df)} compounds from ClinTox")
print(f"    CT_TOX positive: {df['CT_TOX'].sum()} ({100*df['CT_TOX'].mean():.1f}%)")

# 2. Calculate molecular descriptors
print("\n[2] Calculating molecular descriptors...")
predictor = QSARPredictor(auto_load=False)  # Don't load existing models

X_list = []
y_list = []
valid_indices = []

for idx, row in df.iterrows():
    features = predictor.smiles_to_features(row['smiles'])
    if features is not None:
        X_list.append(features)
        y_list.append(int(row['CT_TOX']))
        valid_indices.append(idx)

X = np.array(X_list)
y = np.array(y_list)

print(f"    Valid compounds: {len(X)}")
print(f"    Features: {X.shape[1]}")
print(f"    Class distribution: {np.bincount(y)}")

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[3] Train/Test Split")
print(f"    Train: {len(X_train)}, Test: {len(X_test)}")

# 4. Apply SMOTE for class balancing
if SMOTE_AVAILABLE:
    print("\n[4] Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"    After SMOTE: {len(X_train_balanced)} samples")
    print(f"    Class distribution: {np.bincount(y_train_balanced)}")
else:
    X_train_balanced, y_train_balanced = X_train, y_train
    print("\n[4] SMOTE not available, using original data")

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# 6. Train Random Forest
print("\n[5] Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
model.fit(X_train_scaled, y_train_balanced)

# 7. Cross-validation
print("\n[6] Cross-validation...")
cv_scores = cross_val_score(model, X_train_scaled, y_train_balanced, cv=5, scoring='f1')
print(f"    CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 8. Evaluate on test set
print("\n[7] Test Set Evaluation")
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print(f"    Accuracy:  {acc:.1%}")
print(f"    Precision: {prec:.3f}")
print(f"    Recall:    {rec:.3f}")
print(f"    F1 Score:  {f1:.3f}")
print(f"    ROC-AUC:   {roc_auc:.3f}")
print()
print(f"    Confusion Matrix:")
print(f"                 Pred Safe   Pred Toxic")
print(f"    True Safe      {cm[0,0]:4d}        {cm[0,1]:4d}")
print(f"    True Toxic     {cm[1,0]:4d}        {cm[1,1]:4d}")

# 9. Save model
output_dir = Path('models/qsar')
output_dir.mkdir(parents=True, exist_ok=True)
model_path = output_dir / 'clintox_ct_tox_model.pkl'

model_data = {
    'model': model,
    'scaler': scaler,
    'feature_names': predictor.feature_names,
    'performance': {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc
    }
}

with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n[8] Model saved to: {model_path}")

# 10. Threshold optimization analysis
print("\n[9] Threshold Optimization Analysis")
print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
print("-" * 50)

best_f1 = 0
best_thresh = 0.5

for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    y_pred_t = (y_prob >= thresh).astype(int)
    p = precision_score(y_test, y_pred_t, zero_division=0)
    r = recall_score(y_test, y_pred_t, zero_division=0)
    f = f1_score(y_test, y_pred_t, zero_division=0)
    
    marker = ""
    if f > best_f1:
        best_f1 = f
        best_thresh = thresh
        marker = " <-- Best"
    
    print(f"{thresh:>10.2f} | {p:>10.3f} | {r:>10.3f} | {f:>10.3f}{marker}")

print(f"\n    Optimal Threshold: {best_thresh} (F1={best_f1:.3f})")

print("\n" + "=" * 60)
print("  Training & Validation Complete")
print("=" * 60)
