"""
Validate ClinTox Model with Threshold 0.20
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load data to match training split
clintox_path = Path('data/raw/clintox.csv')
df = pd.read_csv(clintox_path)

# Load model
model_path = Path('models/qsar/clintox_ct_tox_model.pkl')
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']

# Re-create features (simplified for validation script - usually we'd use the saved test set if available, 
# but here we reconstruct the split used in training using random_state=42)
from src.models.qsar_predictor import QSARPredictor
predictor = QSARPredictor(auto_load=False)

X_list = []
y_list = []
smiles_list = []

print("Generating features...")
for idx, row in df.iterrows():
    features = predictor.smiles_to_features(row['smiles'])
    if features is not None:
        X_list.append(features)
        y_list.append(int(row['CT_TOX']))
        smiles_list.append(row['smiles'])

X = np.array(X_list)
y = np.array(y_list)

# Split (Same seed as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
X_test_scaled = scaler.transform(X_test)

# Predict
THRESHOLD = 0.20
y_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print(f"=== QSAR ClinTox Validation (Threshold={THRESHOLD}) ===")
print(f"Test Set Size: {len(y_test)}")
print(f"accuracy: {acc:.1%}")
print(f"precision: {prec:.3f}")
print(f"recall: {rec:.3f}")
print(f"f1_score: {f1:.3f}")
print()
print("Confusion Matrix:")
print(f"             Pred Safe   Pred Toxic")
print(f"True Safe      {cm[0,0]:4d}        {cm[0,1]:4d}")
print(f"True Toxic     {cm[1,0]:4d}        {cm[1,1]:4d}")

if cm[0,0] + cm[0,1] > 0:
    print(f"Specificity (Safe Pass Rate): {cm[0,0]/(cm[0,0]+cm[0,1]):.1%}")
if cm[1,0] + cm[1,1] > 0:
    print(f"Sensitivity (Toxic Detect Rate): {cm[1,1]/(cm[1,0]+cm[1,1]):.1%}")

# Example Predictions
print("\nExample High Risk Drugs Detected:")
test_indices = np.where((y_test == 1) & (y_pred == 1))[0]
for i in test_indices[:5]:
    print(f"  Prob: {y_prob[i]:.2f}")
