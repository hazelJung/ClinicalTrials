"""
PKSmart Phase 2: Toxicity Model Training
-----------------------------------------
Trains 12 toxicity classifiers using the Tox21 dataset.
Uses the same feature pipeline (Morgan + Mordred) as PK models for consistency.

NO DeepChem or TensorFlow required - pure pandas + sklearn.
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from pksmart.features import standardize_smiles, calculate_morgan_fingerprints, calculate_mordred_descriptors

# Tox21 Task Names (12 endpoints)
TOX21_TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

def train_toxicity_models():
    print("=== PKSmart Phase 2: Toxicity Model Training ===")
    print("    (Lightweight version - No DeepChem/TensorFlow)")
    
    # 1. Load Tox21 Data from CSV
    print("\n[1/4] Loading Tox21 dataset from data/tox21.csv...")
    try:
        df = pd.read_csv("data/tox21.csv")
    except FileNotFoundError:
        print("Error: data/tox21.csv not found!")
        return
    
    print(f"      Total rows: {len(df)}")
    print(f"      Columns: {list(df.columns)}")
    
    # The CSV has columns: NR-AR, NR-AR-LBD, ..., mol_id, smiles
    # Values are 0/1 for active/inactive, empty for unknown
    
    # 2. Standardize SMILES and Generate Features
    print("\n[2/4] Generating Features (Standardization + Morgan + Mordred)...")
    
    # Standardize SMILES
    print("      Standardizing SMILES...")
    df['smiles_r'] = df['smiles'].apply(standardize_smiles)
    
    # Drop invalid SMILES
    n_total = len(df)
    df.dropna(subset=['smiles_r'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    n_valid = len(df)
    print(f"      Valid SMILES: {n_valid}/{n_total}")
    
    # Calculate Features
    valid_smiles = df['smiles_r'].tolist()
    
    print("      Calculating Morgan Fingerprints...")
    df_morgan = calculate_morgan_fingerprints(valid_smiles)
    
    print("      Calculating Mordred Descriptors...")
    df_mordred = calculate_mordred_descriptors(valid_smiles)
    
    # Combine Features (X)
    X = pd.concat([df_mordred, df_morgan], axis=1)
    
    # Impute missing values
    print("      Imputing missing feature values...")
    # Use 'median' instead of 'mean' to be robust to outliers in descriptors
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Save imputer for inference
    os.makedirs("models/tox", exist_ok=True)
    joblib.dump(imputer, "models/tox/imputer.joblib")
    
    # 3. Train Models (One per Task)
    print("\n[3/4] Training Toxicity Classifiers...")
    
    for task in TOX21_TASKS:
        if task not in df.columns:
            print(f"      [Warning] Task '{task}' not found in CSV. Skipping.")
            continue
            
        print(f"      Training model for: {task}")
        
        # Get labels for this task
        y_task = df[task].values
        
        # Create mask for valid labels (0 or 1, not NaN/empty)
        # pd.read_csv treats empty as NaN
        mask = ~pd.isna(y_task)
        
        n_samples = np.sum(mask)
        if n_samples < 10:
            print(f"      [Warning] Skipping {task}: Too few samples ({n_samples})")
            continue
        
        X_train = X_imputed[mask]
        y_train = y_task[mask].astype(int)
        
        # Check class balance
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        print(f"        Samples: {n_samples} (Pos: {n_pos}, Neg: {n_neg})")
        
        # Train Random Forest
        # Regularization added: min_samples_leaf=5 to prevent overfitting to noise
        clf = RandomForestClassifier(
            n_estimators=200,          # Increased trees for stability
            max_depth=15,              # Limit depth to prevent memorization
            min_samples_leaf=5,        # Require 5 samples per leaf (Smooths predictions)
            class_weight='balanced',   # Handle class imbalance
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        # Save model
        model_path = f"models/tox/{task}_model.joblib"
        joblib.dump(clf, model_path)
        
        # Report training accuracy (just for sanity check)
        acc = clf.score(X_train, y_train)
        print(f"        -> Saved: {model_path} (Train Acc: {acc:.2%})")
    
    print("\n[4/4] All Toxicity models trained and saved successfully!")
    print("      Models saved to: models/tox/")

if __name__ == "__main__":
    train_toxicity_models()
