"""
PKSmart Phase 2: Toxicity Model Evaluation
-------------------------------------------
Evaluates toxicity models using proper Train/Test split.
Reports AUC-ROC for each of the 12 Tox21 endpoints.
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from pksmart.features import standardize_smiles, calculate_morgan_fingerprints, calculate_mordred_descriptors

# Tox21 Task Names (12 endpoints)
TOX21_TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

def evaluate_toxicity_models():
    print("=== PKSmart Phase 2: Toxicity Model Evaluation ===")
    print("    (Proper Train/Test Split - AUC-ROC Metrics)")
    
    # 1. Load Tox21 Data from CSV
    print("\n[1/5] Loading Tox21 dataset from data/tox21.csv...")
    try:
        df = pd.read_csv("data/tox21.csv")
    except FileNotFoundError:
        print("Error: data/tox21.csv not found!")
        return
    
    print(f"      Total rows: {len(df)}")
    
    # 2. Standardize SMILES
    print("\n[2/5] Standardizing SMILES...")
    df['smiles_r'] = df['smiles'].apply(standardize_smiles)
    
    # Drop invalid SMILES
    n_total = len(df)
    df.dropna(subset=['smiles_r'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    n_valid = len(df)
    print(f"      Valid SMILES: {n_valid}/{n_total}")
    
    # 3. Generate Features
    print("\n[3/5] Generating Features (Morgan + Mordred)...")
    valid_smiles = df['smiles_r'].tolist()
    
    print("      Calculating Morgan Fingerprints...")
    df_morgan = calculate_morgan_fingerprints(valid_smiles)
    
    print("      Calculating Mordred Descriptors...")
    df_mordred = calculate_mordred_descriptors(valid_smiles)
    
    # Combine Features (X)
    X = pd.concat([df_mordred, df_morgan], axis=1)
    
    # Impute missing values
    print("      Imputing missing feature values...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # 4. Train/Test Split & Evaluate Each Task
    print("\n[4/5] Evaluating Models (80/20 Train/Test Split)...")
    print("-" * 70)
    print(f"{'Task':<20} {'N_Train':<10} {'N_Test':<10} {'AUC-ROC':<12} {'Accuracy':<10}")
    print("-" * 70)
    
    results = []
    
    for task in TOX21_TASKS:
        if task not in df.columns:
            print(f"{task:<20} [SKIPPED - Column not found]")
            continue
        
        # Get labels for this task
        y_task = df[task].values
        
        # Create mask for valid labels (0 or 1, not NaN/empty)
        mask = ~pd.isna(y_task)
        
        n_samples = np.sum(mask)
        if n_samples < 50:
            print(f"{task:<20} [SKIPPED - Too few samples: {n_samples}]")
            continue
        
        X_valid = X_imputed[mask]
        y_valid = y_task[mask].astype(int)
        
        # Check class distribution
        n_pos = np.sum(y_valid == 1)
        n_neg = np.sum(y_valid == 0)
        
        if n_pos < 5 or n_neg < 5:
            print(f"{task:<20} [SKIPPED - Class imbalance too severe: {n_pos}/{n_neg}]")
            continue
        
        # Train/Test Split (Stratified)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_valid, y_valid, 
                test_size=0.2, 
                random_state=42,
                stratify=y_valid
            )
        except ValueError as e:
            print(f"{task:<20} [SKIPPED - Split error: {e}]")
            continue
        
        # Train Random Forest
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        # Predict probabilities for AUC
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)
        
        # Calculate Metrics
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc = float('nan')
        
        acc = accuracy_score(y_test, y_pred)
        
        print(f"{task:<20} {len(y_train):<10} {len(y_test):<10} {auc:<12.4f} {acc:<10.2%}")
        
        results.append({
            'Task': task,
            'N_Train': len(y_train),
            'N_Test': len(y_test),
            'AUC-ROC': auc,
            'Accuracy': acc
        })
    
    print("-" * 70)
    
    # 5. Summary
    print("\n[5/5] Summary")
    if results:
        results_df = pd.DataFrame(results)
        mean_auc = results_df['AUC-ROC'].mean()
        print(f"      Average AUC-ROC across all tasks: {mean_auc:.4f}")
        
        # Interpretation
        if mean_auc >= 0.8:
            print("      Interpretation: GOOD (AUC >= 0.8)")
        elif mean_auc >= 0.7:
            print("      Interpretation: ACCEPTABLE (AUC >= 0.7)")
        else:
            print("      Interpretation: NEEDS IMPROVEMENT (AUC < 0.7)")
    else:
        print("      No valid tasks could be evaluated!")

if __name__ == "__main__":
    evaluate_toxicity_models()
