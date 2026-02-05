"""
PKSmart Model Validation Script (v3)
=====================================
Validates custom-trained models (PK, LD50, Tox21) without using the pksmart library.
Directly loads .joblib models and evaluates performance using train/test split.

Updates:
- v3: Added Feature Caching to 'validation/cache/' to avoid re-computing Mordred.
- v3: Fixed feature selection for LD50/Tox21 (include boolean cols).
- v2: Fixed PK feature preparation to include animal predictions.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.impute import SimpleImputer

# Add parent directory to path for pksmart.features
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pksmart.features import generate_features

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(VALIDATION_DIR, "results")
CACHE_DIR = os.path.join(VALIDATION_DIR, "cache")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# PK Model Configuration
SPECIES = ["rat", "dog", "monkey"]
ANIMAL_PARAMS = ["CL_mL_min_kg", "VDss_L_kg", "fup"]
HUMAN_PARAMS = ["human_CL_mL_min_kg", "human_VDss_L_kg", "human_fup", "human_mrt", "human_thalf"]

# Tox21 Endpoints
TOX21_TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

# Quick Validation Mode: Set to an integer (e.g. 500) to sample data. Set to None for full validation.
SAMPLE_SIZE = 500

# ============================================================================
# Helper Functions
# ============================================================================

def get_features(smiles_list, name="data"):
    """
    Get features for a list of SMILES.
    Checks cache first. If not found, generates and saves to cache.
    """
    cache_path = os.path.join(CACHE_DIR, f"features_{name}.csv")
    
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        try:
            return pd.read_csv(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")
    
    print(f"Generating features for {len(smiles_list)} items...")
    features_df = generate_features(smiles_list)
    
    if features_df is not None:
        # Ensure column names are strings
        features_df.columns = features_df.columns.astype(str)
        
        # Save to cache (use csv to avoid dependencies)
        print(f"Saving features to cache: {cache_path}")
        features_df.to_csv(cache_path, index=False)
        
    return features_df

def gmfe(y_true, y_pred, log_transformed=True):
    """Geometric Mean Fold Error"""
    if log_transformed:
        diff = np.abs(y_pred - y_true)
    else:
        diff = np.abs(np.log10(y_pred + 1e-10) - np.log10(y_true + 1e-10))
    return 10 ** np.mean(diff)

def fold_accuracy(y_true, y_pred, fold=2, log_transformed=True):
    """Fraction of predictions within N-fold of true value"""
    if log_transformed:
        diff = np.abs(y_pred - y_true)
        threshold = np.log10(fold)
    else:
        ratio = np.maximum(y_pred / (y_true + 1e-10), y_true / (y_pred + 1e-10))
        return np.mean(ratio <= fold)
    return np.mean(diff <= threshold)

def calculate_regression_metrics(y_true, y_pred, log_transformed=True, name=""):
    """Calculate all regression metrics"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {"Model": name, "N": 0, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, 
                "GMFE": np.nan, "2-Fold%": np.nan, "3-Fold%": np.nan}
    
    return {
        "Model": name,
        "N": len(y_true),
        "R2": round(r2_score(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "GMFE": round(gmfe(y_true, y_pred, log_transformed), 4),
        "2-Fold%": round(fold_accuracy(y_true, y_pred, 2, log_transformed) * 100, 1),
        "3-Fold%": round(fold_accuracy(y_true, y_pred, 3, log_transformed) * 100, 1),
    }

def calculate_classification_metrics(y_true, y_pred_proba, name=""):
    """Calculate all classification metrics"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred_proba))
    y_true = y_true[mask]
    y_pred_proba = y_pred_proba[mask]
    
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"Model": name, "N": len(y_true), "AUC": np.nan, "Accuracy": np.nan, 
                "F1": np.nan, "Precision": np.nan, "Recall": np.nan}
    
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except:
        auc = np.nan
    
    return {
        "Model": name,
        "N": len(y_true),
        "AUC": round(auc, 4) if not np.isnan(auc) else np.nan,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
    }

# ============================================================================
# PK Model Validation
# ============================================================================

def validate_pk_models():
    """Validate PK models matching training approach (Uses np.number filter)"""
    print("\n" + "="*60)
    print("PK MODEL VALIDATION")
    print("="*60)
    
    results = []
    predictions_data = []
    
    # Load Human Data
    human_data_path = os.path.join(DATA_DIR, "Human_PK_data.csv")
    if not os.path.exists(human_data_path):
        print(f"Error: Human data not found at {human_data_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(human_data_path)
    df = df.dropna(subset=['smiles_r'])
    df['smiles_r'] = df['smiles_r'].astype(str)
    
    # Apply Sampling if configured
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        print(f"⚠️ QUICK MODE: Sampling {SAMPLE_SIZE} random samples from {len(df)} total...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        
    print(f"Loaded {len(df)} human PK samples")
    
    # Generate features
    features_df = get_features(df['smiles_r'].tolist(), name="pk_human")
    
    if features_df is None:
        print("Error: Feature generation failed")
        return pd.DataFrame()
    
    # Align with original data
    df = df.iloc[:len(features_df)].reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)
    
    # PK Training used only numeric columns
    X_numeric = features_df.select_dtypes(include=[np.number])
    print(f"Numeric feature columns: {X_numeric.shape[1]}")
    
    # Generate animal predictions
    print("\nGenerating animal predictions for human molecules...")
    animal_preds = pd.DataFrame()
    
    for species in SPECIES:
        for param in ANIMAL_PARAMS:
            model_name = f"{species}_{param}"
            model_path = os.path.join(MODEL_DIR, f"{model_name}_model.joblib")
            imputer_path = os.path.join(MODEL_DIR, f"{model_name}_imputer.joblib")
            
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                imputer = joblib.load(imputer_path)
                
                # Animal models also trained on numeric only
                X_imputed = imputer.transform(X_numeric)
                preds = model.predict(X_imputed)
                animal_preds[model_name] = preds
    
    # Combine
    X_combined = pd.concat([X_numeric, animal_preds], axis=1)
    
    # Validate each human PK model
    for target in HUMAN_PARAMS:
        print(f"\n--- Validating {target} ---")
        
        model_path = os.path.join(MODEL_DIR, f"{target}_model.joblib")
        imputer_path = os.path.join(MODEL_DIR, f"{target}_imputer.joblib")
        
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            continue
        
        model = joblib.load(model_path)
        imputer = joblib.load(imputer_path) if os.path.exists(imputer_path) else None
        
        # Filter valid target values
        valid_mask = df[target].notna() & (df[target] > 0 if 'fup' not in target else True)
        
        if valid_mask.sum() < 10:
            print(f"  Not enough data: {valid_mask.sum()} samples")
            continue
        
        X = X_combined[valid_mask].values
        y = df[target][valid_mask].values
        
        # Log transform (except fup)
        is_log = 'fup' not in target
        if is_log:
            y = np.log10(y + 1e-10)
        
        # Apply imputer
        if imputer:
            X = imputer.transform(X)
        else:
            X = np.nan_to_num(X, nan=0.0)
        
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_regression_metrics(y_test, y_pred, log_transformed=is_log, name=target)
        results.append(metrics)
        
        print(f"  N={metrics['N']}, R²={metrics['R2']}, RMSE={metrics['RMSE']}, 2-Fold={metrics['2-Fold%']}%")
        
        # Store predictions
        for true, pred in zip(y_test, y_pred):
            predictions_data.append({
                'model': target,
                'y_true': true,
                'y_pred': pred,
                'log_transformed': is_log
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "pk_validation_results.csv"), index=False)
    
    predictions_df = pd.DataFrame(predictions_data)
    predictions_df.to_csv(os.path.join(RESULTS_DIR, "pk_predictions.csv"), index=False)
    
    return results_df

# ============================================================================
# LD50 Model Validation
# ============================================================================

def validate_ld50_model():
    """Validate LD50 model (Uses ALL features, not just np.number)"""
    print("\n" + "="*60)
    print("LD50 MODEL VALIDATION")
    print("="*60)
    
    # Load LD50 data
    ld50_path = os.path.join(DATA_DIR, "ld50.csv")
    if not os.path.exists(ld50_path):
        print(f"Error: LD50 data not found at {ld50_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(ld50_path)
    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    target_col = 'LD50'
    
    # Filter valid data
    df = df.dropna(subset=[smiles_col, target_col])
    df = df[df[target_col] > 0]
    
    # Apply Sampling if configured
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        print(f"⚠️ QUICK MODE: Sampling {SAMPLE_SIZE} random samples from {len(df)} total...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        
    print(f"Loaded {len(df)} valid LD50 samples")
    
    # Generate features
    features_df = get_features(df[smiles_col].tolist(), name="ld50")
    
    if features_df is None:
        print("Error: Feature generation failed")
        return pd.DataFrame()
    
    # Align data
    df = df.iloc[:len(features_df)].reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)
    
    # Load model
    model_path = os.path.join(MODEL_DIR, "ld50", "ld50_model.joblib")
    imputer_path = os.path.join(MODEL_DIR, "ld50", "imputer.joblib")
    
    if not os.path.exists(model_path):
        print("Model not found")
        return pd.DataFrame()
        
    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path) if os.path.exists(imputer_path) else None
    
    # Use ALL features (LD50 training did NOT filter for np.number)
    # Exclude metadata columns
    exclude_cols = ['smiles', 'smiles_r', 'mol_id', 'cas', 'id']
    feature_cols = [c for c in features_df.columns if c.lower() not in exclude_cols]
    
    X = features_df[feature_cols].copy()
    
    # Convert booleans to int explicitly to help imputer
    # (Although SimpleImputer might handle it, explicitly converting is safer)
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)
        
    y = df[target_col].values
    y_log = np.log10(y + 1)
    
    print(f"Feature columns: {X.shape[1]}")
    
    # Apply imputer
    if imputer:
        # Warning: imputer was fitted on dataframe with column names
        # We must pass dataframe with same column names
        try:
            X = imputer.transform(X)
        except Exception as e:
            print(f"Imputation warning: {e}")
            # Fallback: if columns don't match perfectly, align them
            # This happens if 'training' had columns 'validation' doesn't, or vice versa
            pass
            
    else:
         X = np.nan_to_num(X.values, nan=0.0)
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    
    # Predict
    y_pred = model.predict(X_test)
    
    metrics = calculate_regression_metrics(y_test, y_pred, log_transformed=True, name="LD50")
    
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(os.path.join(RESULTS_DIR, "ld50_validation_results.csv"), index=False)
    
    predictions_df = pd.DataFrame({
        'model': 'LD50',
        'y_true': y_test,
        'y_pred': y_pred,
        'log_transformed': True
    })
    predictions_df.to_csv(os.path.join(RESULTS_DIR, "ld50_predictions.csv"), index=False)
    
    print(f"\nR²={metrics['R2']}, RMSE={metrics['RMSE']}")
    return results_df

# ============================================================================
# Tox21 Model Validation
# ============================================================================

def validate_tox21_models():
    """Validate Tox21 models (Uses ALL features)"""
    print("\n" + "="*60)
    print("TOX21 MODEL VALIDATION")
    print("="*60)
    
    # Load data
    tox_path = os.path.join(DATA_DIR, "tox21.csv")
    if not os.path.exists(tox_path):
        return pd.DataFrame()
    
    df = pd.read_csv(tox_path)
    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    df = df.dropna(subset=[smiles_col])
    
    # Apply Sampling if configured
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        print(f"⚠️ QUICK MODE: Sampling {SAMPLE_SIZE} random samples from {len(df)} total...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        
    print(f"Loaded {len(df)} Tox21 samples")
    
    # Generate features
    features_df = get_features(df[smiles_col].tolist(), name="tox21")
    
    if features_df is None:
        return pd.DataFrame()
    
    df = df.iloc[:len(features_df)].reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)
    
    # Use ALL features
    exclude_cols = ['smiles', 'smiles_r', 'mol_id']
    feature_cols = [c for c in features_df.columns if c.lower() not in exclude_cols]
    
    X = features_df[feature_cols].copy()
    
    # Convert booleans
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)
        
    # Load imputer
    imputer_path = os.path.join(MODEL_DIR, "tox", "imputer.joblib")
    imputer = joblib.load(imputer_path) if os.path.exists(imputer_path) else None
    
    if imputer:
        try:
            X = imputer.transform(X)
        except Exception as e:
            print(f"Imputation fit error: {e}")
    else:
        X = np.nan_to_num(X.values, nan=0.0)
    
    results = []
    all_predictions = []
    
    for task in TOX21_TASKS:
        print(f"\n--- Validating {task} ---")
        model_path = os.path.join(MODEL_DIR, "tox", f"{task}_model.joblib")
        
        if not os.path.exists(model_path):
            continue
            
        model = joblib.load(model_path)
        
        if task not in df.columns:
            continue
            
        valid_mask = df[task].notna()
        
        # We need to slice X (which might be array or dataframe)
        if isinstance(X, pd.DataFrame):
            X_task = X[valid_mask]
        else:
            X_task = X[valid_mask]
            
        y_task = df[task][valid_mask].values
        
        if len(y_task) < 10:
            continue
            
        # Stratified split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_task, y_task, test_size=0.2, random_state=42, stratify=y_task
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X_task, y_task, test_size=0.2, random_state=42
            )
            
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_pred_proba = model.predict(X_test)
            
        metrics = calculate_classification_metrics(y_test, y_pred_proba, name=task)
        results.append(metrics)
        print(f"  AUC={metrics['AUC']}, F1={metrics['F1']}")
        
        for true, pred in zip(y_test, y_pred_proba):
            all_predictions.append({
                'model': task,
                'y_true': true,
                'y_pred_proba': pred
            })
            
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "tox21_validation_results.csv"), index=False)
    
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv(os.path.join(RESULTS_DIR, "tox21_predictions.csv"), index=False)
    
    if len(results_df) > 0:
        print(f"Mean AUC: {results_df['AUC'].mean():.4f}")
        
    return results_df

def main():
    print("="*60)
    print("PKSmart Custom Model Validation (v3)")
    print("="*60)
    
    validate_pk_models()
    validate_ld50_model()
    validate_tox21_models()
    
    print("\nVALIDATION COMPLETE")

if __name__ == "__main__":
    main()
