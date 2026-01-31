"""
PKSmart Phase 3: LD50 (Acute Toxicity) Model Training
------------------------------------------------------
Downloads LD50 data from TDC (Therapeutics Data Commons) directly (no PyTDC needed!)
and trains a regression model to predict LD50 in mg/kg.
"""
import os
import requests
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pksmart.features import standardize_smiles, calculate_morgan_fingerprints, calculate_mordred_descriptors

# TDC LD50_Zhu Dataset - Harvard Dataverse
LD50_DATAVERSE_ID = 4267146
LD50_DOWNLOAD_URL = f"https://dataverse.harvard.edu/api/access/datafile/{LD50_DATAVERSE_ID}"

# EPA Toxicity Classes based on LD50 (mg/kg) for oral rat
def get_toxicity_class(ld50):
    """
    Assign EPA/GHS toxicity class based on LD50 value.
    Class I: ≤5 mg/kg (Fatal if swallowed)
    Class II: 5-50 mg/kg (Fatal if swallowed)
    Class III: 50-300 mg/kg (Toxic if swallowed)
    Class IV: 300-2000 mg/kg (Harmful if swallowed)
    Class V: 2000-5000 mg/kg (May be harmful if swallowed)
    Class VI: >5000 mg/kg (Non-toxic)
    """
    if ld50 <= 5:
        return "I", "Fatal"
    elif ld50 <= 50:
        return "II", "Fatal"
    elif ld50 <= 300:
        return "III", "Toxic"
    elif ld50 <= 2000:
        return "IV", "Harmful"
    elif ld50 <= 5000:
        return "V", "May be harmful"
    else:
        return "VI", "Non-toxic"

def download_ld50_data():
    """Download LD50 dataset directly from Harvard Dataverse"""
    print("Downloading LD50 dataset from Harvard Dataverse (TDC)...")
    
    try:
        response = requests.get(LD50_DOWNLOAD_URL, timeout=60)
        response.raise_for_status()
        
        # Save to data folder
        os.makedirs("data", exist_ok=True)
        
        # TDC stores as .tab format (TSV)
        with open("data/ld50_zhu.tab", "wb") as f:
            f.write(response.content)
        
        print(f"  Downloaded to data/ld50_zhu.tab")
        
        # Load and convert to CSV
        df = pd.read_csv("data/ld50_zhu.tab", sep="\t")
        
        # Rename columns to standard names
        # TDC format: Drug (SMILES), Y (LD50 value)
        if 'Drug' in df.columns:
            df = df.rename(columns={'Drug': 'smiles', 'Y': 'LD50'})
        elif 'X' in df.columns:
            df = df.rename(columns={'X': 'smiles', 'Y': 'LD50'})
        
        # Also save as CSV for convenience
        df.to_csv("data/ld50.csv", index=False)
        print(f"  Converted to data/ld50.csv ({len(df)} compounds)")
        
        return df
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Please download manually from: https://tdcommons.ai/single_pred_tasks/tox/#acute-toxicity-ld50")
        return None

def train_ld50_model():
    print("=" * 60)
    print("PKSmart Phase 3: LD50 (Acute Toxicity) Model Training")
    print("=" * 60)
    
    # 1. Load or Download Data
    print("\n[1/5] Loading LD50 dataset...")
    
    if os.path.exists("data/ld50.csv"):
        df = pd.read_csv("data/ld50.csv")
        print(f"  Loaded {len(df)} compounds from data/ld50.csv")
    elif os.path.exists("data/ld50_zhu.tab"):
        df = pd.read_csv("data/ld50_zhu.tab", sep="\t")
        if 'Drug' in df.columns:
            df = df.rename(columns={'Drug': 'smiles', 'Y': 'LD50'})
        print(f"  Loaded {len(df)} compounds from data/ld50_zhu.tab")
    else:
        df = download_ld50_data()
        if df is None:
            return
    
    # Check column names
    if 'smiles' not in df.columns:
        # Try to find SMILES column
        for col in ['Drug', 'SMILES', 'Smiles', 'drug', 'X']:
            if col in df.columns:
                df = df.rename(columns={col: 'smiles'})
                break
    
    if 'LD50' not in df.columns:
        for col in ['Y', 'y', 'ld50', 'Label']:
            if col in df.columns:
                df = df.rename(columns={col: 'LD50'})
                break
    
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  LD50 range: {df['LD50'].min():.2f} - {df['LD50'].max():.2f} mg/kg")
    
    # 2. Standardize SMILES
    print("\n[2/5] Standardizing SMILES...")
    df['smiles_r'] = df['smiles'].apply(standardize_smiles)
    
    n_total = len(df)
    df = df.dropna(subset=['smiles_r', 'LD50']).reset_index(drop=True)
    n_valid = len(df)
    print(f"  Valid SMILES: {n_valid}/{n_total}")
    
    # 3. Generate Features
    print("\n[3/5] Generating Features (Morgan + Mordred)...")
    valid_smiles = df['smiles_r'].tolist()
    
    print("  Calculating Morgan Fingerprints...")
    df_morgan = calculate_morgan_fingerprints(valid_smiles)
    
    print("  Calculating Mordred Descriptors...")
    df_mordred = calculate_mordred_descriptors(valid_smiles)
    
    # Combine Features (X)
    X = pd.concat([df_mordred, df_morgan], axis=1)
    
    # Target: log10(LD50) for better regression (LD50 is log-normally distributed)
    y = np.log10(df['LD50'].values + 1)  # +1 to avoid log(0)
    
    # Impute missing values
    print("  Imputing missing feature values...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Save imputer
    os.makedirs("models/ld50", exist_ok=True)
    joblib.dump(imputer, "models/ld50/imputer.joblib")
    
    # 4. Train/Test Split & Train Model
    print("\n[4/5] Training LD50 Regression Model (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    # Metrics in log10 space
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae_log = mean_absolute_error(y_test, y_pred)
    
    print(f"\n  === Log10 Scale Metrics ===")
    print(f"  RMSE: {rmse_log:.4f}")
    print(f"  MAE:  {mae_log:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    # Convert back to original scale for interpretation
    y_test_orig = 10 ** y_test - 1
    y_pred_orig = 10 ** y_pred - 1
    rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
    
    print(f"\n  === Original Scale Metrics (mg/kg) ===")
    print(f"  RMSE: {rmse_orig:.2f} mg/kg")
    print(f"  MAE:  {mae_orig:.2f} mg/kg")
    
    # 5. Save Model
    print("\n[5/5] Saving model...")
    joblib.dump(model, "models/ld50/ld50_model.joblib")
    print("  Model saved to models/ld50/ld50_model.joblib")
    print("  Imputer saved to models/ld50/imputer.joblib")
    
    # Show sample predictions
    print("\n" + "=" * 70)
    print("Sample Predictions (Test Set)")
    print("=" * 70)
    print(f"{'Actual (mg/kg)':<18} {'Predicted (mg/kg)':<18} {'Class':<8} {'Label':<15}")
    print("-" * 70)
    
    n_samples = min(10, len(y_test_orig))
    sample_indices = np.random.choice(len(y_test_orig), n_samples, replace=False)
    
    for idx in sample_indices:
        actual = y_test_orig[idx]
        pred = y_pred_orig[idx]
        tox_class, tox_label = get_toxicity_class(pred)
        print(f"{actual:<18.2f} {pred:<18.2f} {tox_class:<8} {tox_label:<15}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return {
        'rmse_log': rmse_log,
        'mae_log': mae_log,
        'r2': r2,
        'rmse_orig': rmse_orig,
        'mae_orig': mae_orig
    }

if __name__ == "__main__":
    train_ld50_model()
