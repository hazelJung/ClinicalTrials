import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from pksmart.features import generate_features

# Configuration
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

ANIMAL_DATA_PATH = os.path.join(DATA_DIR, "Animal_PK_data.csv")
HUMAN_DATA_PATH = os.path.join(DATA_DIR, "Human_PK_data.csv")

# Species and Parameters
SPECIES = ["rat", "dog", "monkey"]
PARAMS = ["CL_mL_min_kg", "VDss_L_kg", "fup"]

HUMAN_PARAMS = ["human_CL_mL_min_kg", "human_VDss_L_kg", "human_fup", "human_mrt", "human_thalf"]

def train_animal_models():
    print("Loading Animal Data...")
    df = pd.read_csv(ANIMAL_DATA_PATH)
    
    # Pre-filter
    df = df.dropna(subset=['smiles_r'])
    df['smiles_r'] = df['smiles_r'].astype(str)
    
    # Generate features for all molecules in animal data
    # We use 'smiles_r' column
    print("Generating features for Animal data (this may take a while)...")
    features_df = generate_features(df['smiles_r'].tolist())
    
    # Ensure feature names are strings (mix of int Morgan bits and str Mordred names causes issues)
    features_df.columns = features_df.columns.astype(str)
    
    if features_df is None:
        raise ValueError("Feature generation failed for animal data.")
    
    # Align dataframes
    df = df.iloc[:len(features_df)].reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)
    
    animal_models = {}
    
    for species in SPECIES:
        for param in PARAMS:
            col_name = f"{species}_{param}"
            model_path = os.path.join(MODEL_DIR, f"{species}_{param}_model.joblib")
            
            if os.path.exists(model_path):
                print(f"✅ Model {species} {param} already exists. Skipping.")
                continue

            if col_name not in df.columns:
                print(f"Warning: Column {col_name} not found. Skipping.")
                continue
            
            print(f"Training {species} {param} model...")
            
            # Filter valid data for this specific parameter
            valid_mask = df[col_name].notna()
            y = df[col_name][valid_mask]
            
            # Apply Log10 transformation for non-fup parameters
            if "fup" not in param:
                 # Check for non-positive values before log
                 if (y <= 0).any():
                     print(f"Warning: Non-positive values found in {col_name}. Dropping them.")
                     valid_mask = valid_mask & (df[col_name] > 0)
                     y = df[col_name][valid_mask]
                     X = features_df[valid_mask]
                 else:
                     X = features_df[valid_mask]
                 y = np.log10(y)
            else:
                 X = features_df[valid_mask]
            
            # Simple Imputation for NaN features (if any)
            # Although Mordred usually returns numbers or errors (handled in features.py)
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X.select_dtypes(include=[np.number]))
            
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_imputed, y)
            
            # Save model and imputer
            model_path = os.path.join(MODEL_DIR, f"{species}_{param}_model.joblib")
            imputer_path = os.path.join(MODEL_DIR, f"{species}_{param}_imputer.joblib")
            joblib.dump(model, model_path)
            joblib.dump(imputer, imputer_path)
            print(f"Saved {model_path}")

def train_human_models():
    print("Loading Human Data...")
    df = pd.read_csv(HUMAN_DATA_PATH)
    
    # Pre-filter
    df = df.dropna(subset=['smiles_r'])
    df['smiles_r'] = df['smiles_r'].astype(str)
    
    print("Generating features for Human data...")
    features_df = generate_features(df['smiles_r'].tolist())
    
    # Ensure feature names are strings
    features_df.columns = features_df.columns.astype(str)
    
    if features_df is None:
        raise ValueError("Feature generation failed for human data.")
        
    df = df.iloc[:len(features_df)].reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)
    
    # ---------------------------------------------------------
    # Two-Stage Approach: Predict Animal Params for Human Molecules
    # ---------------------------------------------------------
    print("Predicting Animal Priors for Human molecules...")
    animal_preds = pd.DataFrame()
    
    for species in SPECIES:
        for param in PARAMS:
            model_name = f"{species}_{param}"
            model_path = os.path.join(MODEL_DIR, f"{model_name}_model.joblib")
            imputer_path = os.path.join(MODEL_DIR, f"{model_name}_imputer.joblib")
            
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                imputer = joblib.load(imputer_path)
                
                X_imputed = imputer.transform(features_df.select_dtypes(include=[np.number]))
                preds = model.predict(X_imputed)
                animal_preds[model_name] = preds
    
    # Combine Molecular Features + Animal Predictions
    X_combined = pd.concat([features_df.select_dtypes(include=[np.number]), animal_preds], axis=1)
    
    # Train Human Models
    for param in HUMAN_PARAMS:
        save_name = param.replace("human_", "human_") # keep human_ prefix or strictly follow file naming convention
        model_path = os.path.join(MODEL_DIR, f"{save_name}_model.joblib")
        
        if os.path.exists(model_path):
             print(f"✅ Model {save_name} already exists. Skipping.")
             continue
             
        target_col = param
        if param not in df.columns:
            # Map simplified names for MRT/Thalf if necessary or skip
            # The column names in CSV usually match, see checking earlier.
            if param == "human_mrt" and "human_MRT_h" in df.columns: # Adjust based on actual CSV
                 target_col = "human_MRT_h"
            elif param == "human_thalf" and "human_thalf_h" in df.columns:
                 target_col = "human_thalf_h"
                 
        if target_col not in df.columns:
             print(f"Warning: Human target {target_col} not found.")
             continue

        print(f"Training Human {param} model...")
        valid_mask = df[target_col].notna()
        y = df[target_col][valid_mask]
        
        # Apply Log10 transformation for non-fup parameters
        if "fup" not in param:
             if (y <= 0).any():
                 print(f"Warning: Non-positive values found in {target_col}. Dropping them.")
                 valid_mask = valid_mask & (df[target_col] > 0)
                 y = df[target_col][valid_mask]
                 X = X_combined[valid_mask]
             else:
                 X = X_combined[valid_mask]
             y = np.log10(y)
        else:
             X = X_combined[valid_mask]
        
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X.select_dtypes(include=[np.number]))
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_imputed, y)
        
        save_name = param.replace("human_", "human_") # keep human_ prefix or strictly follow file naming convention
        # We will use 'human_{param}' naming for simplicity
        
        model_path = os.path.join(MODEL_DIR, f"{save_name}_model.joblib")
        imputer_path = os.path.join(MODEL_DIR, f"{save_name}_imputer.joblib")
        joblib.dump(model, model_path)
        joblib.dump(imputer, imputer_path)
        print(f"Saved {model_path}")

if __name__ == "__main__":
    print("Starting PKSmart Retraining Pipeline...")
    train_animal_models()
    train_human_models()
    print("All models trained and saved successfully!")
