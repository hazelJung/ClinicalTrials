import argparse
import os
import joblib
import pandas as pd
import numpy as np
from pksmart.features import generate_features
from pksmart.models import PKSmartPipeline

# Tox21 Task Names (12 endpoints)
TOX21_TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

def load_toxicity_models(model_dir):
    """Load all trained toxicity models from models/tox/"""
    tox_dir = os.path.join(model_dir, "models", "tox")
    if not os.path.exists(tox_dir):
        return None, None
    
    models = {}
    for task in TOX21_TASKS:
        model_path = os.path.join(tox_dir, f"{task}_model.joblib")
        if os.path.exists(model_path):
            models[task] = joblib.load(model_path)
    
    imputer_path = os.path.join(tox_dir, "imputer.joblib")
    imputer = joblib.load(imputer_path) if os.path.exists(imputer_path) else None
    
    return models, imputer

def load_ld50_model(model_dir):
    """Load trained LD50 regression model from models/ld50/"""
    ld50_dir = os.path.join(model_dir, "models", "ld50")
    model_path = os.path.join(ld50_dir, "ld50_model.joblib")
    imputer_path = os.path.join(ld50_dir, "imputer.joblib")
    
    if not os.path.exists(model_path):
        return None, None
    
    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path) if os.path.exists(imputer_path) else None
    
    return model, imputer

def get_toxicity_class(ld50):
    """Assign EPA/GHS toxicity class based on LD50 value (mg/kg)"""
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

def predict_ld50(features_df, model, imputer):
    """Predict LD50 values"""
    if model is None:
        return None
    
    # Prepare features
    feature_cols = [c for c in features_df.columns if c != 'smiles_r']
    X = features_df[feature_cols].values
    
    # Apply imputer
    if imputer:
        X = imputer.transform(X)
    else:
        X = np.nan_to_num(X, nan=0.0)
    
    # Predict (model outputs log10(LD50))
    y_pred_log = model.predict(X)
    y_pred = 10 ** y_pred_log - 1  # Convert back to original scale
    
    return y_pred

def predict_toxicity(features_df, tox_models, imputer):
    """Run toxicity predictions for all loaded models"""
    if not tox_models:
        return None
    
    # Prepare features (same as PK but need to select only feature columns)
    feature_cols = [c for c in features_df.columns if c != 'smiles_r']
    X = features_df[feature_cols].values
    
    # Apply imputer (handle NaNs)
    if imputer:
        X = imputer.transform(X)
    else:
        # Fallback: simple mean imputation
        X = np.nan_to_num(X, nan=0.0)
    
    # Predict for each task
    results = {}
    for task, model in tox_models.items():
        try:
            # Get probability of positive class (toxic)
            proba = model.predict_proba(X)[:, 1]
            results[f"tox_{task}"] = proba
        except Exception as e:
            print(f"Warning: Toxicity prediction failed for {task}: {e}")
            results[f"tox_{task}"] = [np.nan] * len(X)
    
    return pd.DataFrame(results)

def format_tox_output(tox_df):
    """Format toxicity predictions with risk levels"""
    formatted = pd.DataFrame()
    
    for col in tox_df.columns:
        task_name = col.replace("tox_", "")
        proba = tox_df[col]
        
        # Add probability
        formatted[f"{task_name} (Prob)"] = proba.round(3)
        
        # Add risk level
        def get_risk(p):
            if pd.isna(p): return "N/A"
            if p >= 0.7: return "HIGH"
            if p >= 0.3: return "MEDIUM"
            return "LOW"
        
        formatted[f"{task_name} (Risk)"] = proba.apply(get_risk)
    
    return formatted

def main():
    parser = argparse.ArgumentParser(description="PKSmart: Predict Human Pharmacokinetics + Toxicity from SMILES")
    parser.add_argument("--smiles", type=str, help="Single SMILES string to predict")
    parser.add_argument("--input", type=str, help="Path to input CSV file containing a 'smiles' column")
    parser.add_argument("--output", type=str, default="pksmart_predictions.csv", help="Path to output CSV file")
    parser.add_argument("--model_dir", type=str, default=".", help="Directory containing trained models")
    parser.add_argument("--tox", action="store_true", help="Enable Tox21 toxicity prediction (requires models/tox/)")
    parser.add_argument("--ld50", action="store_true", help="Enable LD50 acute toxicity prediction (requires models/ld50/)")
    parser.add_argument("--all", action="store_true", help="Enable all predictions (PK + Tox21 + LD50)")
    parser.add_argument("--tox_only", action="store_true", help="Run toxicity prediction only (skip PK)")
    
    args = parser.parse_args()
    
    # --all enables all predictions
    if args.all:
        args.tox = True
        args.ld50 = True
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    
    # 1. Load Data
    smiles_list = []
    if args.smiles:
        smiles_list = [args.smiles]
        print(f"Processing single SMILES: {args.smiles}")
    elif args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found.")
            return
        df = pd.read_csv(args.input)
        if 'smiles' not in df.columns:
             # Try common variations
             for col in ['SMILES', 'Smiles', 'smile']:
                 if col in df.columns:
                     df.rename(columns={col: 'smiles'}, inplace=True)
                     break
        
        if 'smiles' not in df.columns:
            print("Error: Input CSV must contain a 'smiles' column.")
            return
        smiles_list = df['smiles'].tolist()
        print(f"Processing {len(smiles_list)} molecules from {args.input}")
    else:
        print("Error: Please provide either --smiles or --input")
        parser.print_help()
        return

    # 2. Generate Features (shared between PK and Toxicity)
    print("\n[Step 1/4] Generating Features (Standardization + Morgan + Mordred)...")
    features_df = generate_features(smiles_list)
    
    if features_df is None or features_df.empty:
        print("Error: Feature generation failed (invalid SMILES?)")
        return
    
    # Start building final output
    final_output = pd.DataFrame({'smiles': features_df['smiles_r']})
    
    # 3. PK Prediction (unless tox_only mode)
    if not args.tox_only:
        print("\n[Step 2/4] Running PK Predictions (Animal -> Human)...")
        
        if not os.path.exists(args.model_dir):
            print(f"Error: Model directory '{args.model_dir}' not found.")
            return

        pipeline = PKSmartPipeline(os.path.join(args.model_dir, "models"))
        results_df = pipeline.run_pipeline(features_df)
        
        # Format PK results
        def get_sort_key(col_name):
            if col_name.startswith("rat_"): return 0
            if col_name.startswith("dog_"): return 1
            if col_name.startswith("monkey_"): return 2
            if col_name.startswith("human_"): return 3
            return 4 
            
        sorted_cols = sorted(results_df.columns, key=get_sort_key)
        
        for col in sorted_cols:
            val = results_df[col]
            # fup is not log transformed
            if 'fup' in col:
                final_output[col] = val
            else:
                final_output[f"{col} (Log)"] = val
                final_output[f"{col} (Linear)"] = 10 ** val
    else:
        print("\n[Step 2/4] Skipping PK predictions (--tox_only mode)")

    # 4. Tox21 Toxicity Prediction
    if args.tox or args.tox_only:
        print("\n[Step 3/4] Running Tox21 Predictions (12 endpoints)...")
        
        tox_models, tox_imputer = load_toxicity_models(args.model_dir)
        
        if not tox_models:
            print("  Warning: Tox21 models not found in models/tox/")
            print("  Run 'python pksmart_train_tox.py' first to train toxicity models.")
        else:
            print(f"  Loaded {len(tox_models)} Tox21 models")
            tox_df = predict_toxicity(features_df, tox_models, tox_imputer)
            
            if tox_df is not None:
                tox_formatted = format_tox_output(tox_df)
                
                # Add separator column
                final_output["---"] = "---"
                final_output["TOX21"] = "TOX21"
                
                # Add toxicity results
                for col in tox_formatted.columns:
                    final_output[col] = tox_formatted[col].values
    else:
        print("\n[Step 3/4] Skipping Tox21 (use --tox flag to enable)")

    # 5. LD50 Acute Toxicity Prediction
    if args.ld50:
        print("\n[Step 4/4] Running LD50 Prediction (Acute Toxicity)...")
        
        ld50_model, ld50_imputer = load_ld50_model(args.model_dir)
        
        if ld50_model is None:
            print("  Warning: LD50 model not found in models/ld50/")
            print("  Run 'python pksmart_train_ld50.py' first to train the LD50 model.")
        else:
            print("  Loaded LD50 regression model")
            ld50_values = predict_ld50(features_df, ld50_model, ld50_imputer)
            
            if ld50_values is not None:
                # Add separator
                final_output["----"] = "----"
                final_output["LD50"] = "LD50"
                
                # Add LD50 value
                final_output["LD50 (mg/kg)"] = np.round(ld50_values, 2)
                
                # Add toxicity class
                classes = []
                labels = []
                for val in ld50_values:
                    cls, lbl = get_toxicity_class(val)
                    classes.append(cls)
                    labels.append(lbl)
                
                final_output["Tox Class"] = classes
                final_output["Tox Label"] = labels
    else:
        print("\n[Step 4/4] Skipping LD50 (use --ld50 flag to enable)")

    # 6. Save/Display Results
    print("\n" + "="*60)
    print("=== PREDICTION RESULTS ===")
    print("="*60)
    
    if args.smiles:
        # Single molecule: display transposed
        print(final_output.iloc[0].T.to_string())
        output_file = "single_prediction_output.csv"
        # Avoid creating multiple output files if output arg is specified
        if args.output != "pksmart_predictions.csv":
             output_file = args.output
             
        final_output.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    else:
        final_output.to_csv(args.output, index=False)
        print(final_output.head())
        print(f"\n... ({len(final_output)} total)")
        print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
