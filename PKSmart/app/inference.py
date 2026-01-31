import os
import joblib
import pandas as pd
import numpy as np
from pksmart.features import generate_features
from pksmart.models import PKSmartPipeline
from rdkit import Chem
from rdkit.Chem import Descriptors

# New Import for PBPK
from pksmart.mpbpk_engine import DrugParams, TargetParams, simulate_population_cohort

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Tox21 Task Names
TOX21_TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

class PKSmartInference:
    def __init__(self):
        self.pipeline = None
        self.tox_models = None
        self.tox_imputer = None
        self.ld50_model = None
        self.ld50_imputer = None
        self.load_models()

    def load_models(self):
        print("Loading models...")
        # 1. PK Pipeline
        if os.path.exists(MODEL_DIR): 
             self.pipeline = PKSmartPipeline(MODEL_DIR)
        else:
             print(f"Warning: Model directory {MODEL_DIR} not found.")

        # 2. Tox21 Models
        tox_dir = os.path.join(MODEL_DIR, "tox")
        self.tox_models = {}
        if os.path.exists(tox_dir):
            for task in TOX21_TASKS:
                path = os.path.join(tox_dir, f"{task}_model.joblib")
                if os.path.exists(path):
                    self.tox_models[task] = joblib.load(path)
            imp_path = os.path.join(tox_dir, "imputer.joblib")
            if os.path.exists(imp_path):
                self.tox_imputer = joblib.load(imp_path)
        
        # 3. LD50 Model
        ld50_dir = os.path.join(MODEL_DIR, "ld50")
        ld50_path = os.path.join(ld50_dir, "ld50_model.joblib")
        if os.path.exists(ld50_path):
            self.ld50_model = joblib.load(ld50_path)
            imp_path = os.path.join(ld50_dir, "imputer.joblib")
            if os.path.exists(imp_path):
                self.ld50_imputer = joblib.load(imp_path)
        print(f"Models loaded. LD50 Model: {'Loaded' if self.ld50_model else 'Not Found'}")

    def predict(self, smiles: str):
        # 1. Generate Features
        try:
            features = generate_features([smiles])
            if features is None or features.empty:
                return {"error": "Invalid SMILES or feature generation failed"}
            
            # 1.5 Calculate Physicochemical Properties
            mol = Chem.MolFromSmiles(smiles)
            props = {
                "mw": round(Descriptors.MolWt(mol), 2),
                "logp": round(Descriptors.MolLogP(mol), 2),
                "tpsa": round(Descriptors.TPSA(mol), 2),
                "hbd": Descriptors.NumHDonors(mol),
                "hba": Descriptors.NumHAcceptors(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol)
            }
        except Exception as e:
            return {"error": str(e)}

        result = {
            "smiles": smiles,
            "properties": props,
            "pk": {},
            "tox21": {},
            "ld50": {}
        }

        # 2. PK Prediction
        try:
            if self.pipeline:
                pk_df = self.pipeline.run_pipeline(features)
                for col in pk_df.columns:
                    val = pk_df[col].iloc[0]
                    result["pk"][col] = float(val)
                    if not 'fup' in col:
                         result["pk"][f"{col}_linear"] = float(10**val)
        except Exception as e:
             print(f"PK Prediction failed: {e}")

        # 3. Tox21 Prediction
        if self.tox_models:
            feature_cols = [c for c in features.columns if c != 'smiles_r']
            X = features[feature_cols].values
            if self.tox_imputer:
                 X = self.tox_imputer.transform(X)
            
            for task, model in self.tox_models.items():
                prob = float(model.predict_proba(X)[0][1])
                # Risk Classification matching pksmart_predict.py
                risk = "LOW"
                if prob >= 0.7: risk = "HIGH"
                elif prob >= 0.3: risk = "MEDIUM"
                
                result["tox21"][task] = {
                    "probability": prob,
                    "risk": risk
                }
        
        # 4. LD50 Prediction
        if self.ld50_model:
            feature_cols = [c for c in features.columns if c != 'smiles_r']
            X = features[feature_cols].values
            if self.ld50_imputer:
                 X = self.ld50_imputer.transform(X)
            
            y_pred_log = self.ld50_model.predict(X)[0]
            ld50_val = 10 ** float(y_pred_log) - 1 # Use formula from pksmart_predict.py
            
            # Classification Logic (GHS) matching pksmart_predict.py
            tox_class = "VI"
            tox_label = "Non-toxic"
            if ld50_val <= 5: tox_class, tox_label = "I", "Fatal"
            elif ld50_val <= 50: tox_class, tox_label = "II", "Fatal"
            elif ld50_val <= 300: tox_class, tox_label = "III", "Toxic"
            elif ld50_val <= 2000: tox_class, tox_label = "IV", "Harmful"
            elif ld50_val <= 5000: tox_class, tox_label = "V", "May be harmful"
            
            result["ld50"]["LD50_mgkg"] = float(ld50_val)
            result["ld50"]["value"] = round(ld50_val, 2)
            result["ld50"]["class"] = tox_class
            result["ld50"]["label"] = tox_label
        else:
             result["ld50"]["value"] = "N/A"
             result["ld50"]["class"] = "N/A"
             result["ld50"]["label"] = "N/A"

        return result
    
# Global Inference Object
engine = PKSmartInference()

# Sanitize helper (Already added)
def _sanitize_data(data):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(data, dict):
        return {k: _sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize_data(v) for v in data]
    elif isinstance(data, (np.integer, int)):
        return int(data)
    elif isinstance(data, (np.floating, float)):
        return float(data)
    elif isinstance(data, (np.bool_, bool)):
        return bool(data)
    elif isinstance(data, np.ndarray):
        return _sanitize_data(data.tolist())
    else:
        return data

def run_cohort_simulation(
    cl_ml_min_kg: float, 
    vd_l_kg: float, 
    mw_g_mol: float, 
    fup: float, 
    pop: str = "EUR", 
    n_subjects: int = 1000,
    age_min: int = 18,
    age_max: int = 80,
    female_ratio: float = 0.5,
    dose_mg: float = 100.0
):
    """
    Runs the mPBPK virtual cohort simulation.
    Handles unit conversion and calls the engine.
    """
    print(f"Running Cohort Sim: Pop={pop}, N={n_subjects}, CL={cl_ml_min_kg}, Vd={vd_l_kg}")
    
    # 1. Unit Conversions
    # CL (mL/min/kg) -> (L/day) for 70kg (Simulation Engine standardizes usually, but let's follow run_pbpk_study.py logic)
    patient_weight = 70.0
    cl_L_day = cl_ml_min_kg * patient_weight * 1.44
    vd_L = vd_l_kg * patient_weight
    mw_kda = mw_g_mol / 1000.0
    
    # 2. Setup Params
    drug = DrugParams(dose_mg=dose_mg, MW_kDa=mw_kda)
    target = TargetParams()
    
    # 3. Run Simulation
    try:
        results = simulate_population_cohort(
            drug=drug, 
            target=target, 
            population=pop,
            n_subjects=n_subjects,
            predicted_CL_L_day=cl_L_day, 
            predicted_Vd_L=vd_L,
            predicted_Fu=fup,
            gender_mode="BOTH", # Front-end slider implies mixture
            age_min=age_min,
            age_max=age_max,
            female_ratio=female_ratio
        )
        return _sanitize_data(results)
    except Exception as e:
        print(f"Simulation Error: {e}")
        return {"error": str(e)}

def run_dose_response_analysis(
    cl_ml_min_kg: float, 
    vd_l_kg: float, 
    mw_g_mol: float, 
    fup: float, 
    base_dose_mg: float = 100.0
):
    """
    Runs a lightweight dose-response analysis for the Heatmap.
    Simulates 5 dose levels using a small representative cohort.
    """
    results = []
    doses = [base_dose_mg * x for x in [0.25, 0.5, 1.0, 2.0, 4.0]]
    
    # Representative 'Average' parameters
    patient_weight = 70.0
    cl_L_day = cl_ml_min_kg * patient_weight * 1.44
    vd_L = vd_l_kg * patient_weight
    mw_kda = mw_g_mol / 1000.0
    target = TargetParams()

    for dose in doses:
        drug = DrugParams(dose_mg=dose, MW_kDa=mw_kda)
        # Run small simulation (N=5 is enough for trend)
        cohort_res = simulate_population_cohort(
            drug=drug, target=target, population="EUR", n_subjects=5,
            predicted_CL_L_day=cl_L_day, predicted_Vd_L=vd_L, predicted_Fu=fup
        )[0] # Just take the first/valid one or mean? 
        
        # Actually simulate_population_cohort returns a list of results.
        # Let's take the mean of the small cohort
        sims = simulate_population_cohort(
            drug=drug, target=target, population="EUR", n_subjects=5,
            predicted_CL_L_day=cl_L_day, predicted_Vd_L=vd_L, predicted_Fu=fup
        )
        if hasattr(sims, '__iter__') and len(sims) > 0:
            mean_to = np.mean([s['mean_to'] for s in sims])
            results.append({'dose': round(dose, 1), 'mean_to': round(mean_to, 1)})
    
    return results
