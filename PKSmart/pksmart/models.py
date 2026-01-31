import os
import joblib
import pandas as pd
import numpy as np

class PKSmartPipeline:
    def __init__(self, model_dir):
        # Use the provided model_dir directly
        self.model_dir = model_dir
        
        self.models = {}
        self.imputers = {}
        
        self.species = ['rat', 'dog', 'monkey']
        self.params = ['CL_mL_min_kg', 'VDss_L_kg', 'fup']
        self.human_params = ['human_CL_mL_min_kg', 'human_VDss_L_kg', 'human_fup', 'human_mrt', 'human_thalf']
        
        self._load_assets()

    def _load_assets(self):
        """Loads models and imputers."""
        # Load Animal Models
        for species in self.species:
            for param in self.params:
                name = f"{species}_{param}"
                self._load_single_asset(name)

        # Load Human Models
        for param in self.human_params:
            self._load_single_asset(param)

    def _load_single_asset(self, name):
        model_path = os.path.join(self.model_dir, f"{name}_model.joblib")
        imputer_path = os.path.join(self.model_dir, f"{name}_imputer.joblib")
        
        if os.path.exists(model_path):
            self.models[name] = joblib.load(model_path)
        else:
            print(f"Warning: Model {name} not found at {model_path}")
            
        if os.path.exists(imputer_path):
            self.imputers[name] = joblib.load(imputer_path)
        else:
            print(f"Warning: Imputer {name} not found at {imputer_path}")

    def predict_animal_pk(self, features_df):
        """Predicts animal PK parameters."""
        # Ensure feature names are strings to avoid sklearn error
        features_df.columns = features_df.columns.astype(str)
        predictions = {}
        
        for species in self.species:
            for param in self.params:
                name = f"{species}_{param}"
                
                if name in self.models and name in self.imputers:
                    model = self.models[name]
                    imputer = self.imputers[name]
                    
                    try:
                        # Select only numeric columns for imputation
                        # Using numpy select_dtypes equivalent if features_df is all numeric or handling mixed
                        X = features_df.select_dtypes(include=[np.number])
                        
                        # Impute
                        X_imputed = imputer.transform(X)
                        
                        # Predict
                        preds = model.predict(X_imputed)
                        predictions[name] = preds
                        
                    except Exception as e:
                        print(f"Error predicting {name}: {e}")
                        predictions[name] = np.zeros(len(features_df))
                else:
                    predictions[name] = np.zeros(len(features_df))
        
        return pd.DataFrame(predictions)

    def predict_human_pk(self, features_df, animal_preds_df):
        """Predicts human PK parameters using features + animal predictions."""
        
        # Combine molecular features and animal predictions
        # Animal predictions are used as features for the human model (Two-Stage)
        combined_df = pd.concat([features_df.select_dtypes(include=[np.number]).reset_index(drop=True), 
                                 animal_preds_df.reset_index(drop=True)], axis=1)
        
        # Ensure feature names are strings
        combined_df.columns = combined_df.columns.astype(str)
        
        results = {}
        for param in self.human_params:
            if param in self.models and param in self.imputers:
                model = self.models[param]
                imputer = self.imputers[param]
                
                try:
                    # Impute
                    X_imputed = imputer.transform(combined_df)
                    
                    # Predict
                    preds = model.predict(X_imputed)
                    results[param] = preds
                except Exception as e:
                    import traceback
                    print(f"Error predicting {param}: {e}")
                    traceback.print_exc()
                    results[param] = np.zeros(len(features_df))
            else:
                results[param] = np.zeros(len(features_df))
        
        return pd.DataFrame(results)

    def run_pipeline(self, features_df):
        # Predict Animal Params
        animal_preds = self.predict_animal_pk(features_df)
        
        # Predict Human Params
        human_preds = self.predict_human_pk(features_df, animal_preds)
        
        return pd.concat([animal_preds, human_preds], axis=1)
