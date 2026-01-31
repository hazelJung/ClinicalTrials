"""
mPBPK ML Classifier - Decision Tree & Random Forest for Efficacy Prediction
===========================================================================
Trains ML models on mPBPK simulation data to predict drug-patient outcomes.

Features:
- Decision Tree (interpretable)
- Random Forest (performance)
- SMOTE for class imbalance
- Feature importance analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. SMOTE disabled.")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MLConfig:
    """ML training configuration."""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    use_smote: bool = True
    
    # Decision Tree parameters
    dt_max_depth: int = 8
    dt_min_samples_split: int = 20
    dt_min_samples_leaf: int = 10
    
    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_depth: int = 10


# =============================================================================
# Feature Engineering
# =============================================================================

FEATURE_COLUMNS = [
    'log_KD', 'log_dose', 'log_potency', 'charge', 'log_MW',
    'log_T0', 'log_halflife',
    'activity_score', 'cl_multiplier',
]

CATEGORICAL_COLUMNS = ['population', 'phenotype']

def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features for ML training.
    
    Returns:
        X: Feature matrix
        feature_names: List of feature names
    """
    X_list = []
    feature_names = FEATURE_COLUMNS.copy()
    
    # Numeric features
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            X_list.append(df[col].values.reshape(-1, 1))
        # Derived feature: log_potency (Dose / KD) -> log(Dose) - log(KD)
        elif col == 'log_potency':
            if 'log_dose' in df.columns and 'log_KD' in df.columns:
                potency = df['log_dose'].values - df['log_KD'].values
                X_list.append(potency.reshape(-1, 1))
            else:
                print("Warning: Cannot calculate log_potency. Missing inputs.")
    
    # One-hot encode population
    if 'population' in df.columns:
        pop_dummies = pd.get_dummies(df['population'], prefix='pop')
        for col in pop_dummies.columns:
            X_list.append(pop_dummies[col].values.reshape(-1, 1))
            feature_names.append(col)
    
    # Encode phenotype as ordinal
    if 'phenotype' in df.columns:
        phenotype_map = {'PM': 0, 'IM': 1, 'NM': 2, 'UM': 3}
        pheno_encoded = df['phenotype'].map(phenotype_map).fillna(2).values
        X_list.append(pheno_encoded.reshape(-1, 1))
        feature_names.append('phenotype_encoded')
    
    X = np.hstack(X_list)
    return X, feature_names


# =============================================================================
# ML Classifier
# =============================================================================

class mPBPKClassifier:
    """mPBPK efficacy classifier using Decision Tree and Random Forest."""
    
    def __init__(self, config: MLConfig = None):
        self.config = config or MLConfig()
        self.dt_model = None
        self.rf_model = None
        self.feature_names = None
        self.label_encoder = None
        self.results = {}
    
    def train(self, 
              data_path: Path = None,
              target_col: str = 'efficacy_success') -> Dict:
        """
        Train ML models on simulation data.
        
        Args:
            data_path: Path to training data CSV
            target_col: Target column name
            
        Returns:
            Dictionary with training results
        """
        # Load data
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'mpbpk_training_data_latest.csv'
        
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"Dataset size: {len(df)} samples")
        
        # Prepare features and labels
        X, self.feature_names = prepare_features(df)
        y = df[target_col].values
        
        print(f"Features: {len(self.feature_names)}")
        print(f"Class distribution: {np.bincount(y.astype(int))}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Apply SMOTE if available and enabled
        if SMOTE_AVAILABLE and self.config.use_smote:
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=self.config.random_state)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE: {len(X_train_balanced)} samples")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Train Decision Tree
        print("\n" + "=" * 50)
        print("Training Decision Tree...")
        self.dt_model = DecisionTreeClassifier(
            max_depth=self.config.dt_max_depth,
            min_samples_split=self.config.dt_min_samples_split,
            min_samples_leaf=self.config.dt_min_samples_leaf,
            random_state=self.config.random_state,
            class_weight='balanced'
        )
        self.dt_model.fit(X_train_balanced, y_train_balanced)
        
        dt_results = self._evaluate_model(self.dt_model, X_test, y_test, "Decision Tree")
        
        # Train Random Forest
        print("\n" + "=" * 50)
        print("Training Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            random_state=self.config.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        self.rf_model.fit(X_train_balanced, y_train_balanced)
        
        rf_results = self._evaluate_model(self.rf_model, X_test, y_test, "Random Forest")
        
        # Cross-validation
        print("\n" + "=" * 50)
        print(f"Running {self.config.cv_folds}-Fold Cross-Validation...")
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                            random_state=self.config.random_state)
        
        dt_cv_scores = cross_val_score(self.dt_model, X, y, cv=cv, scoring='f1')
        rf_cv_scores = cross_val_score(self.rf_model, X, y, cv=cv, scoring='f1')
        
        print(f"Decision Tree CV F1: {dt_cv_scores.mean():.3f} ± {dt_cv_scores.std():.3f}")
        print(f"Random Forest CV F1: {rf_cv_scores.mean():.3f} ± {rf_cv_scores.std():.3f}")
        
        # Store results
        self.results = {
            'decision_tree': dt_results,
            'random_forest': rf_results,
            'dt_cv_scores': dt_cv_scores,
            'rf_cv_scores': rf_cv_scores,
            'feature_names': self.feature_names,
        }
        
        return self.results
    
    def _evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict:
        """Evaluate model performance."""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
        
        print(f"\n{model_name} Results:")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        print(f"  ROC-AUC:   {roc_auc:.3f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from Random Forest."""
        if self.rf_model is None:
            return None
        
        importance = self.rf_model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def get_decision_rules(self, max_depth: int = 3) -> str:
        """Extract decision rules from Decision Tree."""
        if self.dt_model is None:
            return ""
        
        return export_text(self.dt_model, feature_names=self.feature_names, max_depth=max_depth)
    
    def save(self, output_dir: Path = None) -> Path:
        """Save trained models."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / 'models'
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = output_dir / f'mpbpk_classifier_{timestamp}.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'dt_model': self.dt_model,
                'rf_model': self.rf_model,
                'feature_names': self.feature_names,
                'config': self.config,
                'results': self.results,
            }, f)
        
        print(f"Models saved to: {model_path}")
        return model_path
    
    @classmethod
    def load(cls, model_path: Path) -> 'mPBPKClassifier':
        """Load trained models."""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        classifier = cls(config=data['config'])
        classifier.dt_model = data['dt_model']
        classifier.rf_model = data['rf_model']
        classifier.feature_names = data['feature_names']
        classifier.results = data['results']
        
        return classifier


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  mPBPK ML Classifier Training")
    print("=" * 60)
    
    # Train classifier
    classifier = mPBPKClassifier()
    results = classifier.train()
    
    # Print feature importance
    print("\n" + "=" * 60)
    print("Feature Importance (Random Forest):")
    print("=" * 60)
    importance_df = classifier.get_feature_importance()
    print(importance_df.to_string(index=False))
    
    # Print decision rules
    print("\n" + "=" * 60)
    print("Decision Tree Rules (depth=3):")
    print("=" * 60)
    print(classifier.get_decision_rules(max_depth=3))
    
    # Save models
    model_path = classifier.save()
    
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
