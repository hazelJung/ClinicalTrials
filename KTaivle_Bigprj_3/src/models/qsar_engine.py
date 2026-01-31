"""
QSAR Engine - Quantitative Structure-Activity Relationship Model
=================================================================
Based on:
- Tropsha (2010): Best Practices for QSAR Model Development, Validation
- pharmaceuticals-18-00096: ML Implementation Guide

Implements Tropsha's 5 Validation Principles:
1. Applicability Domain
2. External Validation
3. Y-Randomization Test
4. Multiple Metrics
5. Interpretability (SHAP)

Author: AI-Driven Clinical Trial Platform
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
)
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QSARConfig:
    """QSAR 모델 설정"""
    task_type: str = 'classification'  # 'classification' or 'regression'
    model_type: str = 'random_forest'  # 'random_forest', 'xgboost'
    n_estimators: int = 100
    max_depth: int = 10
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    y_randomization_iter: int = 10


@dataclass
class ValidationResult:
    """검증 결과"""
    # Classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    
    # Regression metrics
    r2: float = 0.0
    q2: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    
    # Cross-validation
    cv_mean: float = 0.0
    cv_std: float = 0.0
    
    # Y-randomization
    y_random_mean: float = 0.0
    y_random_std: float = 0.0
    y_random_passed: bool = False
    
    # Applicability Domain
    ad_coverage: float = 0.0


# =============================================================================
# Applicability Domain (Tropsha Principle 1)
# =============================================================================

class ApplicabilityDomain:
    """
    Applicability Domain - 모델 적용 가능 범위 정의
    
    Tropsha 원칙: 모델은 학습 데이터 범위 내에서만 신뢰할 수 있음
    """
    
    def __init__(self, method: str = 'leverage'):
        """
        Args:
            method: 'leverage' (Williams plot), 'distance' (Euclidean), 'range'
        """
        self.method = method
        self.X_train = None
        self.threshold = None
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray):
        """학습 데이터로 AD 경계 설정"""
        self.X_train = self.scaler.fit_transform(X)
        n, p = self.X_train.shape
        
        if self.method == 'leverage':
            # Williams plot threshold: h* = 3(p+1)/n
            self.threshold = 3 * (p + 1) / n
        elif self.method == 'distance':
            # 평균 거리 + 3σ
            centroid = np.mean(self.X_train, axis=0)
            distances = np.linalg.norm(self.X_train - centroid, axis=1)
            self.threshold = np.mean(distances) + 3 * np.std(distances)
        elif self.method == 'range':
            # Min-Max 범위
            self.min_vals = np.min(self.X_train, axis=0)
            self.max_vals = np.max(self.X_train, axis=0)
    
    def check(self, X: np.ndarray) -> np.ndarray:
        """
        새 데이터가 AD 내에 있는지 확인
        
        Returns:
            bool array: True = AD 내, False = AD 외
        """
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'leverage':
            # Leverage 계산: h = x(X'X)^-1 x'
            XtX_inv = np.linalg.pinv(self.X_train.T @ self.X_train)
            leverages = np.array([x @ XtX_inv @ x.T for x in X_scaled])
            return leverages < self.threshold
        
        elif self.method == 'distance':
            centroid = np.mean(self.X_train, axis=0)
            distances = np.linalg.norm(X_scaled - centroid, axis=1)
            return distances < self.threshold
        
        elif self.method == 'range':
            in_range = np.all((X_scaled >= self.min_vals - 0.1) & 
                             (X_scaled <= self.max_vals + 0.1), axis=1)
            return in_range
        
        return np.ones(len(X), dtype=bool)


# =============================================================================
# QSAR Engine
# =============================================================================

class QSAREngine:
    """
    QSAR 모델 엔진 - Tropsha 5원칙 기반 구현
    
    Features:
    - Random Forest / XGBoost 지원
    - Applicability Domain 체크
    - Y-Randomization 테스트
    - 다중 검증 지표
    - SHAP 기반 해석
    """
    
    def __init__(self, config: QSARConfig = None):
        """
        Initialize QSAR Engine
        
        Args:
            config: QSAR 설정
        """
        self.config = config if config else QSARConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.ad = ApplicabilityDomain(method='leverage')
        self.feature_names = None
        self.is_fitted = False
        
        self._init_model()
    
    def _init_model(self):
        """모델 초기화"""
        if self.config.model_type == 'random_forest':
            if self.config.task_type == 'classification':
                self.model = RandomForestClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    random_state=self.config.random_state,
                    n_jobs=-1
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    random_state=self.config.random_state,
                    n_jobs=-1
                )
        # XGBoost는 설치 시 추가 가능
    
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray,
              feature_names: List[str] = None) -> ValidationResult:
        """
        모델 학습 및 검증 (Tropsha 원칙 적용)
        
        Args:
            X: Feature matrix
            y: Target labels/values
            feature_names: Feature 이름 리스트
            
        Returns:
            ValidationResult: 검증 결과
        """
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Train/Test Split (Principle 2: External Validation)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if self.config.task_type == 'classification' else None
        )
        
        # Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Applicability Domain (Principle 1)
        self.ad.fit(X_train)
        
        # Model Training
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Validation
        result = self._validate(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Y-Randomization Test (Principle 3)
        y_rand_scores = self._y_randomization_test(X_train_scaled, y_train)
        result.y_random_mean = np.mean(y_rand_scores)
        result.y_random_std = np.std(y_rand_scores)
        
        # Y-randomization 통과 기준: 실제 성능 > 랜덤 성능 + 3σ
        if self.config.task_type == 'classification':
            result.y_random_passed = result.accuracy > (result.y_random_mean + 3 * result.y_random_std)
        else:
            result.y_random_passed = result.r2 > (result.y_random_mean + 3 * result.y_random_std)
        
        # AD Coverage
        ad_check = self.ad.check(X_test)
        result.ad_coverage = np.mean(ad_check) * 100
        
        return result
    
    def _validate(self, X_train, y_train, X_test, y_test) -> ValidationResult:
        """내부 검증 (Principle 4: Multiple Metrics)"""
        result = ValidationResult()
        
        y_pred = self.model.predict(X_test)
        
        if self.config.task_type == 'classification':
            result.accuracy = accuracy_score(y_test, y_pred)
            result.precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            result.recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            result.f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(X_test)
                if y_prob.shape[1] == 2:
                    result.roc_auc = roc_auc_score(y_test, y_prob[:, 1])
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.model, X_train, y_train, 
                cv=self.config.cv_folds, scoring='accuracy'
            )
            result.cv_mean = np.mean(cv_scores)
            result.cv_std = np.std(cv_scores)
            
        else:  # Regression
            result.r2 = r2_score(y_test, y_pred)
            result.rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            result.mae = mean_absolute_error(y_test, y_pred)
            
            # Q² (Leave-one-out cross-validation approximation)
            cv_scores = cross_val_score(
                self.model, X_train, y_train,
                cv=self.config.cv_folds, scoring='r2'
            )
            result.q2 = np.mean(cv_scores)
            result.cv_mean = result.q2
            result.cv_std = np.std(cv_scores)
        
        return result
    
    def _y_randomization_test(self, X, y) -> List[float]:
        """
        Y-Randomization Test (Principle 3)
        
        Label을 무작위로 섞어서 학습했을 때 성능이 크게 떨어지는지 확인
        → 우연한 상관관계가 아님을 검증
        """
        scores = []
        
        for i in range(self.config.y_randomization_iter):
            y_shuffled = np.random.permutation(y)
            
            if self.config.task_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=50,  # 빠른 테스트
                    max_depth=self.config.max_depth,
                    random_state=i
                )
                model.fit(X, y_shuffled)
                y_pred = model.predict(X)
                score = accuracy_score(y_shuffled, y_pred)
            else:
                model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=self.config.max_depth,
                    random_state=i
                )
                model.fit(X, y_shuffled)
                y_pred = model.predict(X)
                score = r2_score(y_shuffled, y_pred)
            
            scores.append(score)
        
        return scores
    
    def predict(self, X: np.ndarray, check_ad: bool = True) -> Dict:
        """
        예측 수행
        
        Args:
            X: Feature matrix
            check_ad: Applicability Domain 체크 여부
            
        Returns:
            dict: predictions, probabilities, ad_check
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first!")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        result = {
            'predictions': predictions,
            'ad_check': None,
            'probabilities': None
        }
        
        # AD Check
        if check_ad:
            result['ad_check'] = self.ad.check(X)
        
        # Probabilities (for classification)
        if self.config.task_type == 'classification' and hasattr(self.model, 'predict_proba'):
            result['probabilities'] = self.model.predict_proba(X_scaled)
        
        return result
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Feature Importance 반환"""
        if not self.is_fitted:
            return {}
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))
    
    def explain_prediction(self, X: np.ndarray, idx: int = 0):
        """
        SHAP 기반 예측 설명 (Principle 5: Interpretability)
        """
        if not SHAP_AVAILABLE:
            return {"error": "SHAP not installed. Run: pip install shap"}
        
        if not self.is_fitted:
            return {"error": "Model must be trained first!"}
        
        X_scaled = self.scaler.transform(X)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Binary classification
        
        explanations = dict(zip(
            self.feature_names, 
            shap_values[idx]
        ))
        
        return explanations


# =============================================================================
# Convenience Functions
# =============================================================================

def create_toxicity_model(X: np.ndarray, 
                          y: np.ndarray,
                          feature_names: List[str] = None) -> Tuple[QSAREngine, ValidationResult]:
    """
    독성 예측 QSAR 모델 생성 (편의 함수)
    
    Args:
        X: Molecular descriptors matrix
        y: Toxicity labels (0=non-toxic, 1=toxic)
        feature_names: Descriptor names
        
    Returns:
        Trained QSAREngine, ValidationResult
    """
    config = QSARConfig(
        task_type='classification',
        model_type='random_forest',
        n_estimators=100,
        max_depth=10
    )
    
    engine = QSAREngine(config)
    result = engine.train(X, y, feature_names)
    
    return engine, result


def generate_sample_data(n_samples: int = 1000, 
                         n_features: int = 10,
                         toxic_ratio: float = 0.3) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    샘플 데이터 생성 (테스트용)
    """
    np.random.seed(42)
    
    # Features (molecular descriptors simulation)
    X = np.random.randn(n_samples, n_features)
    
    # Target (toxicity based on some features)
    # Complex rule: toxic if (Feature_0 > 0.5 AND Feature_1 < -0.3) OR Feature_2 > 1.5
    y = ((X[:, 0] > 0.5) & (X[:, 1] < -0.3)) | (X[:, 2] > 1.5)
    y = y.astype(int)
    
    feature_names = [f'Descriptor_{i}' for i in range(n_features)]
    
    return X, y, feature_names


# =============================================================================
# Main (Test)
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("  QSAR Engine Test (Tropsha 5 Principles)")
    print("="*60)
    
    # Generate sample data
    print("\n>>> Generating sample data...")
    X, y, feature_names = generate_sample_data(n_samples=1000, n_features=10)
    print(f"    Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"    Toxic ratio: {np.mean(y)*100:.1f}%")
    
    # Train QSAR model
    print("\n>>> Training QSAR model...")
    engine, result = create_toxicity_model(X, y, feature_names)
    
    # Print results
    print("\n" + "="*60)
    print("  Validation Results (Tropsha Principles)")
    print("="*60)
    
    print(f"\n[Principle 2] External Validation:")
    print(f"  Accuracy:  {result.accuracy:.3f}")
    print(f"  Precision: {result.precision:.3f}")
    print(f"  Recall:    {result.recall:.3f}")
    print(f"  F1-Score:  {result.f1:.3f}")
    print(f"  ROC-AUC:   {result.roc_auc:.3f}")
    
    print(f"\n[Principle 3] Y-Randomization Test:")
    print(f"  Random Mean: {result.y_random_mean:.3f} ± {result.y_random_std:.3f}")
    print(f"  Test Passed: {'✅ Yes' if result.y_random_passed else '❌ No'}")
    
    print(f"\n[Principle 4] Cross-Validation:")
    print(f"  CV Score: {result.cv_mean:.3f} ± {result.cv_std:.3f}")
    
    print(f"\n[Principle 1] Applicability Domain:")
    print(f"  AD Coverage: {result.ad_coverage:.1f}%")
    
    print(f"\n[Principle 5] Feature Importance:")
    importances = engine.get_feature_importance()
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, imp in sorted_imp:
        print(f"  {name}: {imp:.3f}")
    
    print("\n" + "="*60)
    print("  QSAR Engine Test Complete!")
    print("="*60)
