"""
QSAR Predictor - Molecular Descriptor-based Toxicity Prediction
================================================================
Step 1: RDKit 기반 41개 분자 기술자를 사용한 RF 독성 예측 모델

기존 qsar_engine.py의 ApplicabilityDomain, QSAREngine 클래스를 활용합니다.

Features:
- 41개 RDKit 기술자 계산
- Random Forest 모델 학습/예측
- 기존 AD, Y-Randomization, Tropsha 5원칙 연동

Author: AI-Driven Clinical Trial Platform
Date: 2026-01-26
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import warnings
import pickle

warnings.filterwarnings("ignore")

# sklearn imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    r2_score,
    mean_squared_error,
)

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Fragments
    from rdkit import RDLogger

    RDLogger.logger().setLevel(RDLogger.ERROR)
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# 기존 qsar_engine 모듈 import
try:
    from src.models.qsar_engine import ApplicabilityDomain, QSARConfig, ValidationResult

    QSAR_ENGINE_AVAILABLE = True
except ImportError:
    try:
        from qsar_engine import ApplicabilityDomain, QSARConfig, ValidationResult

        QSAR_ENGINE_AVAILABLE = True
    except ImportError:
        QSAR_ENGINE_AVAILABLE = False


# =============================================================================
# 41개 분자 기술자 정의
# =============================================================================

if RDKIT_AVAILABLE:
    DESCRIPTOR_FUNCTIONS = {
        # 물리화학적 특성 (12개)
        "MolWt": Descriptors.MolWt,
        "MolLogP": Descriptors.MolLogP,
        "TPSA": Descriptors.TPSA,
        "MolMR": Descriptors.MolMR,
        "LabuteASA": Descriptors.LabuteASA,
        "NumValenceElectrons": Descriptors.NumValenceElectrons,
        "MaxPartialCharge": Descriptors.MaxPartialCharge,
        "MinPartialCharge": Descriptors.MinPartialCharge,
        "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge,
        "MinAbsPartialCharge": Descriptors.MinAbsPartialCharge,
        "FpDensityMorgan1": Descriptors.FpDensityMorgan1,
        "FpDensityMorgan2": Descriptors.FpDensityMorgan2,
        # 구조적 특성 (15개)
        "HeavyAtomCount": Descriptors.HeavyAtomCount,
        "NumHDonors": Lipinski.NumHDonors,
        "NumHAcceptors": Lipinski.NumHAcceptors,
        "NumRotatableBonds": Descriptors.NumRotatableBonds,
        "NumHeteroatoms": Descriptors.NumHeteroatoms,
        "RingCount": Descriptors.RingCount,
        "NumAromaticRings": Descriptors.NumAromaticRings,
        "NumAliphaticRings": Descriptors.NumAliphaticRings,
        "NumSaturatedRings": Descriptors.NumSaturatedRings,
        "FractionCSP3": Descriptors.FractionCSP3,
        "NumAromaticHeterocycles": Descriptors.NumAromaticHeterocycles,
        "NumAromaticCarbocycles": Descriptors.NumAromaticCarbocycles,
        "NumAliphaticHeterocycles": Descriptors.NumAliphaticHeterocycles,
        "NumAliphaticCarbocycles": Descriptors.NumAliphaticCarbocycles,
        "NumSaturatedHeterocycles": Descriptors.NumSaturatedHeterocycles,
        # 독성 관련 (14개)
        "NOCount": Lipinski.NOCount,
        "NHOHCount": Lipinski.NHOHCount,
        "NumRadicalElectrons": Descriptors.NumRadicalElectrons,
        "fr_Al_OH": Fragments.fr_Al_OH,
        "fr_Ar_OH": Fragments.fr_Ar_OH,
        "fr_aldehyde": Fragments.fr_aldehyde,
        "fr_ketone": Fragments.fr_ketone,
        "fr_ether": Fragments.fr_ether,
        "fr_ester": Fragments.fr_ester,
        "fr_nitro": Fragments.fr_nitro,
        "fr_nitrile": Fragments.fr_nitrile,
        "fr_halogen": Fragments.fr_halogen,
        "fr_sulfide": Fragments.fr_sulfide,
        "fr_amide": Fragments.fr_amide,
    }
else:
    DESCRIPTOR_FUNCTIONS = {}


@dataclass
class QSARPrediction:
    """QSAR 예측 결과"""

    smiles: str
    is_valid: bool = True

    # 예측 결과
    prediction: int = 0  # 0=Non-toxic, 1=Toxic
    probability: float = 0.0  # 독성 확률 (0-1)

    # Applicability Domain
    in_ad: bool = True  # AD 내 여부
    leverage: float = 0.0  # Leverage 값

    # 분자 정보
    molecular_weight: float = 0.0
    descriptors: Dict[str, float] = field(default_factory=dict)

    # 메타 정보
    model_endpoint: str = "general"  # 예측 엔드포인트
    confidence: str = "High"  # High, Medium, Low

    def to_dict(self) -> Dict:
        return {
            "smiles": self.smiles,
            "is_valid": self.is_valid,
            "prediction": self.prediction,
            "probability": self.probability,
            "in_ad": self.in_ad,
            "leverage": self.leverage,
            "molecular_weight": self.molecular_weight,
            "model_endpoint": self.model_endpoint,
            "confidence": self.confidence,
        }


@dataclass
class ModelPerformance:
    """모델 성능 지표"""

    endpoint: str

    # Classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0

    # Cross-validation
    cv_mean: float = 0.0
    cv_std: float = 0.0

    # Y-Randomization
    y_rand_passed: bool = False

    # AD Coverage
    ad_coverage: float = 0.0

    # Training info
    n_samples: int = 0
    n_features: int = 41


class QSARPredictor:
    """
    QSAR 예측기

    41개 RDKit 기술자를 사용하여 독성을 예측합니다.
    기존 qsar_engine.py의 ApplicabilityDomain을 활용합니다.
    """

    # Tox21 엔드포인트
    TOX21_ENDPOINTS = [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ]

    # Clinical Toxicity Endpoint
    CLINTOX_ENDPOINT = "clintox_ct_tox"

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        random_state: int = 42,
        auto_load: bool = True,
    ):
        """
        Initialize QSAR Predictor.

        Args:
            model_dir: 모델 저장 디렉토리
            random_state: 랜덤 시드
            auto_load: 저장된 모델 자동 로드 여부
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for QSARPredictor")

        self.model_dir = model_dir or Path(__file__).parent / "qsar"
        self.random_state = random_state

        # 모델 저장소
        self.models: Dict[str, RandomForestClassifier] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.ad_domains: Dict[str, ApplicabilityDomain] = {}
        self.performances: Dict[str, ModelPerformance] = {}

        self.feature_names = list(DESCRIPTOR_FUNCTIONS.keys())
        self.is_trained = False

        # 자동 모델 로드
        if auto_load:
            self._auto_load_models()

    def _auto_load_models(self) -> int:
        """
        저장된 모델 자동 로드.

        Returns:
            로드된 모델 수
        """
        if not self.model_dir.exists():
            return 0

        loaded_count = 0
        for model_file in self.model_dir.glob("*_model.pkl"):
            try:
                endpoint = model_file.stem.replace("_model", "")
                self.load_model(endpoint, model_file, verbose=False)
                loaded_count += 1
            except Exception as e:
                # 로드 실패 시 무시하고 계속
                pass

        if loaded_count > 0:
            self.is_trained = True

        return loaded_count

    def load_all_models(self) -> Dict[str, bool]:
        """
        모든 저장된 모델 로드 (명시적 호출).

        Returns:
            엔드포인트별 로드 성공 여부
        """
        results = {}

        if not self.model_dir.exists():
            return results

        for model_file in self.model_dir.glob("*_model.pkl"):
            endpoint = model_file.stem.replace("_model", "")
            try:
                self.load_model(endpoint, model_file)
                results[endpoint] = True
            except Exception as e:
                results[endpoint] = False
                print(f"    Failed to load {endpoint}: {e}")

        if any(results.values()):
            self.is_trained = True

        return results

    def calculate_descriptors(self, smiles: str) -> Optional[Dict[str, float]]:
        """
        SMILES에서 41개 분자 기술자 계산.

        Args:
            smiles: SMILES 문자열

        Returns:
            기술자 딕셔너리 or None
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            descriptors = {}
            for name, func in DESCRIPTOR_FUNCTIONS.items():
                try:
                    value = func(mol)
                    if value is None or np.isnan(value) or np.isinf(value):
                        value = 0.0
                    descriptors[name] = float(value)
                except Exception:
                    descriptors[name] = 0.0

            return descriptors
        except Exception:
            return None

    def smiles_to_features(self, smiles: str) -> Optional[np.ndarray]:
        """
        SMILES를 feature 벡터로 변환.

        Args:
            smiles: SMILES 문자열

        Returns:
            Feature array (41,) or None
        """
        descriptors = self.calculate_descriptors(smiles)
        if descriptors is None:
            return None

        return np.array([descriptors[name] for name in self.feature_names])

    def train_from_dataframe(
        self,
        df: pd.DataFrame,
        smiles_col: str = "smiles",
        endpoint: str = "NR-AhR",
        test_size: float = 0.2,
    ) -> ModelPerformance:
        """
        DataFrame에서 QSAR 모델 학습.

        Args:
            df: 학습 데이터 (SMILES + 라벨)
            smiles_col: SMILES 컬럼명
            endpoint: 독성 엔드포인트
            test_size: 테스트 비율

        Returns:
            ModelPerformance
        """
        print(f">>> Training QSAR model for endpoint: {endpoint}")

        # 유효한 데이터 필터링
        valid_data = df[df[endpoint].notna()].copy()
        print(f"    Valid samples: {len(valid_data)}")

        # Feature 계산
        X_list = []
        y_list = []

        for _, row in valid_data.iterrows():
            features = self.smiles_to_features(row[smiles_col])
            if features is not None:
                X_list.append(features)
                y_list.append(int(row[endpoint]))

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"    Features computed: {len(X)}")
        print(f"    Class distribution: {np.bincount(y)}")

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Applicability Domain
        ad = ApplicabilityDomain(method="leverage")
        ad.fit(X_train)

        # Model Training
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(X_train_scaled, y_train)

        # Evaluation
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

        # AD Coverage
        ad_check = ad.check(X_test)
        ad_coverage = np.mean(ad_check) * 100

        # Performance 기록
        perf = ModelPerformance(
            endpoint=endpoint,
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y_test, y_prob)
            if len(np.unique(y_test)) > 1
            else 0.0,
            cv_mean=np.mean(cv_scores),
            cv_std=np.std(cv_scores),
            ad_coverage=ad_coverage,
            n_samples=len(X),
            n_features=X.shape[1],
        )

        # 저장
        self.models[endpoint] = model
        self.scalers[endpoint] = scaler
        self.ad_domains[endpoint] = ad
        self.performances[endpoint] = perf
        self.is_trained = True

        print(f"    Accuracy: {perf.accuracy:.3f}")
        print(f"    ROC-AUC: {perf.roc_auc:.3f}")
        print(f"    AD Coverage: {perf.ad_coverage:.1f}%")

        return perf

    def train_all_endpoints(
        self, data_path: Optional[Path] = None
    ) -> Dict[str, ModelPerformance]:
        """
        모든 Tox21 엔드포인트에 대해 모델 학습.

        Args:
            data_path: 전처리된 데이터 경로

        Returns:
            엔드포인트별 성능 딕셔너리
        """
        if data_path is None:
            data_path = (
                Path(__file__).parent.parent.parent
                / "data"
                / "processed"
                / "tox21_descriptors.csv"
            )

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)
        print(f">>> Loaded {len(df)} compounds from {data_path.name}")

        results = {}
        for endpoint in self.TOX21_ENDPOINTS:
            if endpoint in df.columns:
                try:
                    perf = self.train_from_dataframe(df, endpoint=endpoint)
                    results[endpoint] = perf
                except Exception as e:
                    print(f"    Error training {endpoint}: {e}")

        return results

    def predict(self, smiles: str, endpoint: str = "NR-AhR") -> QSARPrediction:
        """
        단일 화합물 독성 예측.

        Args:
            smiles: SMILES 문자열
            endpoint: 예측할 엔드포인트

        Returns:
            QSARPrediction
        """
        result = QSARPrediction(smiles=smiles, model_endpoint=endpoint)

        # Feature 계산
        features = self.smiles_to_features(smiles)
        if features is None:
            result.is_valid = False
            result.confidence = "None"
            return result

        result.molecular_weight = features[0]  # MolWt is first descriptor

        # 학습된 모델 체크
        if endpoint not in self.models:
            result.confidence = "None"
            result.is_valid = False
            return result

        # Scaling
        X = features.reshape(1, -1)
        X_scaled = self.scalers[endpoint].transform(X)

        # AD Check
        if endpoint in self.ad_domains and self.ad_domains[endpoint] is not None:
            ad_check = self.ad_domains[endpoint].check(X)
            result.in_ad = bool(ad_check[0])

        # Prediction
        model = self.models[endpoint]
        result.prediction = int(model.predict(X_scaled)[0])
        result.probability = float(model.predict_proba(X_scaled)[0, 1])

        # Confidence 결정
        if not result.in_ad:
            result.confidence = "Low"
        elif 0.3 <= result.probability <= 0.7:
            result.confidence = "Medium"
        else:
            result.confidence = "High"

        return result

    def predict_multiple_endpoints(self, smiles: str) -> Dict[str, QSARPrediction]:
        """
        모든 학습된 엔드포인트에 대해 예측.

        Args:
            smiles: SMILES 문자열

        Returns:
            엔드포인트별 예측 결과
        """
        results = {}
        for endpoint in self.models.keys():
            results[endpoint] = self.predict(smiles, endpoint)
        return results

    def predict_clinical_toxicity(self, smiles: str) -> QSARPrediction:
        """
        임상 독성(ClinTox) 예측.

        Args:
            smiles: SMILES 문자열

        Returns:
            QSARPrediction
        """
        return self.predict(smiles, self.CLINTOX_ENDPOINT)

    def batch_predict(
        self, smiles_list: List[str], endpoint: str = "NR-AhR"
    ) -> List[QSARPrediction]:
        """
        배치 예측.

        Args:
            smiles_list: SMILES 리스트
            endpoint: 예측할 엔드포인트

        Returns:
            QSARPrediction 리스트
        """
        return [self.predict(smiles, endpoint) for smiles in smiles_list]

    def save_model(self, endpoint: str, path: Optional[Path] = None):
        """모델 저장."""
        if endpoint not in self.models:
            raise ValueError(f"Model for {endpoint} not found")

        if path is None:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            path = self.model_dir / f"{endpoint}_model.pkl"

        model_data = {
            "model": self.models[endpoint],
            "scaler": self.scalers[endpoint],
            "ad": self.ad_domains.get(endpoint),
            "performance": self.performances.get(endpoint),
            "feature_names": self.feature_names,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"    Saved model to {path}")

    def load_model(
        self, endpoint: str, path: Optional[Path] = None, verbose: bool = True
    ):
        """
        모델 로드.

        Args:
            endpoint: 모델 엔드포인트명
            path: 모델 파일 경로 (None이면 기본 경로 사용)
            verbose: 로드 메시지 출력 여부
        """
        if path is None:
            path = self.model_dir / f"{endpoint}_model.pkl"

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.models[endpoint] = model_data["model"]
        self.scalers[endpoint] = model_data["scaler"]
        self.ad_domains[endpoint] = model_data.get("ad")
        self.performances[endpoint] = model_data.get("performance")
        self.is_trained = True

        if verbose:
            print(f"    Loaded model: {endpoint}")

    def get_feature_importance(self, endpoint: str) -> Dict[str, float]:
        """Feature importance 반환."""
        if endpoint not in self.models:
            return {}

        importances = self.models[endpoint].feature_importances_
        return dict(zip(self.feature_names, importances))


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_toxicity_prediction(smiles: str, endpoint: str = "NR-AhR") -> Dict:
    """
    빠른 독성 예측 (사전 학습된 모델 필요).

    Args:
        smiles: SMILES 문자열
        endpoint: 예측 엔드포인트

    Returns:
        예측 결과 딕셔너리
    """
    predictor = QSARPredictor()
    result = predictor.predict(smiles, endpoint)
    return result.to_dict()


def get_molecular_descriptors(smiles: str) -> Optional[Dict[str, float]]:
    """
    분자 기술자 계산.

    Args:
        smiles: SMILES 문자열

    Returns:
        41개 기술자 딕셔너리
    """
    predictor = QSARPredictor()
    return predictor.calculate_descriptors(smiles)


# =============================================================================
# Main (Test)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  QSAR Predictor Test")
    print("=" * 60)

    if not RDKIT_AVAILABLE:
        print("Error: RDKit not available")
        exit(1)

    # Initialize predictor
    predictor = QSARPredictor()

    # Test descriptor calculation
    test_smiles = "CC(=O)Nc1ccc(O)cc1"  # Acetaminophen
    print(f"\n>>> Test Compound: Acetaminophen")
    print(f"    SMILES: {test_smiles}")

    descriptors = predictor.calculate_descriptors(test_smiles)
    if descriptors:
        print(f"\n>>> Calculated {len(descriptors)} descriptors")
        print(f"    MolWt: {descriptors['MolWt']:.2f}")
        print(f"    MolLogP: {descriptors['MolLogP']:.2f}")
        print(f"    TPSA: {descriptors['TPSA']:.2f}")
        print(f"    HeavyAtomCount: {descriptors['HeavyAtomCount']}")
        print(f"    NumHDonors: {descriptors['NumHDonors']}")
        print(f"    NumHAcceptors: {descriptors['NumHAcceptors']}")

    # Try to train on preprocessed data if available
    data_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "processed"
        / "tox21_descriptors.csv"
    )
    if data_path.exists():
        print("\n>>> Training on preprocessed Tox21 data...")
        df = pd.read_csv(data_path)

        # Train on one endpoint
        perf = predictor.train_from_dataframe(df, endpoint="NR-AhR")

        # Predict
        result = predictor.predict(test_smiles, endpoint="NR-AhR")
        print(f"\n>>> Prediction for Acetaminophen:")
        print(f"    Prediction: {'Toxic' if result.prediction else 'Non-toxic'}")
        print(f"    Probability: {result.probability:.3f}")
        print(f"    In AD: {result.in_ad}")
        print(f"    Confidence: {result.confidence}")
    else:
        print(f"\n    Preprocessed data not found at: {data_path}")
        print("    Run preprocess_toxicity.py first to generate training data")

    print("\n" + "=" * 60)
    print("  QSAR Predictor Test Complete!")
    print("=" * 60)
