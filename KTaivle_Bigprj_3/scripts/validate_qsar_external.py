"""
External Validation Script for QSAR Models
============================================

ClinTox 데이터셋을 사용하여 학습된 QSAR 모델의 외부 검증 수행.

외부 검증 목표:
- FDA_APPROVED 예측: Accuracy >= 80%
- CT_TOX 예측: Accuracy >= 75%

Author: AI-Driven Clinical Trial Platform
Date: 2026-01-26
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.qsar_predictor import QSARPredictor


def load_clintox_data(data_path: Path = None) -> pd.DataFrame:
    """ClinTox 데이터 로드."""
    if data_path is None:
        data_path = (
            Path(__file__).parent.parent
            / "data"
            / "processed"
            / "clintox_descriptors.csv"
        )

    if not data_path.exists():
        raise FileNotFoundError(f"ClinTox data not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f">>> Loaded {len(df)} compounds from {data_path.name}")
    print(
        f"    FDA_APPROVED: {df['FDA_APPROVED'].sum()} approved, {(1 - df['FDA_APPROVED']).sum()} not"
    )
    print(
        f"    CT_TOX: {df['CT_TOX'].sum()} toxic, {(1 - df['CT_TOX']).sum()} non-toxic"
    )

    return df


def predict_toxicity_ensemble(predictor: QSARPredictor, smiles: str) -> dict:
    """
    모든 학습된 엔드포인트에 대해 예측하고 앙상블 결과 반환.

    Returns:
        {
            'toxic_count': 독성 예측 엔드포인트 수,
            'total_endpoints': 전체 엔드포인트 수,
            'toxic_ratio': 독성 비율,
            'avg_probability': 평균 독성 확률,
            'max_probability': 최대 독성 확률,
            'predictions': 엔드포인트별 예측
        }
    """
    predictions = predictor.predict_multiple_endpoints(smiles)

    if not predictions:
        return {
            "toxic_count": 0,
            "total_endpoints": 0,
            "toxic_ratio": 0.0,
            "avg_probability": 0.0,
            "max_probability": 0.0,
            "is_valid": False,
        }

    toxic_count = sum(1 for p in predictions.values() if p.prediction == 1)
    total = len(predictions)
    probabilities = [p.probability for p in predictions.values()]

    return {
        "toxic_count": toxic_count,
        "total_endpoints": total,
        "toxic_ratio": toxic_count / total if total > 0 else 0.0,
        "avg_probability": np.mean(probabilities),
        "max_probability": max(probabilities),
        "is_valid": True,
        "predictions": predictions,
    }


def validate_on_clintox(
    predictor: QSARPredictor,
    df: pd.DataFrame,
    toxicity_threshold: float = 0.25,
) -> dict:
    """
    ClinTox 데이터셋에서 외부 검증 수행.

    Args:
        predictor: 학습된 QSAR 예측기
        df: ClinTox 데이터
        toxicity_threshold: 독성 분류 임계값 (toxic_ratio)

    Returns:
        검증 결과 딕셔너리
    """
    print(f"\n>>> Running external validation on ClinTox...")
    print(f"    Toxicity threshold: {toxicity_threshold:.0%}")

    results = []
    valid_count = 0
    skip_count = 0
    total = len(df)

    for idx, row in df.iterrows():
        smiles = str(row["smiles"])
        fda_approved = int(row["FDA_APPROVED"])
        ct_tox = int(row["CT_TOX"])

        try:
            ensemble = predict_toxicity_ensemble(predictor, smiles)

            if not ensemble.get("is_valid", False):
                skip_count += 1
                continue

            # 앙상블 기반 독성 예측
            # toxic_ratio가 threshold 이상이면 독성으로 분류
            pred_toxic = 1 if ensemble["toxic_ratio"] >= toxicity_threshold else 0

            results.append({
                "smiles": smiles,
                "fda_approved": fda_approved,
                "ct_tox": ct_tox,
                "toxic_count": ensemble["toxic_count"],
                "total_endpoints": ensemble["total_endpoints"],
                "toxic_ratio": ensemble["toxic_ratio"],
                "avg_probability": ensemble["avg_probability"],
                "max_probability": ensemble["max_probability"],
                "pred_toxic": pred_toxic,
            })
            valid_count += 1

        except Exception as e:
            skip_count += 1
            continue

        if valid_count % 50 == 0:
            print(f"    Processed {valid_count} valid compounds...")

    print(f"    Valid: {valid_count}, Skipped: {skip_count}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        return {
            "n_compounds": 0,
            "n_skipped": skip_count,
            "toxicity_threshold": toxicity_threshold,
            "ct_tox_metrics": {},
            "fda_approved_metrics": {},
            "results_df": results_df,
        }

    # Calculate metrics
    # 1. CT_TOX prediction (독성 예측 -> CT_TOX)
    y_true_tox = results_df["ct_tox"].values
    y_pred_tox = results_df["pred_toxic"].values

    tox_metrics = {
        "accuracy": accuracy_score(y_true_tox, y_pred_tox),
        "precision": precision_score(y_true_tox, y_pred_tox, zero_division=0),
        "recall": recall_score(y_true_tox, y_pred_tox, zero_division=0),
        "f1": f1_score(y_true_tox, y_pred_tox, zero_division=0),
    }

    if len(np.unique(y_true_tox)) > 1:
        tox_metrics["roc_auc"] = roc_auc_score(
            y_true_tox, results_df["toxic_ratio"].values
        )

    # 2. FDA_APPROVED prediction (비독성 -> FDA 승인)
    # 독성이 낮으면 (pred_toxic=0) FDA 승인 가능성 높음
    y_true_fda = results_df["fda_approved"].values
    y_pred_fda = 1 - results_df["pred_toxic"].values  # 비독성 -> 승인

    fda_metrics = {
        "accuracy": accuracy_score(y_true_fda, y_pred_fda),
        "precision": precision_score(y_true_fda, y_pred_fda, zero_division=0),
        "recall": recall_score(y_true_fda, y_pred_fda, zero_division=0),
        "f1": f1_score(y_true_fda, y_pred_fda, zero_division=0),
    }

    return {
        "n_compounds": valid_count,
        "n_skipped": skip_count,
        "toxicity_threshold": toxicity_threshold,
        "ct_tox_metrics": tox_metrics,
        "fda_approved_metrics": fda_metrics,
        "results_df": results_df,
    }
            )
            valid_count += 1

        except Exception as e:
            skip_count += 1
            continue

        if (idx + 1) % 200 == 0:
            print(f"    Processed {idx + 1}/{len(df)} compounds...")

    print(f"    Valid: {valid_count}, Skipped: {skip_count}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate metrics
    # 1. CT_TOX prediction (독성 예측 -> CT_TOX)
    y_true_tox = results_df["ct_tox"].values
    y_pred_tox = results_df["pred_toxic"].values

    tox_metrics = {
        "accuracy": accuracy_score(y_true_tox, y_pred_tox),
        "precision": precision_score(y_true_tox, y_pred_tox, zero_division=0),
        "recall": recall_score(y_true_tox, y_pred_tox, zero_division=0),
        "f1": f1_score(y_true_tox, y_pred_tox, zero_division=0),
    }

    if len(np.unique(y_true_tox)) > 1:
        tox_metrics["roc_auc"] = roc_auc_score(
            y_true_tox, results_df["toxic_ratio"].values
        )

    # 2. FDA_APPROVED prediction (비독성 -> FDA 승인)
    # 독성이 낮으면 (pred_toxic=0) FDA 승인 가능성 높음
    y_true_fda = results_df["fda_approved"].values
    y_pred_fda = 1 - results_df["pred_toxic"].values  # 비독성 -> 승인

    fda_metrics = {
        "accuracy": accuracy_score(y_true_fda, y_pred_fda),
        "precision": precision_score(y_true_fda, y_pred_fda, zero_division=0),
        "recall": recall_score(y_true_fda, y_pred_fda, zero_division=0),
        "f1": f1_score(y_true_fda, y_pred_fda, zero_division=0),
    }

    return {
        "n_compounds": valid_count,
        "n_skipped": skip_count,
        "toxicity_threshold": toxicity_threshold,
        "ct_tox_metrics": tox_metrics,
        "fda_approved_metrics": fda_metrics,
        "results_df": results_df,
    }


def print_validation_report(validation_result: dict):
    """검증 결과 보고서 출력."""
    print("\n" + "=" * 60)
    print("  EXTERNAL VALIDATION REPORT")
    print("=" * 60)

    print(f"\n  Compounds evaluated: {validation_result['n_compounds']}")
    print(f"  Compounds skipped: {validation_result['n_skipped']}")
    print(f"  Toxicity threshold: {validation_result['toxicity_threshold']:.0%}")

    # CT_TOX metrics
    tox = validation_result["ct_tox_metrics"]
    print("\n  --- CT_TOX Prediction (Clinical Trial Toxicity) ---")
    print(f"  Accuracy:  {tox['accuracy']:.3f}")
    print(f"  Precision: {tox['precision']:.3f}")
    print(f"  Recall:    {tox['recall']:.3f}")
    print(f"  F1 Score:  {tox['f1']:.3f}")
    if "roc_auc" in tox:
        print(f"  ROC-AUC:   {tox['roc_auc']:.3f}")

    target_acc = 0.75
    status = "[PASS]" if tox["accuracy"] >= target_acc else "[FAIL]"
    print(f"\n  Target: Accuracy >= {target_acc:.0%}  {status}")

    # FDA_APPROVED metrics
    fda = validation_result["fda_approved_metrics"]
    print("\n  --- FDA_APPROVED Prediction ---")
    print(f"  Accuracy:  {fda['accuracy']:.3f}")
    print(f"  Precision: {fda['precision']:.3f}")
    print(f"  Recall:    {fda['recall']:.3f}")
    print(f"  F1 Score:  {fda['f1']:.3f}")

    target_acc = 0.80
    status = "[PASS]" if fda["accuracy"] >= target_acc else "[FAIL]"
    print(f"\n  Target: Accuracy >= {target_acc:.0%}  {status}")

    print("\n" + "=" * 60)


def find_optimal_threshold(predictor: QSARPredictor, df: pd.DataFrame) -> float:
    """
    다양한 threshold에서 성능을 측정하여 최적 임계값 찾기.
    """
    print("\n>>> Finding optimal toxicity threshold...")

    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    best_threshold = 0.25
    best_score = 0

    # 먼저 모든 예측 수행 (한 번만)
    predictions_cache = {}
    for idx, row in df.iterrows():
        smiles = row["smiles"]
        try:
            ensemble = predict_toxicity_ensemble(predictor, smiles)
            if ensemble.get("is_valid", False):
                predictions_cache[smiles] = ensemble
        except:
            continue

        if (idx + 1) % 300 == 0:
            print(f"    Cached {idx + 1}/{len(df)} predictions...")

    print(f"    Total cached: {len(predictions_cache)}")

    # 각 threshold에서 성능 측정
    print("\n    Threshold | CT_TOX Acc | FDA Acc | Combined")
    print("    " + "-" * 50)

    for threshold in thresholds:
        results = []
        for smiles, ensemble in predictions_cache.items():
            row = df[df["smiles"] == smiles].iloc[0]
            pred_toxic = 1 if ensemble["toxic_ratio"] >= threshold else 0
            results.append(
                {
                    "ct_tox": row["CT_TOX"],
                    "fda_approved": row["FDA_APPROVED"],
                    "pred_toxic": pred_toxic,
                }
            )

        results_df = pd.DataFrame(results)

        ct_tox_acc = accuracy_score(results_df["ct_tox"], results_df["pred_toxic"])
        fda_acc = accuracy_score(
            results_df["fda_approved"], 1 - results_df["pred_toxic"]
        )
        combined = (ct_tox_acc + fda_acc) / 2

        print(
            f"    {threshold:9.2f} | {ct_tox_acc:10.3f} | {fda_acc:7.3f} | {combined:.3f}"
        )

        if combined > best_score:
            best_score = combined
            best_threshold = threshold

    print(f"\n    Best threshold: {best_threshold} (combined: {best_score:.3f})")
    return best_threshold


def main():
    """메인 실행 함수."""
    print("=" * 60)
    print("  QSAR External Validation on ClinTox Dataset")
    print("=" * 60)

    # 1. 데이터 로드
    df = load_clintox_data()

    # Use sample for faster validation (optional)
    sample_size = min(500, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    print(f"\n>>> Using sample of {sample_size} compounds for validation")

    # 2. QSAR 예측기 초기화 (자동 모델 로드)
    print("\n>>> Initializing QSAR Predictor...")
    predictor = QSARPredictor()

    if not predictor.is_trained:
        print("    ERROR: No trained models found!")
        print("    Please run train_qsar_models.py first.")
        return

    print(f"    Loaded {len(predictor.models)} endpoint models:")
    for endpoint in predictor.models.keys():
        print(f"      - {endpoint}")

    # 3. 최적 threshold 찾기 (옵션)
    # optimal_threshold = find_optimal_threshold(predictor, df.sample(n=min(300, len(df))))

    # 4. 외부 검증 수행
    validation_result = validate_on_clintox(
        predictor, df_sample, toxicity_threshold=0.25
    )

    # 5. 결과 출력
    print_validation_report(validation_result)

    # 6. 결과 저장
    output_path = (
        Path(__file__).parent.parent
        / "data"
        / "processed"
        / "clintox_validation_results.csv"
    )
    validation_result["results_df"].to_csv(output_path, index=False)
    print(f"\n>>> Results saved to: {output_path.name}")

    print("\n" + "=" * 60)
    print("  External Validation Complete!")
    print("=" * 60)
    print("  QSAR External Validation on ClinTox Dataset")
    print("=" * 60)

    # 1. 데이터 로드
    df = load_clintox_data()

    # 2. QSAR 예측기 초기화 (자동 모델 로드)
    print("\n>>> Initializing QSAR Predictor...")
    predictor = QSARPredictor()

    if not predictor.is_trained:
        print("    ERROR: No trained models found!")
        print("    Please run train_qsar_models.py first.")
        return

    print(f"    Loaded {len(predictor.models)} endpoint models:")
    for endpoint in predictor.models.keys():
        print(f"      - {endpoint}")

    # 3. 최적 threshold 찾기 (옵션)
    # optimal_threshold = find_optimal_threshold(predictor, df.sample(n=min(300, len(df))))

    # 4. 외부 검증 수행
    validation_result = validate_on_clintox(predictor, df, toxicity_threshold=0.25)

    # 5. 결과 출력
    print_validation_report(validation_result)

    # 6. 결과 저장
    output_path = (
        Path(__file__).parent.parent
        / "data"
        / "processed"
        / "clintox_validation_results.csv"
    )
    validation_result["results_df"].to_csv(output_path, index=False)
    print(f"\n>>> Results saved to: {output_path.name}")

    print("\n" + "=" * 60)
    print("  External Validation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
