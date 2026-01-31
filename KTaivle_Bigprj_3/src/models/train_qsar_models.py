"""
Train QSAR Models Script
========================
Tox21 데이터셋의 모든 엔드포인트(12개)에 대해 Random Forest 모델을 학습하고 저장합니다.

Steps:
1. 전처리된 데이터 로드 (tox21_descriptors.csv)
2. QSARPredictor 초기화
3. 12개 엔드포인트 일괄 학습
4. 모델 저장 (src/models/qsar/)
5. 성능 리포트 생성

Author: AI-Driven Clinical Trial Platform
Date: 2026-01-26
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.qsar_predictor import QSARPredictor


def main():
    print("=" * 60)
    print("  Starting QSAR Model Training Pipeline")
    print("=" * 60)

    # 1. Initialize Predictor
    # -----------------------
    # 모델 저장 경로 설정
    model_dir = Path("src/models/qsar")
    model_dir.mkdir(parents=True, exist_ok=True)

    predictor = QSARPredictor(model_dir=model_dir, auto_load=False)

    # 2. Check Data
    # -------------
    data_path = Path("data/processed/tox21_descriptors.csv")
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please run src/data/preprocess_toxicity.py first.")
        return

    print(f"Using training data: {data_path}")

    # 3. Train All Endpoints
    # ----------------------
    print("\nTraining models for all 12 Tox21 endpoints...")
    print("This may take a few minutes depending on your hardware.\n")

    results = predictor.train_all_endpoints(data_path=data_path)

    # 4. Summarize Results
    # --------------------
    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)
    print(
        f"{'Endpoint':<15} | {'Accuracy':<10} | {'ROC-AUC':<10} | {'AD Coverage':<12}"
    )
    print("-" * 55)

    overall_auc = []

    for endpoint, perf in results.items():
        print(
            f"{endpoint:<15} | {perf.accuracy:.3f}      | {perf.roc_auc:.3f}      | {perf.ad_coverage:.1f}%"
        )
        overall_auc.append(perf.roc_auc)

        # Save explicitly (already done inside train_all_endpoints but good to be sure)
        predictor.save_model(endpoint)

    if overall_auc:
        print("-" * 55)
        print(f"Average ROC-AUC: {sum(overall_auc) / len(overall_auc):.3f}")

    print(f"\nModels saved to: {model_dir.absolute()}")
    print("=" * 60)
    print("  Done!")


if __name__ == "__main__":
    main()
