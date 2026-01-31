"""
QSAR Model Training Script
==========================
Tox21 데이터로 12개 엔드포인트 QSAR 모델 학습

실행:
    python scripts/train_qsar_models.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.models.qsar_predictor import QSARPredictor


def main():
    print("=" * 60)
    print("  QSAR Model Training (Tox21 Dataset)")
    print("=" * 60)

    # Load preprocessed data
    data_path = project_root / "data" / "processed" / "tox21_descriptors.csv"

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Run 'python src/data/preprocess_toxicity.py' first")
        return

    df = pd.read_csv(data_path)
    print(f"\n>>> Loaded {len(df)} compounds with {len(df.columns)} columns")

    # Initialize predictor
    predictor = QSARPredictor()

    # Define endpoints to train
    endpoints = [
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

    # Train models
    results = {}
    print("\n>>> Training QSAR models for each endpoint...")
    print("-" * 60)

    for endpoint in endpoints:
        if endpoint not in df.columns:
            print(f"    {endpoint}: Column not found, skipping")
            continue

        # Check if enough data
        valid_count = df[endpoint].notna().sum()
        if valid_count < 100:
            print(
                f"    {endpoint}: Insufficient data ({valid_count} samples), skipping"
            )
            continue

        try:
            perf = predictor.train_from_dataframe(df, endpoint=endpoint)
            results[endpoint] = perf

            print(f"    {endpoint}:")
            print(f"      Accuracy: {perf.accuracy:.3f}")
            print(f"      ROC-AUC:  {perf.roc_auc:.3f}")
            print(f"      F1:       {perf.f1:.3f}")
            print(f"      AD:       {perf.ad_coverage:.1f}%")
            print()

        except Exception as e:
            print(f"    {endpoint}: Training failed - {e}")

    # Summary
    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)

    if results:
        # Calculate averages
        avg_acc = np.mean([r.accuracy for r in results.values()])
        avg_auc = np.mean([r.roc_auc for r in results.values()])
        avg_f1 = np.mean([r.f1 for r in results.values()])
        avg_ad = np.mean([r.ad_coverage for r in results.values()])

        print(f"\n  Models trained: {len(results)}/{len(endpoints)}")
        print(f"\n  Average Metrics:")
        print(f"    Accuracy:    {avg_acc:.3f}")
        print(f"    ROC-AUC:     {avg_auc:.3f}")
        print(f"    F1-Score:    {avg_f1:.3f}")
        print(f"    AD Coverage: {avg_ad:.1f}%")

        # Best performing endpoints
        print(f"\n  Best ROC-AUC:")
        sorted_by_auc = sorted(
            results.items(), key=lambda x: x[1].roc_auc, reverse=True
        )
        for endpoint, perf in sorted_by_auc[:3]:
            print(f"    {endpoint}: {perf.roc_auc:.3f}")

        # Save models
        print("\n>>> Saving models...")
        model_dir = project_root / "models" / "qsar"
        model_dir.mkdir(parents=True, exist_ok=True)

        for endpoint in results.keys():
            try:
                predictor.save_model(endpoint, model_dir / f"{endpoint}_model.pkl")
            except Exception as e:
                print(f"    Warning: Failed to save {endpoint}: {e}")

        print(f"    Models saved to: {model_dir}")
    else:
        print("\n  No models were trained successfully.")

    # Test prediction
    print("\n" + "=" * 60)
    print("  Testing Prediction")
    print("=" * 60)

    test_smiles = [
        ("Acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ]

    for name, smiles in test_smiles:
        print(f"\n  >>> {name}: {smiles}")

        for endpoint in list(results.keys())[:3]:  # Test on first 3 endpoints
            pred = predictor.predict(smiles, endpoint)
            if pred.is_valid:
                label = "Toxic" if pred.prediction == 1 else "Non-toxic"
                print(
                    f"      {endpoint}: {label} (prob={pred.probability:.2f}, AD={pred.in_ad})"
                )

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
