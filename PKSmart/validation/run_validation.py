
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# 경로 설정
PROJECT_ROOT = ".."
sys.path.append(PROJECT_ROOT)
# validation folder is current dir, but pksmart is in parent

from pksmart.models import PKSmartPipeline
from pksmart.features import generate_features, standardize_smiles

# Metric Functions
def fraction_within_fold(pred, true, fold=2, log_transformed=True, fu_endpoint=False):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    if fu_endpoint:
        ratio = pred / np.where(true == 0, 1e-10, true)
        within = (ratio <= fold) & (ratio >= 1/fold)
    elif log_transformed:
        diff = pred - true
        within = np.abs(diff) <= np.log10(fold)
    else:
        ratio = pred / np.where(true == 0, 1e-10, true)
        within = (ratio <= fold) & (ratio >= 1/fold)

    return within.mean() * 100.0

def gmfe(pred, true, log_transformed=True, fu_endpoint=False):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    if fu_endpoint:
        logs = np.abs(np.log10(pred / np.where(true == 0, 1e-10, true)))
    elif log_transformed:
        logs = np.abs(pred - true)
    else:
        logs = np.abs(np.log10(pred / np.where(true == 0, 1e-10, true)))

    return 10 ** logs.mean()

def bias_median(pred, true, log_transformed=True, fu_endpoint=False):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    if fu_endpoint:
        diff = pred - true
    elif log_transformed:
        diff = 10 ** pred - 10 ** true
    else:
        diff = pred - true

    return float(np.median(diff))

def eval_regression_metrics(pred, true, log_transformed=True, fu_endpoint=False):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)

    return {
        "R2": float(r2_score(true, pred)),
        "RMSE": float(rmse),
        "GMFE": float(gmfe(pred, true, log_transformed=log_transformed, fu_endpoint=fu_endpoint)),
        "bias_median": float(bias_median(pred, true, log_transformed, fu_endpoint)),
    }

# 1. 필터링된 외부 데이터 및 모델 로드
EXTERNAL_DATA_PATH = "external_test_filtered.csv"
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

print(f"Loading data from: {EXTERNAL_DATA_PATH}")
df = pd.read_csv(EXTERNAL_DATA_PATH)
print(f"Data size: {len(df)}")

# 2. 피처 추출
print("Generating features...")
features_df = generate_features(df['smiles_r'])

if features_df is None or len(features_df) == 0:
    print("Feature generation failed.")
    sys.exit(1)

print(f"Features generated: {len(features_df)}")

# Merge with original data to align targets
# features_df['smiles_r'] contains STANDARDIZED smiles.
# We must standardize df['smiles_r'] to match.
print("Standardizing original SMILES for merge...")
df['std_smiles_r'] = df['smiles_r'].apply(standardize_smiles)

# Drop rows where standardization failed (None)
df_clean = df.dropna(subset=['std_smiles_r'])
print(f"Original valid SMILES: {len(df_clean)}")

# Merge on standardized smiles
# features_df has 'smiles_r' as the standardized smiles
features_df = features_df.rename(columns={'smiles_r': 'std_smiles_r'})

merged_df = pd.merge(df_clean, features_df, on='std_smiles_r', how='inner')
print(f"Merged Data size: {len(merged_df)}")

print("Loading pipeline...")
pipeline = PKSmartPipeline(MODEL_DIR)

# 3. 예측 수행
print("Running prediction pipeline...")
# Pipeline expects features. 
preds = pipeline.run_pipeline(merged_df)

# 결과 병합
df_pred = pd.concat([merged_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

# 타겟 변수 및 설정
# Model targets: 'human_VDss_L_kg', 'human_CL_mL_min_kg', 'human_fup', 'human_mrt', 'human_thalf'
# Pipeline outputs: same names?
# predict_human_pk returns df with cols: human_CL_mL_min_kg, etc.
# But check pipeline.run_pipeline return.
# It returns animal preds concat human preds.
# Columns should include human params.

targets = {
    'human_VDss_L_kg': {'log': True, 'fu': False}, 
    'human_CL_mL_min_kg': {'log': True, 'fu': False}, 
    'human_fup': {'log': False, 'fu': True}, 
    'human_mrt': {'log': True, 'fu': False}, 
    'human_thalf': {'log': True, 'fu': False}
}

# 평가
print("\nValidation Results (Filtered Data):")
for target, config in targets.items():
    # Pipeline returns columns with same names as target?
    # Usually pipeline returns preds. Let's assume col names match target keys.
    if target in df.columns and target in df_pred.columns:
        # Note: df_pred has both original and predicted?
        # pd.concat might duplicate columns if they have same name.
        # But pipeline result likely has same names as targets.
        # Let's assume pipeline output columns are the predictions.
        # But if we concat, we get two columns with same name.
        # We should check pipeline output columns.
        
        # Pipeline returns DataFrame with predictions.
        # Columns in preds: likely 'human_VDss_L_kg', etc.
        # Original df has 'human_VDss_L_kg' (observed).
        # We need to distinguish.
        
        y_true = merged_df[target]
        # Preds from pipeline. 
        # Since we did concat, accessing df_pred[target] might return DF with 2 cols.
        # So we should use preds dataframe directly for y_pred.
        
        if target in preds.columns:
            y_pred = preds[target]
            
            # NaN 제거
            valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            if valid_mask.sum() > 0:
                metrics = eval_regression_metrics(
                    y_pred[valid_mask], 
                    y_true[valid_mask],
                    log_transformed=config['log'],
                    fu_endpoint=config['fu']
                )
                print(f"\nTarget: {target} (N={valid_mask.sum()})")
                print(f"  R2: {metrics['R2']:.4f}")
                print(f"  RMSE: {metrics['RMSE']:.4f}")
                print(f"  GMFE: {metrics['GMFE']:.4f}")
            else:
                print(f"\nTarget: {target} - No valid samples")
        else:
             print(f"Target {target} not in predictions.")
    else:
        print(f"Target {target} not in dataframe.")
