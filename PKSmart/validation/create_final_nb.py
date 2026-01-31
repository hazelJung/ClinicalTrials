import json
import os

nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# PKSmart 모델 성능 검증 최종 리포트 (V2)\n",
    "\n",
    "이 노트북은 PKSmart 모델의 예측 성능을 다각도로 분석하고, 시각화 자료를 생성하기 위해 제작되었습니다.\n",
    "\n",
    "### 분석 내용:\n",
    "1. **학습 데이터 재현성 검증**: R2, GMFE, within 2-fold%\n",
    "2. **누적 오차 곡선 (Cumulative Fold Error)**: 재현 신뢰도 증명\n",
    "3. **잔차 분석 (Residual Plot)**: 예측 일관성 및 편향 확인\n",
    "4. **주요 피처 분석 (Feature Importance)**: 과학적 타당성 검증\n",
    "5. **전체 지표 요약 (Summary Chart)**: 최종 보고용 바 차트"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# 경로 설정\n",
    "PROJECT_ROOT = \"..\"\n",
    "sys.path.append(PROJECT_ROOT)\n",
    "sys.path.append(os.path.join(PROJECT_ROOT, \"validation\"))\n",
    "\n",
    "from pksmart.models import PKSmartPipeline\n",
    "from validation_utils import (eval_regression_metrics, plot_pred_vs_true, \n",
    "                               plot_fold_error_hist, plot_cumulative_fold_error, \n",
    "                               plot_residual, plot_feature_importance)\n",
    "\n",
    "sns.set(style=\"whitegrid\", font_scale=1.2)\n",
    "plt.rcParams[\"figure.figsize\"] = (6, 6)\n",
    "try:\n",
    "    plt.rcParams['font.family'] = 'AppleGothic' # Mac용\n",
    "except:\n",
    "    pass\n",
    "\n",
    "print(\"환경 설정 및 모듈 로드 완료!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. 데이터 및 모델 로드\n",
    "DATA_PATH = os.path.join(PROJECT_ROOT, \"data\", \"Human_PK_data.csv\")\n",
    "MODEL_DIR = os.path.join(PROJECT_ROOT, \"models\")\n",
    "\n",
    "df_raw = pd.read_csv(DATA_PATH)\n",
    "df = df_raw.dropna(subset=['smiles_r']).reset_index(drop=True)\n",
    "\n",
    "pipeline = PKSmartPipeline(MODEL_DIR)\n",
    "print(f\"데이터 로드 완료: {len(df)} compounds\")\n",
    "print(f\"모델 로드 완료: {len(pipeline.models)} models found\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. 피처 생성 및 예측 실행\n",
    "from pksmart.features import generate_features\n",
    "\n",
    "print(\"화학 구조로부터 피처 추출 중 (Mordred & Morgan)...\")\n",
    "features_df = generate_features(df['smiles_r'].tolist())\n",
    "\n",
    "print(\"예측 파이프라인 가동 중...\")\n",
    "preds_df = pipeline.run_pipeline(features_df)\n",
    "preds_df = preds_df.add_suffix(\"_pred\") # 컬럼명 구분\n",
    "\n",
    "results = pd.concat([df, preds_df], axis=1)\n",
    "print(\"예측 완료 및 결과 결합 성공!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. 기본 통계 지표 계산\n",
    "endpoints = {\n",
    "    \"human_CL_mL_min_kg\": \"Clearance (CL)\",\n",
    "    \"human_VDss_L_kg\": \"Vol. of Distribution (VDss)\",\n",
    "    \"human_fup\": \"Fraction Unbound (fup)\",\n",
    "    \"human_mrt\": \"Mean Residence Time (MRT)\",\n",
    "    \"human_thalf\": \"Half-life (t1/2)\"\n",
    "}\n",
    "\n",
    "all_metrics = []\n",
    "for col, name in endpoints.items():\n",
    "    valid_idx = results[col].dropna().index\n",
    "    y_true = results.loc[valid_idx, col]\n",
    "    y_pred = results.loc[valid_idx, col + \"_pred\"]\n",
    "    \n",
    "    is_log = \"fup\" not in col\n",
    "    is_fu = \"fup\" in col\n",
    "    \n",
    "    if is_log:\n",
    "        mask = y_true > 0\n",
    "        y_metrics_true = np.log10(y_true[mask])\n",
    "        y_metrics_pred = y_pred[mask]\n",
    "    else:\n",
    "        y_metrics_true = y_true\n",
    "        y_metrics_pred = y_pred\n",
    "\n",
    "    metrics = eval_regression_metrics(y_metrics_pred, y_metrics_true, log_transformed=is_log, fu_endpoint=is_fu)\n",
    "    metrics['Endpoint'] = name\n",
    "    all_metrics.append(metrics)\n",
    "\n",
    "metrics_df = pd.DataFrame(all_metrics).set_index(\"Endpoint\")\n",
    "metrics_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. 상세 시각화 분석\n",
    "각 지표별로 **예측 accuracy**, **누적 오차(Reliability)**, **잔차(Bias)**를 분석합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "for col, name in endpoints.items():\n",
    "    valid_idx = results[col].dropna().index\n",
    "    y_true = results.loc[valid_idx, col]\n",
    "    y_pred = results.loc[valid_idx, col + \"_pred\"]\n",
    "    \n",
    "    is_log = \"fup\" not in col\n",
    "    if is_log:\n",
    "        mask = y_true > 0\n",
    "        y_t_log = np.log10(y_true[mask])\n",
    "        y_p_log = y_pred[mask]\n",
    "        \n",
    "        # 1. 시각적 비교\n",
    "        plot_pred_vs_true(y_p_log, y_t_log, title=f\"{name}: Predicted vs Observed\")\n",
    "        \n",
    "        # 2. 누적 오차 곡선 (정확도 재현성 증명)\n",
    "        plot_cumulative_fold_error(y_p_log, y_t_log, log_transformed=True, title=f\"{name}: Cumulative Fold Error\")\n",
    "        \n",
    "        # 3. 잔차 분석 (일관성 증명)\n",
    "        plot_residual(y_p_log, y_t_log, title=f\"{name}: Residual Analysis\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. 모델 해석 (Feature Importance)\n",
    "모델이 어떤 근거로 인간의 CL을 예측했는지 주요 피처를 분석합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "target_param = 'human_CL_mL_min_kg'\n",
    "if target_param in pipeline.models:\n",
    "    model = pipeline.models[target_param]\n",
    "    feat_names = features_df.select_dtypes(include=[np.number]).columns.tolist() + [f\"{s}_{p}\" for s in pipeline.species for p in pipeline.params]\n",
    "    plot_feature_importance(model, feat_names, top_n=15, title=f\"What drives {target_param}? (Top 15 Features)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. 결론용 요약 차트\n",
    "전체 파라미터의 2-fold 이내 예측 성공률 요약입니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 6))\n",
    "sns.barplot(x=\"Endpoint\", y=\"within_2fold_%\", data=metrics_df.reset_index(), palette=\"viridis\")\n",
    "plt.axhline(50, color='r', linestyle='--', label='Industry Standard (50%)')\n",
    "plt.title(\"Accuracy Summary: Percentage within 2-fold Error\")\n",
    "plt.ylabel(\"Percentage (%)\")\n",
    "plt.ylim(0, 105)\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open(\"validation_report_v2.ipynb\", \"w\", encoding=\"utf-8\") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
