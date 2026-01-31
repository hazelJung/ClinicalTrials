# PROJECT_CONTEXT.md

> AI 에이전트가 프로젝트를 이해하고 구현할 수 있도록 작성된 통합 컨텍스트 문서

---

## 1. Project Overview

| 항목 | 내용 |
|------|------|
| **Project Name** | AI-Driven In-silico Clinical Trial & Toxicity Prediction Platform |
| **Goal** | Dual-Engine: 약물 효능(mPBPK) 시뮬레이션 + 독성(QSAR) 예측 |
| **Core Philosophy** | "4-Paper Rule" - 4개 논문 기반 엄격한 구현 |

---

## 2. Dual-Engine Architecture

```
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
┌───────────────────┐               ┌───────────────────┐
│  ENGINE 1: mPBPK  │               │  ENGINE 2: QSAR   │
│  (Physics-based)  │               │  (Data-driven)    │
├───────────────────┤               ├───────────────────┤
│ 5-Compartment ODE │               │ ML Classification │
│ + TMDD Binding    │               │ + Toxicity Score  │
│ → TO% (효능)       │               │ → Alerts          │
└─────────┬─────────┘               └─────────┬─────────┘
          └───────────────┬───────────────────┘
                          ▼
              ┌───────────────────┐
              │  FINAL DECISION   │
              │  Success/Failure  │
              └───────────────────┘
```

---

## 3. Directory Structure & Reference Maps

```text
root/
├── data/
│   ├── raw/                  # Raw CSVs from team members
│   └── processed/            # Cleaned data for modeling
├── paper/
│   ├── mPBPK/
│   │   ├── CPT-118-378.pdf   # [Main] Sanofi Model (Equations)
│   │   └── fphar-13-895556.pdf # [Sub] Validation Data
│   └── QSAR/
│       ├── Tropsha-2010.pdf  # [Main] Validation Guidelines
│       └── Pharms-2025.pdf   # [Sub] ML Implementation Guide
├── src/
│   └── models/
│       ├── mpbpk_engine.py   # Physics-based Engine
│       └── qsar_engine.py    # Data-driven Engine
├── venv/                     # Python virtual environment
└── PROJECT_CONTEXT.md
```

---

## 4. mPBPK Engine Implementation Guide

### 4.1 Core ODE System (7 State Variables)

```python
def antibody_mpbpk_tmdd(y, t, params):
    A_plasma, A_tight, A_leaky, A_liver, A_lymph, T_free, DT_complex = y
    
    # Concentration
    C_plasma_free = (A_plasma / V_P) * fu
    
    # Tissue flux
    flux_tight = Q_T * (C_plasma_free - C_tight_free)
    flux_leaky = Q_L * (C_plasma_free - C_leaky_free)
    flux_liver = Q_Liv * (C_plasma_free - C_liver_free)
    
    # TMDD binding
    binding = kon * C_plasma_free * T_free
    unbinding = koff * DT_complex
    
    return [dA_plasma, dA_tight, dA_leaky, dA_liver, dA_lymph, dT_free, dDT_complex]
```

### 4.2 Target Occupancy Calculation

```python
TO_percent = ((T_total - T_free) / T_total) * 100
# Success: TO ≥ 90%
```

### 4.3 Key Parameters

| Category | Variable | Range | Unit |
|----------|----------|-------|------|
| Patient | Age | 18-85 | years |
| Patient | BMI | 16-45 | kg/m² |
| Patient | fu | 0.4-0.99 | - |
| Drug | KD_nM | 0.001-10000 | nM |
| Drug | Dose_mg | 0.1-1000 | mg |
| Drug | Dosing_Interval | 7,14,21,28 | days |
| Target | Baseline_nM | 1-1000 | nM |
| Target | Halflife_hr | 0.5-100 | hr |

### 4.4 Hard Mode Tuning

| Parameter | Default | Hard Mode |
|-----------|---------|-----------|
| ksyn | kdeg × T0 | kdeg × T0 × **10** |
| kint | 0.1 | **0.3** |
| T0 scaling | 1× | **1000×** |

---

## 5. QSAR Engine Implementation Guide

### 5.1 Validation Guidelines (Tropsha-2010)
1. Applicability Domain 정의
2. Y-randomization test
3. External validation set 사용
4. 다중 검증 지표 (R², Q², RMSE)

### 5.2 ML Implementation (Pharms-2025)
- Feature engineering from molecular descriptors
- Ensemble methods (RF, XGBoost)
- SHAP for interpretability

---

## 6. ML Classification Layer

```python
# SMOTE for class imbalance
smote = SMOTE(k_neighbors=min(5, minority_count - 1))
X_res, y_res = smote.fit_resample(X_train, y_train)

# Decision Tree
clf = DecisionTreeClassifier(max_depth=6)
clf.fit(X_res, y_res)
```

### Feature Importance (v5 Model)
1. Dose_mg (57%)
2. Target_Halflife_hr (18%)
3. Target_Baseline_nM (10%)
4. fu (8%)
5. KD_nM (6%)

---

## 7. Performance Benchmarks

| Metric | Target | Achieved |
|--------|--------|----------|
| Accuracy | >90% | 96.4% |
| ROC-AUC | >95% | 98.7% |
| Recall (Failure) | >90% | 95.7% |

---

## 8. Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| Simulation | SciPy, NumPy |
| Data | Pandas |
| ML | Scikit-learn, Imbalanced-learn |
| Visualization | Matplotlib |
| Web (Optional) | Next.js, Mantine UI |

---

## 9. Team Collaboration Rules

| Role | Responsibility |
|------|----------------|
| Data Team | Upload CSVs to `data/raw/` |
| Model Team | Develop engines in `src/models/` |
| Validation Team | Paper-based testing |

### Git Workflow
1. Create feature branch
2. Develop & test
3. PR with code review
4. Merge to main
