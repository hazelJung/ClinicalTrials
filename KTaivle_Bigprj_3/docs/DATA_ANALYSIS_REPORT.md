# 데이터 분석 및 시각화 환경 분석서
## mPBPK-ML 항체 약물 효능 예측 모델

**작성일**: 2026-01-23  
**프로젝트**: Antibody Digital Twin (KTaivle BigProject)

---

## 1. 기술 스택 (Technology Stack)

### 1.1 프로그래밍 언어 및 환경
| 기술 | 버전 | 용도 |
|------|------|------|
| Python | 3.10+ | 전체 프레임워크 기반 언어 |
| Git/GitHub | - | 버전 관리 및 협업 |

### 1.2 데이터 수집 및 처리
| 라이브러리 | 용도 |
|------------|------|
| **Pandas** | 대규모 데이터 관리 (50,000건), CSV I/O |
| **NumPy** | 수치 연산, 행렬 처리 |
| **Requests** | ChEMBL REST API 통신 |
| **xml.etree** | DrugBank XML 파싱 |

### 1.3 과학 연산 및 시뮬레이션
| 라이브러리 | 용도 |
|------------|------|
| **SciPy (odeint)** | 미분방정식(ODE) 풀이 - mPBPK 모델 핵심 |
| **Random** | Monte Carlo 샘플링 |

### 1.4 머신러닝
| 라이브러리 | 용도 |
|------------|------|
| **Scikit-learn** | Decision Tree, Random Forest, Cross-Validation |
| **Imbalanced-learn** | SMOTE (클래스 불균형 해결) |

### 1.5 시각화
| 라이브러리 | 용도 |
|------------|------|
| **Matplotlib** | 성능 지표 차트, 분포도 |

---

## 2. 데이터 수집 (Data Collection)

### 2.1 외부 데이터베이스
| 데이터 소스 | 수집 항목 | 방법 |
|-------------|-----------|------|
| **DrugBank** (XML) | 반감기, 분자량, 약물 유형 | XML 파싱 |
| **ChEMBL** (API) | 결합력 (KD) | REST API 호출 |
| **PharmGKB** (Excel) | CYP2D6 유전자 빈도 | Excel 파싱 |
| **FDA Labels** | 승인 용량, 적응증 | 문헌 조사 |

### 2.2 수집 데이터 요약
- **DrugBank**: 1,300+ 바이오의약품 PK 파라미터
- **ChEMBL**: 500+ 항체 결합력 데이터
- **PharmGKB**: 5개 인종별 CYP2D6 대립유전자 빈도

---

## 3. 데이터 전처리 (Preprocessing)

### 3.1 파라미터 범위 필터링
항체(Antibody) 약물에 특화된 파라미터 범위를 설정:

| 파라미터 | 범위 | 근거 |
|----------|------|------|
| KD | 0.01 - 20 nM | ChEMBL 항체 데이터 |
| Dose | 0.5 - 20 mg/kg | FDA 승인 항체 20종 |
| Half-life | 100 - 700 hours | IgG 표준 |
| MW | 140 - 160 kDa | IgG 분자량 |

### 3.2 Feature Engineering
| 원본 변수 | 변환 | 목적 |
|-----------|------|------|
| Dose, KD, MW, T1/2 | Log 변환 | 정규분포화 |
| Dose / KD | log_potency 생성 | 상대 효력 지표 |
| Population | One-Hot Encoding | 범주형 → 수치형 |
| Phenotype | Ordinal Encoding | PM=0, IM=1, NM=2, UM=3 |

### 3.3 클래스 불균형 처리
- **문제**: Success(52%) vs Fail(48%) 소폭 불균형
- **해결**: SMOTE (Synthetic Minority Over-sampling)

---

## 4. 데이터 분석 결과

### 4.1 생성된 가상 데이터
- **총 샘플 수**: 50,000건 (1,000 약물 × 50 환자)
- **성공률**: 52.9%
- **인종**: EUR, EAS, AFR, AMR, SAS (각 20%)

### 4.2 CYP2D6 표현형 분포
| Phenotype | 비율 | 임상적 의미 |
|-----------|------|-------------|
| NM (Normal) | 78% | 정상 대사 |
| IM (Intermediate) | 16% | 중간 대사 |
| PM (Poor) | 6% | 저대사 (독성 위험↑) |
| UM (Ultra-rapid) | 0% | 초고속 대사 |

---

## 5. 시각화 자료

### 5.1 모델 성능 비교
![Model Performance](file:///c:/Users/User/Downloads/BigPro/data/processed/visualizations/model_performance_comparison.png)

### 5.2 Feature Importance
![Feature Importance](file:///c:/Users/User/Downloads/BigPro/data/processed/visualizations/feature_importance.png)

### 5.3 파라미터 분포
![Parameter Distributions](file:///c:/Users/User/Downloads/BigPro/data/processed/visualizations/parameter_distributions.png)

### 5.4 클래스 분포 (Success/Fail)
![Class Distribution](file:///c:/Users/User/Downloads/BigPro/data/processed/visualizations/class_distribution.png)

### 5.5 인종별 CYP2D6 표현형 분포
![Phenotype by Population](file:///c:/Users/User/Downloads/BigPro/data/processed/visualizations/phenotype_by_population.png)

### 5.6 교차 검증 결과
![Cross Validation](file:///c:/Users/User/Downloads/BigPro/data/processed/visualizations/cross_validation_scores.png)

---

## 6. 모델 성능 지표

### 6.1 최종 성능 (Test Set)
| 지표 | Decision Tree | Random Forest |
|------|---------------|---------------|
| **Accuracy** | 0.711 | 0.715 |
| **Precision** | 0.634 | 0.639 |
| **Recall** | 0.829 | 0.827 |
| **F1 Score** | 0.719 | 0.721 |
| **ROC-AUC** | 0.782 | 0.787 |

### 6.2 교차 검증 (5-Fold CV)
| 모델 | F1 Score (Mean ± Std) |
|------|----------------------|
| Decision Tree | 0.723 ± 0.005 |
| Random Forest | 0.727 ± 0.004 |

### 6.3 주요 Feature Importance
| 순위 | Feature | Importance |
|------|---------|------------|
| 1 | log_dose | 52.9% |
| 2 | log_potency | 23.8% |
| 3 | log_KD | 7.4% |
| 4 | log_halflife | 4.9% |
| 5 | log_T0 | 3.4% |

---

## 7. 결론

### 7.1 모델 특성
- **도메인**: 단일클론항체(mAb) 전용
- **강점**: 용량(Dose) 및 결합력(KD) 기반 예측 정확도 높음
- **한계**: Cytokine 등 저분자 단백질에는 적용 불가

### 7.2 핵심 인사이트
1. **Dose가 가장 중요**: 적절한 용량 설정이 성공의 핵심
2. **Potency 효과 확인**: Dose/KD 비율이 독립적 예측력 보유
3. **유전적 다양성 반영**: CYP2D6 표현형이 독성 예측에 기여

---

**작성자**: AI Assistant (Antigravity)  
**검토 필요**: 프로젝트 담당자
