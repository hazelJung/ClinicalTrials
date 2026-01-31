# 통합 mPBPK-ML 프레임워크 구현 계획

> **프로젝트**: AI-Driven In-silico Clinical Trial & Toxicity Prediction Platform
> **상태**: Phase 1 준비 완료
> **최종 업데이트**: 2026-01-20

---

## 1. 프로젝트 목표

**약물 파라미터 변이** + **환자 유전체 변이** = 정밀 의료 예측 플랫폼

### 핵심 가치
- "이 약물이 **어떤 환자 집단**에서 효과적인가?"
- "**어떤 환자**에서 노출 기반 독성 위험이 높은가?"

### 예측 라벨 (Multi-Label)
| 라벨 | 정의 | 임계값 |
|------|------|--------|
| `efficacy_success` | TO ≥ 90% | 성공=1 |
| `exposure_toxicity_risk` | Cmax 또는 AUC > 안전 한계 | 위험=1 |
| `um_effective` | UM 환자에서도 TO ≥ 90% | 성공=1 |
| `pm_safe` | PM 환자에서 Cmax < 독성 한계 | 안전=1 |

### QSAR vs mPBPK 독성 예측 구분
| 모델 | 예측 대상 | 환자 맞춤형 |
|------|----------|------------|
| **QSAR** | 분자 자체의 **본질적 독성** | ❌ 불가 |
| **mPBPK** | 환자별 **노출 기반 독성** (축적) | ✅ 가능 |

---

## 2. 데이터 소스 (수집 완료)

| 데이터 | 파일 | 상태 |
|--------|------|------|
| DrugBank | `data/raw/full database.xml` | ✅ 1.9GB |
| ChEMBL API | `src/data/fetch_chembl.py` | ✅ 구현됨 |
| CYP2D6 | `data/raw/genes/CYP2D6_*.xlsx` | ✅ 로드됨 |
| Tox21 | `data/raw/tox21.csv` | ✅ QSAR용 |
| ClinTox | `data/raw/clintox.csv` | ✅ QSAR용 |
| DILIrank | `data/raw/Drug Induced Liver...xlsx` | ✅ QSAR용 |

---

## 3. 현재 진행 상태 (완료된 작업)

### ✅ CYP2D6 가상 코호트 생성기
- `src/data/parse_cyp2d6.py` - PharmGKB 데이터 로드
- 5개 인구(EUR, EAS, AFR, AMR, SAS) 지원
- PM/IM/NM/UM 표현형 분포 검증 완료
- CNV 데이터 보정으로 UM 비율 정상화

### ✅ mPBPK + CYP2D6 통합
- `src/models/mpbpk_engine.py` - 동적 파서 연동
- `simulate_population_cohort()` - 가상 코호트 시뮬레이션
- `summarize_cohort_results()` - 표현형별 결과 요약
- 테스트 스크립트: `tests/test_mpbpk_integration.py`

---

## 4. 남은 작업 (Phase 1-4)

### Phase 1: 파라미터 범위 추출 (완료)
- [x] DrugBank 파싱 → PK 범위 통계 (Half-life)
- [x] ChEMBL API → KD 분포 (항체 필터링)
- [x] FDA 승인 항체(n=20) 기반 정밀 범위 검증 (KD, Dose, MW, T1/2)

### Phase 2: 시뮬레이션 매트릭스 (완료)
- [x] `src/models/batch_simulator.py` 구현
- [x] 약물 1,000 × 환자 50 = 50K 시뮬레이션 완료
- [x] Multi-label 생성 (efficacy + exposure_toxicity)
- [x] 항체 특화 파라미터 적용 (MW 140-160kDa, Dose 0.5-20mg/kg)

### Phase 3: ML 훈련 (완료)
- [x] `src/models/mpbpk_ml.py` 구현
- [x] Decision Tree + Random Forest 훈련
- [x] SMOTE 불균형 처리 적용
- [x] 5-Fold CV 검증 (F1 Score ~0.73 달성)

### Phase 4: 검증 (완료)
- [x] ML 성능 (Accuracy 72%, ROC-AUC 0.80)
- [x] 약리학적 검증 (Dose/KD 중요도 확인)
- [x] 실제 약물 매칭 검증 (Adalimumab 등 20종)

---

## 5. 파일 구조

```
Big_Project/
├── src/
│   ├── data/
│   │   ├── parse_cyp2d6.py     ✅ CYP2D6 파서 (동적 로딩)
│   │   ├── parse_drugbank.py   ✅ DrugBank XML 파서
│   │   └── fetch_chembl.py     ✅ ChEMBL API 클라이언트
│   └── models/
│       ├── mpbpk_engine.py     ✅ mPBPK + CYP2D6 통합
│       ├── qsar_engine.py      ✅ QSAR 독성 예측
│       ├── batch_simulator.py  ✅ 베치 시뮬레이션 (50K 환자)
│       └── mpbpk_ml.py         ✅ ML 분류기 (RF/DT)
├── data/
│   ├── raw/                    ✅ 원본 데이터
│   └── processed/              ✅ 처리된 데이터 (생성됨)
└── tests/
    ├── test_cyp2d6_validation.py    ✅ CYP2D6 검증
    └── test_mpbpk_integration.py    ✅ 통합 테스트
```

---

## 6. 향후 계획 (Phase 5+)

- **Web Interface**: Streamlit 기반 시뮬레이션 대시보드 구축
- **Report Generation**: PDF/HTML 자동 보고서 생성 기능
- **Expansion**: Peptide 등 다른 모달리티로 확장
