# QSAR-mPBPK 통합 임상시험 시뮬레이션 플랫폼 구현 계획서

**작성일**: 2026-01-26  
**버전**: 3.0 (Full Workflow 지원)

---

## 1. 프로젝트 개요

### 1.1 목표
FDA 현대화법 2.0에 부합하는 **AI 기반 임상시험 설계 지원 플랫폼** 구축.
동물실험 대체/보완을 위한 인실리코 시뮬레이션과 지능형 임상 가이드를 제공.

### 1.2 시스템 워크플로우 (5단계)

```
┌─────────────────────────────────────────────────────────────────┐
│  1단계: 입력 및 De-risking                                       │
│  - 화학 구조(SMILES), CLint, fu 입력                            │
│  - QSAR 독성 스크리닝 (hERG, 간독성)                            │
│  - 유사 약물 실패 히스토리 매칭                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  2단계: 유전체 결합 인실리코 시뮬레이션                          │
│  - 다인종 가상 코호트 생성 (1000+ 가상 환자)                     │
│  - PBPK 모델링 + PharmGKB 유전자 통합                           │
│  - 유전형별 AUC, Cmax 시뮬레이션                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  3단계: 지능형 임상 가이드 제안                                  │
│  - 성공 확률(PoS) 산출                                          │
│  - 최적 환자군 추천 (유전형 기반)                               │
│  - 독성 고위험군 제외 권고                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  4단계: 사용자 미세 조정                                         │
│  - 웹 UI 기반 파라미터 튜닝                                     │
│  - What-if 시나리오 시뮬레이션                                  │
│  - 실시간 성공 확률 변화 확인                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  5단계: 최종 리포트 생성                                         │
│  - 임상시험 프로토콜 초안 (DOCX/PDF)                            │
│  - SAS/R 호환 통계 데이터셋                                     │
│  - 시뮬레이션 근거 문서화                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 파일 구조 (확장판)

```
src/
├── models/
│   ├── mpbpk_engine.py          # ✅ 기존 (PBPK 시뮬레이션)
│   ├── batch_simulator.py       # ✅ 기존 (코호트 + 노이즈)
│   ├── qsar_predictor.py        # 🆕 QSAR 기술자 + 활성 예측
│   ├── herg_predictor.py        # 🆕 hERG 심독성 예측
│   ├── ivive_calculator.py      # 🆕 IVIVE Well-Stirred Model
│   ├── safety_calculator.py     # 🆕 Safety Margin + PoS
│   ├── toxicophore_analyzer.py  # 🆕 독성 작용기 스크리닝
│   └── ad_analyzer.py           # 🆕 적용 도메인 분석
│
├── data/
│   ├── parse_cyp2d6.py          # ✅ 기존 (CYP2D6 유전체)
│   └── drug_similarity.py       # 🆕 유사 약물 검색 (PubChem/DrugBank)
│
├── services/
│   ├── drug_safety_service.py   # 🆕 1-2단계 통합 서비스
│   ├── recommendation_engine.py # 🆕 3단계 추천 엔진
│   └── report_generator.py      # 🆕 5단계 리포트 생성
│
└── ui/
    └── streamlit_app.py         # 🆕 4단계 웹 인터페이스

data/
├── models/
│   ├── qsar_rf_model.pkl        # QSAR RF 모델
│   └── herg_rf_model.pkl        # hERG 예측 모델
└── reference/
    └── failed_drugs_db.json     # 실패 약물 히스토리 DB
```

---

## 3. 구현 단계 (6 Phases)

### Phase 1: 핵심 인프라 (2-3일)
**목표**: 기본 계산 모듈 구현

| 모듈 | 기능 | 우선순위 |
|------|------|----------|
| `ivive_calculator.py` | CLint → CLh → Cmax 계산 | 필수 |
| `safety_calculator.py` | SM = IC50/Cmax, 위험 등급 분류 | 필수 |
| `DrugInput/DrugResult` | 공통 데이터클래스 정의 | 필수 |

**핵심 수식:**
```python
# Well-Stirred Model
CLh = (Qh × fu × CLint) / (Qh + fu × CLint)
# Safety Margin
SM = IC50 / Cmax
# Risk: SM ≥ 30 (Safe), 10-30 (Moderate), < 10 (Concern)
```

---

### Phase 2: 분자 분석 (3-4일)
**목표**: 구조 기반 독성 예측

| 모듈 | 기능 | 의존성 |
|------|------|--------|
| `qsar_predictor.py` | 41개 분자 기술자 계산 | RDKit |
| `toxicophore_analyzer.py` | 6개 독성 작용기 SMARTS | RDKit |
| `herg_predictor.py` | hERG IC50 예측 (심독성) | QSAR 모델 |

**Toxicophore SMARTS 패턴:**
```python
TOXIC_SMARTS = {
    'nitro': '[N+](=O)[O-]',
    'quinone': 'O=C1C=CC(=O)C=C1',
    'epoxide': 'C1OC1',
    'michael_acceptor': 'C=CC(=O)',
    'aldehyde': '[CH]=O',
    'acyl_halide': 'C(=O)[F,Cl,Br,I]'
}
```

---

### Phase 3: 데이터 연동 (2-3일)
**목표**: 외부 DB 연동 및 유사 약물 검색

| 모듈 | 기능 | API |
|------|------|-----|
| `drug_similarity.py` | 유사 구조 약물 검색 | PubChem REST API |
| `drug_similarity.py` | 실패 히스토리 조회 | DrugBank (로컬 캐시) |
| `ad_analyzer.py` | 적용 도메인 분석 | 학습 데이터 기반 |

**PubChem 유사도 검색:**
```python
# Tanimoto 유사도 기반 검색
url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/{smiles}/cids/JSON?Threshold=85"
```

---

### Phase 4: 추천 엔진 (3-4일)
**목표**: 지능형 임상 가이드 제안

| 모듈 | 기능 | 입력 |
|------|------|------|
| `recommendation_engine.py` | 성공 확률(PoS) 계산 | SM, AD Score, Toxicophore |
| `recommendation_engine.py` | 최적 환자군 추천 | CYP2D6 표현형별 분석 |
| `recommendation_engine.py` | 고위험군 제외 권고 | 유전형 기반 독성 예측 |

**PoS 계산 로직:**
```python
def calculate_pos(sm: float, ad_score: float, toxicophore_count: int) -> float:
    """
    성공 확률 산출 (0-100%)
    
    가중치:
    - Safety Margin: 40%
    - AD Score: 30%
    - Toxicophore: 30%
    """
    sm_score = min(100, sm * 3.33)  # SM 30 → 100%
    tox_score = max(0, 100 - toxicophore_count * 20)
    
    pos = 0.4 * sm_score + 0.3 * ad_score + 0.3 * tox_score
    return round(pos, 1)
```

**최적 환자군 추천:**
```python
def recommend_patient_group(cohort_results: pd.DataFrame) -> Dict:
    """유전형별 효능/독성 분석 기반 추천"""
    
    # 1. 독성 고위험군 식별 (PM: 약물 축적 위험)
    high_risk = cohort_results[cohort_results['phenotype'] == 'PM']
    
    # 2. 효능 저위험군 식별 (UM: 약효 부족 위험)
    low_efficacy = cohort_results[cohort_results['phenotype'] == 'UM']
    
    # 3. 최적 타겟군 추천
    optimal = cohort_results[cohort_results['phenotype'].isin(['NM', 'IM'])]
    
    return {
        'exclude_phenotypes': ['PM'],
        'caution_phenotypes': ['UM'],
        'target_phenotypes': ['NM', 'IM'],
        'recommended_n': len(optimal),
        'exclusion_rate': len(high_risk) / len(cohort_results) * 100
    }
```

---

### Phase 5: 웹 인터페이스 (3-4일)
**목표**: 사용자 친화적 파라미터 튜닝 UI

| 컴포넌트 | 기능 | 프레임워크 |
|----------|------|------------|
| `streamlit_app.py` | 메인 대시보드 | Streamlit |
| 파라미터 슬라이더 | Dose, CLint, fu 조절 | Streamlit |
| 결과 시각화 | Cmax 분포, PoS 게이지 | Plotly |
| What-if 시나리오 | 조건별 비교 분석 | Pandas |

**Streamlit UI 구조:**
```python
import streamlit as st

st.title("🧬 Drug Safety Simulation Platform")

# Sidebar: 입력 파라미터
with st.sidebar:
    smiles = st.text_input("SMILES", "CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    ic50 = st.slider("IC50 (μM)", 0.1, 100.0, 0.85)
    clint = st.slider("CLint (μL/min/mg)", 1.0, 100.0, 12.5)
    fu = st.slider("fu (Fraction Unbound)", 0.01, 1.0, 0.02)

# Main: 결과 대시보드
col1, col2, col3 = st.columns(3)
col1.metric("Safety Margin", f"{sm:.1f}x", delta_color="normal")
col2.metric("PoS", f"{pos:.1f}%")
col3.metric("AD Score", f"{ad_score:.1f}%")

# What-if 시나리오
if st.button("Run What-if Analysis"):
    scenarios = run_whatif_scenarios(drug_input)
    st.dataframe(scenarios)
```

---

### Phase 6: 리포트 생성 (2-3일)
**목표**: 규제 제출용 문서 자동 생성

| 모듈 | 기능 | 출력 포맷 |
|------|------|-----------|
| `report_generator.py` | 프로토콜 초안 생성 | DOCX, PDF |
| `report_generator.py` | PBPK 분석 요약 | Markdown |
| `report_generator.py` | 통계 데이터셋 | CSV, XPT (SAS), RDS (R) |

**SAS XPT 출력:**
```python
import pyreadstat

def export_to_sas(df: pd.DataFrame, filename: str):
    """SAS 호환 XPT 파일 생성"""
    pyreadstat.write_xport(df, f"{filename}.xpt")
```

**프로토콜 초안 템플릿:**
```markdown
# Clinical Trial Protocol Draft

## 1. Study Design
- **Target Population**: {recommend.target_phenotypes}
- **Exclusion Criteria**: {recommend.exclude_phenotypes} genotype carriers
- **Sample Size**: {recommend.recommended_n} subjects

## 2. PK Simulation Summary
- **Predicted Cmax**: {cmax_median} μM (P5: {cmax_p5}, P95: {cmax_p95})
- **Safety Margin**: {sm}x ({risk_category})

## 3. Risk Assessment
- **Probability of Success**: {pos}%
- **Toxicophore Alerts**: {toxicophore_count}/6
```

---

## 4. 의존성 (확장)

### requirements.txt 추가 항목
```
# Phase 1-2: Core
rdkit-pypi>=2023.9.1

# Phase 3: Data Integration
requests>=2.31.0

# Phase 5: Web UI
streamlit>=1.32.0
plotly>=5.18.0

# Phase 6: Report Generation
python-docx>=1.1.0
pyreadstat>=1.2.0
```

---

## 5. 일정 요약

| Phase | 내용 | 기간 | 누적 |
|-------|------|------|------|
| 1 | 핵심 인프라 (IVIVE, Safety) | 2-3일 | 3일 |
| 2 | 분자 분석 (QSAR, Toxicophore, hERG) | 3-4일 | 7일 |
| 3 | 데이터 연동 (PubChem, DrugBank) | 2-3일 | 10일 |
| 4 | 추천 엔진 (PoS, 환자군) | 3-4일 | 14일 |
| 5 | 웹 UI (Streamlit) | 3-4일 | 18일 |
| 6 | 리포트 생성 (DOCX, SAS) | 2-3일 | **21일** |

**총 예상 기간: 3주 (15-21 영업일)**

---

## 6. 기존 모듈 재사용 매핑

| 기존 모듈 | 활용 Phase | 역할 |
|-----------|------------|------|
| `mpbpk_engine.py` | Phase 1 | PBPK 기반 Cmax 계산 확장 |
| `batch_simulator.py` | Phase 1 | 노이즈 주입, 코호트 시뮬레이션 |
| `parse_cyp2d6.py` | Phase 4 | 유전형별 환자군 분류 |
| `mpbpk_ml.py` | Phase 4 | ML 기반 PoS 보정 |

---

## 7. 검증 기준

| 항목 | 목표 | 측정 방법 |
|------|------|-----------|
| IVIVE 정확도 | 문헌 대비 ±30% | Verapamil, Midazolam 검증 |
| hERG 예측 | AUC ≥ 0.80 | 테스트셋 검증 |
| PoS 신뢰도 | 임상 결과와 상관 ≥ 0.6 | 과거 사례 회고 분석 |
| UI 응답 시간 | ≤ 3초/분석 | 부하 테스트 |

---

## 8. 기대 효과

1. **비임상 시험 혁신**: FDA 현대화법 2.0 부합, 동물실험 보완/대체
2. **임상 성공률 극대화**: 유전형 기반 정밀 환자군 타겟팅
3. **업무 효율화**: 임상 설계 자동화로 수개월 → 수일 단축
4. **규제 대응력**: SAS 호환 데이터 즉시 제출 가능

---

**다음 단계**: Phase 1 구현 시작 (IVIVECalculator, SafetyMarginCalculator)
