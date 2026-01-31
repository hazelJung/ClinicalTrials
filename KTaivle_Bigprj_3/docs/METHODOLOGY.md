# 데이터 생성 및 모델링 방법론
## mPBPK-ML 가상 임상시험 시뮬레이션

**작성일**: 2026-01-23  
**관련 파일**: `batch_simulator.py`, `mpbpk_engine.py`, `parse_cyp2d6.py`

---

## 1. 파라미터 추출 (Parameter Extraction)

### 1.1 데이터 소스별 추출 방법

| 데이터 소스 | 추출 파라미터 | 추출 방법 | 관련 스크립트 |
|-------------|---------------|-----------|---------------|
| **DrugBank** (XML) | Half-life, MW | XML 파싱 (`xml.etree`) | `parse_drugbank.py` |
| **ChEMBL** (API) | KD (결합력) | REST API 호출 | `fetch_chembl.py` |
| **PharmGKB** (Excel) | CYP2D6 빈도 | `pandas.read_excel()` | `parse_cyp2d6.py` |
| **FDA Labels** | Dose | 문헌 수동 검토 | - |

### 1.2 DrugBank 파싱 예시
```python
# XML에서 반감기 추출
for drug in root.findall('.//drug'):
    half_life = drug.find('.//half-life')
    if half_life is not None:
        # 정규표현식으로 숫자+단위 파싱
        match = re.search(r'(\d+\.?\d*)\s*(hours?|days?)', half_life.text)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            # 시간 단위로 통일
            if 'day' in unit:
                value *= 24
```

### 1.3 ChEMBL API 호출
```python
# KD 데이터 조회
url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
params = {
    'target_chembl_id': target_id,
    'standard_type': 'Kd',
    'limit': 100
}
response = requests.get(url, params=params)
```

---

## 2. 파라미터 범위 조정 (Range Refinement)

### 2.1 초기 범위 → 항체 특화 범위

| 파라미터 | 초기 범위 (DrugBank 전체) | 최종 범위 (항체 특화) | 조정 근거 |
|----------|---------------------------|----------------------|-----------|
| **KD** | 0.001-1000 nM | **0.01-20 nM** | FDA 승인 항체 20종 분석 |
| **Dose** | 0.001-100 mg/kg | **0.5-20 mg/kg** | 임상 투여 용량 범위 |
| **Half-life** | 0.1-10000 hours | **100-700 hours** | IgG 표준 반감기 (4-30일) |
| **MW** | 1-200 kDa | **140-160 kDa** | IgG 분자량 범위 |

### 2.2 조정 코드 (`batch_simulator.py`)
```python
@dataclass
class SimulationConfig:
    """항체 특화 파라미터 범위"""
    kd_range: Tuple[float, float] = (0.01, 20)      # nM
    dose_range: Tuple[float, float] = (0.5, 20)     # mg/kg
    halflife_range: Tuple[float, float] = (100, 700) # hours
    mw_range: Tuple[float, float] = (140, 160)      # kDa
    t0_range: Tuple[float, float] = (1, 100)        # nM (표적 농도)
```

---

## 3. 가상 데이터 생성 (Synthetic Data Generation)

### 3.1 생성 규모
- **총 샘플**: 50,000건
- **약물 수**: 1,000개
- **환자 수/약물**: 50명
- **인종**: 5개 (EUR, EAS, AFR, AMR, SAS)

### 3.2 샘플링 방식: Log-Uniform 분포

생물학적 파라미터는 여러 자릿수에 걸쳐 분포하므로 **로그 균등 분포**를 적용:

```python
def _log_uniform(self, low: float, high: float) -> float:
    """Log-uniform 샘플링 (지수 분포 효과)"""
    log_low = np.log(low)
    log_high = np.log(high)
    return np.exp(np.random.uniform(log_low, log_high))

# 사용 예시
kd = self._log_uniform(0.01, 20)      # 0.01~20 nM 사이 로그 균등
dose = self._log_uniform(0.5, 20)     # 0.5~20 mg/kg 사이 로그 균등
```

**장점**: 
- 0.01과 1 사이의 값, 1과 100 사이의 값이 비슷한 확률로 샘플링
- 생물학적 현실 반영 (KD가 0.1nM인 약물과 10nM인 약물이 동등하게 중요)

### 3.3 약물 후보 생성 코드
```python
def generate(self) -> DrugCandidate:
    """가상 약물 후보 생성"""
    return DrugCandidate(
        drug_id=f"DRUG_{self.drug_counter:05d}",
        KD=self._log_uniform(*self.config.kd_range),
        dose=self._log_uniform(*self.config.dose_range),
        charge=np.random.choice(self.config.charges or [-2, -1, 0, 1, 2]),
        MW=self._log_uniform(*self.config.mw_range),
        T0=self._log_uniform(*self.config.t0_range),
        half_life=self._log_uniform(*self.config.halflife_range)
    )
```

---

## 4. 노이즈 적용 (Biological Noise Injection)

### 4.1 노이즈 적용 목적
- **과적합 방지**: 완벽한 시뮬레이션 결과는 ML 모델이 "암기"할 수 있음
- **현실 반영**: 실제 생물학적 측정에는 10-30% 변동성 존재

### 4.2 노이즈 적용 방식 (CV 10-20%)
```python
def _generate_labels(self, sim_result: SimulationResult, drug: DrugCandidate):
    """시뮬레이션 결과에 생물학적 노이즈 적용"""
    
    # Coefficient of Variation (CV) = 10~20%
    noise_cv = 0.15  # 15%
    
    # 노이즈가 적용된 결과
    TO_noisy = sim_result['TO_trough'] * (1 + np.random.normal(0, noise_cv))
    Cmax_noisy = sim_result['C_max'] * (1 + np.random.normal(0, noise_cv))
    AUC_noisy = sim_result['AUC'] * (1 + np.random.normal(0, noise_cv))
    
    # 노이즈 적용된 값으로 라벨 결정
    TO_noisy = max(0, min(100, TO_noisy))  # Clamp to valid range
```

### 4.3 노이즈 분포
- **정규 분포**: `N(0, σ)` where σ = 0.15 (15% CV)
- **곱셈 노이즈**: 원본 값 × (1 + noise)

---

## 5. 약물유전체 기반 가상 코호트 생성 (Pharmacogenomics)

### 5.1 CYP2D6 유전자형 시뮬레이션

#### Step 1: 인종별 대립유전자 빈도 로딩
```python
# PharmGKB Excel에서 인종별 빈도 로딩
ALLELE_FREQUENCIES = {
    'EUR': {'*1': 0.35, '*2': 0.25, '*4': 0.20, '*10': 0.02, ...},
    'EAS': {'*1': 0.20, '*2': 0.10, '*4': 0.01, '*10': 0.45, ...},
    'AFR': {'*1': 0.30, '*2': 0.15, '*4': 0.05, '*17': 0.20, ...},
    ...
}
```

#### Step 2: 하디-바인베르크 평형 기반 유전자형 생성
```python
def generate_diplotype(population: str) -> Tuple[str, str]:
    """두 개의 대립유전자 랜덤 선택 (Hardy-Weinberg)"""
    alleles = list(ALLELE_FREQUENCIES[population].keys())
    probs = list(ALLELE_FREQUENCIES[population].values())
    
    # 두 번 독립적으로 샘플링 (부모 각각으로부터)
    allele1 = np.random.choice(alleles, p=probs)
    allele2 = np.random.choice(alleles, p=probs)
    
    return (allele1, allele2)  # e.g., ('*1', '*4')
```

#### Step 3: Activity Score 계산
```python
# 대립유전자별 활성도 (CPIC 가이드라인)
ACTIVITY_VALUES = {
    '*1': 1.0,   # Normal function
    '*2': 1.0,   # Normal function
    '*4': 0.0,   # No function
    '*10': 0.25, # Decreased function (EAS에서 흔함)
    '*17': 0.5,  # Decreased function (AFR에서 흔함)
    '*1xN': 2.0, # Gene duplication (UM)
    ...
}

def calculate_activity_score(diplotype: Tuple[str, str]) -> float:
    """두 대립유전자의 활성도 합산"""
    return ACTIVITY_VALUES.get(diplotype[0], 1.0) + \
           ACTIVITY_VALUES.get(diplotype[1], 1.0)
```

#### Step 4: 표현형 분류 (CPIC 2024 기준)
```python
def activity_to_phenotype(activity_score: float) -> str:
    """Activity Score → 표현형 변환 (CPIC 2024)"""
    if activity_score <= 0.25:
        return 'PM'   # Poor Metabolizer (저대사)
    elif activity_score <= 1.0:
        return 'IM'   # Intermediate Metabolizer (중간대사)
    elif activity_score <= 2.0:
        return 'NM'   # Normal Metabolizer (정상대사)
    else:
        return 'UM'   # Ultra-rapid Metabolizer (초고속대사)
```

#### Step 5: 청소율 승수 계산
```python
def activity_to_clearance_multiplier(activity_score: float) -> float:
    """Activity Score → 약물 청소율 영향"""
    # PM: 청소율 50% (독성 위험↑)
    # UM: 청소율 200% (약효 부족 위험)
    if activity_score <= 0.25:
        return 0.5   # PM
    elif activity_score <= 1.0:
        return 0.75  # IM
    elif activity_score <= 2.0:
        return 1.0   # NM (기준)
    else:
        return 1.5   # UM
```

### 5.2 가상 환자 생성 예시
```python
def generate_patient(population: str) -> VirtualPatient:
    """약물유전체 정보를 포함한 가상 환자 생성"""
    
    # 1. 유전자형 생성
    diplotype = generate_diplotype(population)
    
    # 2. 활성 점수 계산
    activity_score = calculate_activity_score(diplotype)
    
    # 3. 표현형 결정
    phenotype = activity_to_phenotype(activity_score)
    
    # 4. 청소율 영향 계산
    cl_multiplier = activity_to_clearance_multiplier(activity_score)
    
    return VirtualPatient(
        population=population,
        diplotype=diplotype,
        activity_score=activity_score,
        phenotype=phenotype,
        cl_multiplier=cl_multiplier
    )
```

---

## 6. 전체 파이프라인 요약

```
[DrugBank/ChEMBL/PharmGKB]
        │
        ▼
┌───────────────────────────────┐
│  1. 파라미터 추출 (XML/API)   │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  2. 항체 특화 범위 필터링     │
│     (MW 140-160, Dose 0.5-20) │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  3. Log-Uniform 샘플링        │
│     (1,000개 가상 약물 생성)  │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  4. CYP2D6 가상 코호트 생성   │
│     (5개 인종 × 10명 = 50명)  │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  5. mPBPK 시뮬레이션 실행     │
│     (ODE 풀이 → TO, Cmax, AUC)│
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  6. 생물학적 노이즈 적용      │
│     (CV 15% 정규분포 노이즈)  │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  7. 효능/독성 라벨 생성       │
│     (TO≥80% → Success)        │
└───────────────────────────────┘
        │
        ▼
    [50,000 샘플 CSV]
```

---

**관련 코드 파일**:
- `src/models/batch_simulator.py` - 전체 시뮬레이션 파이프라인
- `src/models/mpbpk_engine.py` - mPBPK ODE 모델
- `src/data/parse_cyp2d6.py` - CYP2D6 유전자형 시뮬레이션
