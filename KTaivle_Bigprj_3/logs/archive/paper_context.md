# Reference Paper Context: Sanofi Antibody mPBPK/ML Framework (2025)

이 문서는 Antigravity 에이전트가 '항체 신약 개발을 위한 임상 디지털 트윈' 프로젝트를 수행하기 위해 필요한 핵심 논문(Scientific Reports, 2025)의 방법론과 로직을 정의한다.

## 1. 논문 개요
- [cite_start]**제목:** Development of an mPBPK machine learning framework for early target pharmacology assessment of biotherapeutics [cite: 3]
- **저자:** Patidar et al. (Sanofi Global DMPK Modeling & Simulation) [cite_start][cite: 4, 24]
- [cite_start]**목적:** 초기 단계의 항체 후보 물질(Lead Candidates) 선정을 위해, 가상 데이터 생성과 mPBPK 모델링, 머신러닝(Decision Tree)을 결합한 평가 프레임워크 제안[cite: 8].

## 2. 모델 구조: Minimal PBPK (mPBPK)
[cite_start]이 논문에서 사용된 생리학적 약물동태 모델은 항체(Antibody)의 거동을 모사하기 위해 다음과 같은 구획(Compartment) 구조를 가진다[cite: 53].

### 2.1. 구획 (Compartments)
1.  **Plasma (혈장):** 약물이 주입되는 중앙 구획.
2.  **Tight Tissue (조밀 조직):** 혈관 투과성이 낮은 조직 (예: 근육, 피부).
3.  **Leaky Tissue (느슨한 조직):** 혈관 투과성이 높은 조직 (예: 간, 비장 등 포함).
4.  **Lymph (림프):** 조직에서 혈장으로 약물을 회수(Recycling)하는 경로.

### 2.2. 주요 이동 기전
- **혈장 $\leftrightarrow$ 조직:** 대류(Convection)와 확산(Diffusion)에 의해 이동.
- [cite_start]**조직 $\to$ 림프 $\to$ 혈장:** 항체는 크기가 커서 혈관으로 직접 재흡수되지 않고, 림프계를 통해 순환함[cite: 53].
- **소실 (Clearance):**
    - 전신 소실(Linear Clearance): 신장 등에서의 배설.
    - 타겟 매개 소실(TMDD): 타겟과 결합하여 사라짐.

## 3. 데이터 생성 및 시뮬레이션 프로세스 (Workflow)

### [cite_start]Step 1: 가상 데이터 생성 (Virtual Data Generation) [cite: 74]
- [cite_start]**규모:** 10,000개의 가상 항체-타겟 후보 쌍(Virtual Candidates) 생성[cite: 77].
- **샘플링 방법:** Log-uniform sampling 사용.
- **주요 입력 변수 (Parameters):**
    - **Drug Properties:**
        - [cite_start]$K_D$ (Binding Affinity): 1 pM ~ 1000 nM[cite: 80].
        - [cite_start]Charge (Surface Charge): +5, 0, -5 (전하에 따라 청소율 보정)[cite: 63, 65].
        - [cite_start]Dosing: 0.1, 1, 10 mg/kg (Bolus or Q2W, Q4W)[cite: 63].
    - **Target Properties:**
        - [cite_start]Baseline ($T_0$): 1 pM ~ 1000 nM[cite: 79].
        - [cite_start]Half-life ($t_{1/2}$): 1 min ~ 300 h[cite: 79].

### [cite_start]Step 2: mPBPK 시뮬레이션 [cite: 51]
- 각 가상 후보에 대해 mPBPK 모델을 실행하여 약동학 프로파일(PK Profile)을 예측.
- **핵심 지표 (Output Endpoint):** Target Occupancy (TO%)
    - [cite_start]$TO\% = \frac{\text{Bound Target}}{\text{Total Target}} \times 100$[cite: 57].
    - [cite_start]성공 기준: 최저 농도($C_{min}$)에서의 TO%가 **90% 이상**일 때 "Optimal(성공)"로 분류[cite: 83].

### [cite_start]Step 3: 머신러닝 분류 (ML Classification) [cite: 87]
- [cite_start]**알고리즘:** Decision Tree Classifier (설명 가능성을 위해 선택됨)[cite: 93].
- **학습:** 입력 변수($K_D$, Charge, $T_0$ 등)를 통해 성공(TO > 90%) 여부를 예측하는 규칙(Rule) 학습.
- [cite_start]**데이터 불균형 처리:** SMOTE 기법을 사용하여 데이터 밸런싱 수행[cite: 85].

## 4. 주요 연구 결과 (Key Findings & Rules)

에이전트는 시뮬레이션 결과 해석 시 다음 규칙을 참고해야 한다.

1.  **전하(Charge)의 영향:**
    - [cite_start]양전하(+5) 항체는 세포막과의 비특이적 결합으로 인해 **청소율(Clearance)이 증가**하고 반감기가 짧아짐[cite: 36, 353].
    - [cite_start]음전하(-5) 항체는 상대적으로 반감기가 길어짐[cite: 356].
    - [cite_start]따라서 양전하 항체는 더 강력한 결합력(낮은 $K_D$)이 요구됨[cite: 353].

2.  **타겟 형태(Soluble vs Membrane)의 영향:**
    - [cite_start]타겟이 가용성(Soluble)인지 세포막 결합형(Membrane-bound)인지에 따라 최적의 $K_D$와 반감기 조건이 다름[cite: 364].
    - [cite_start]가용성 타겟의 경우 Baseline 농도가 7 nM 이하일 때 성공 확률이 높음[cite: 403].

3.  **용량 및 투여 주기:**
    - [cite_start]용량이 낮을수록($0.1 \to 1 \to 10$ mg/kg), 또는 투여 간격이 길수록($Q1W \to Q2W \to Q4W$) 더 엄격한(낮은) $K_D$ 값이 요구됨[cite: 194, 276].

## 5. 프로젝트 적용 가이드 (Application)
이 프로젝트는 위 논문의 4-Compartment 모델을 기반으로 하되, **간(Liver)을 별도로 분리한 5-Compartment 모델**로 확장하여 구현한다.

- **기본 수식:** 논문의 mPBPK 미분방정식을 따름.
- **확장 사항:** Leaky Tissue에서 Liver를 분리하여 간 대사($CL_{liv}$) 및 간 혈류($Q_{liv}$) 변수를 명시적으로 다룸.