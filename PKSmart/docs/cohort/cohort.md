# Antigravity 이전 가이드 - 임상시험 대시보드

## 📁 핵심 파일 구조

### 1. 페이지 파일 (Pages)
```
pages/
├── Phase1.tsx    # 임상 1상 - 400mg 코호트 (PK, AE, PGx)
├── Phase2.tsx    # 임상 2상 - 300mg 코호트 (종양반응, 치료기간, 바이오마커)
├── Phase3.tsx    # 임상 3상 - 시험군 코호트 (생존분석, 타임라인, 바이오마커)
└── Overview.tsx  # 개요 페이지
```

### 2. 컴포넌트 파일 (Components)
```
components/
├── StatCard.tsx       # 통계 카드 (숫자 애니메이션, 상단 악센트 바)
├── ChartCard.tsx      # 차트 컨테이너 (제목, 설명, 페이드인 애니메이션)
├── PageHeader.tsx     # 페이지 헤더 (Phase 뱃지, 제목, 설명)
└── DashboardLayout.tsx # 사이드바 레이아웃
```

### 3. 스타일 파일
```
index.css  # 전역 스타일 (색상 변수, 애니메이션, 커스텀 클래스)
```

---

## 📊 차트 라이브러리

**Recharts** 사용 (React 기반 차트 라이브러리)
```bash
npm install recharts
```

사용된 차트 유형:
- `LineChart` - Spaghetti Plot (개별 환자 PK 프로파일)
- `BarChart` - Waterfall Plot (종양 반응), Swimmer Plot (치료 기간)
- `AreaChart` - Kaplan-Meier 생존 곡선

---

## 🎨 디자인 시스템

### 색상 팔레트 (OKLCH 형식)
```javascript
const chartColors = {
  teal: "oklch(0.65 0.15 180)",      // 주요 강조색 (1상, PR)
  amber: "oklch(0.75 0.15 75)",      // 보조 강조색 (2상, SD)
  indigo: "oklch(0.55 0.18 220)",    // 3차 강조색 (3상)
  rose: "oklch(0.60 0.18 25)",       // 경고색 (PD, DLT)
  green: "oklch(0.65 0.18 145)",     // 성공색 (CR)
  navy: "oklch(0.35 0.05 250)",      // 사이드바, 기준선
};
```

### 반응 유형별 색상
```javascript
const responseColors = {
  CR: chartColors.green,   // Complete Response
  PR: chartColors.teal,    // Partial Response
  SD: chartColors.amber,   // Stable Disease
  PD: chartColors.rose,    // Progressive Disease
};
```

---

## 📋 데이터 구조

### Phase 1 (임상 1상) 데이터
```typescript
// 개별 환자 PK 시계열 데이터
const individualPKData = [
  { time: 0, P01: 0, P02: 0, P03: 0, P04: 0, P05: 0, P06: 0 },
  { time: 1, P01: 580, P02: 620, P03: 545, P04: 690, P05: 510, P06: 605 },
  // ... 시간별 혈중 농도
];

// 개별 환자 PK 파라미터
const patientPKParams = [
  { patient: "P01", cmax: 720, tmax: 2, auc: 8450, t12: 17.2, cl: 47.3 },
  // ...
];

// 개별 환자 이상반응
const patientAEData = [
  { patient: "P01", fatigue: 1, nausea: 0, headache: 1, diarrhea: 0, rash: 0, dlt: false },
  // ...
];

// 개별 환자 약물유전체
const patientPGxData = [
  { patient: "P01", cyp2d6: "EM", cyp3a4: "*1/*1", auc: 8450, cmax: 720 },
  // ...
];
```

### Phase 2 (임상 2상) 데이터
```typescript
// 종양 반응 데이터 (Waterfall Plot)
const patientResponseData = [
  { patient: "P01", change: -72, bestResponse: "PR", biomarker: "KRAS WT" },
  // ...
];

// 치료 기간 데이터 (Swimmer Plot)
const patientDurationData = [
  { patient: "P01", duration: 18, response: "PR", ongoing: true },
  // ...
];

// 바이오마커 데이터
const patientBiomarkerData = [
  { patient: "P01", kras: "WT", egfr: "+", pdl1: 85, response: "PR" },
  // ...
];
```

### Phase 3 (임상 3상) 데이터
```typescript
// 생존 데이터
const patientSurvivalData = [
  { patient: "P01", os: 32, pfs: 18, status: "alive", biomarker: "+" },
  // ...
];

// Kaplan-Meier 곡선 데이터
const kmData = [
  { month: 0, survival: 100, atRisk: 15 },
  { month: 6, survival: 93, atRisk: 14 },
  // ...
];

// 환자 타임라인 데이터
const patientTimelineData = [
  { patient: "P01", enrollment: 0, firstResponse: 3, progression: null, death: null, lastFollowup: 32 },
  // ...
];
```

---

## 🔧 Antigravity 프롬프트

아래 프롬프트를 Antigravity에 복사해서 사용하세요:

---

### 프롬프트 (한국어)

```
임상시험 코호트 결과 대시보드를 만들어줘. 각 페이지는 1개의 특정 코호트의 개별 환자 데이터를 보여줘야 해.

## 기술 스택
- React + TypeScript
- Recharts (차트 라이브러리)
- Tailwind CSS

## 페이지 구조
1. **임상 1상 페이지** (400mg 코호트, N=6)
   - 탭: 약동학(PK), 안전성(AE), 약물유전체
   - PK 탭: Spaghetti Plot (개별 환자 혈중농도-시간 곡선), PK 파라미터 테이블 (Cmax, Tmax, AUC, t1/2, CL/F)
   - AE 탭: 이상반응 매트릭스 (Grade 0-3 색상 표시), DLT 여부
   - PGx 탭: CYP2D6 표현형 (EM/IM/PM), CYP3A4 유전자형

2. **임상 2상 페이지** (300mg 코호트, N=12)
   - 탭: 종양 반응, 치료 기간, 바이오마커
   - 종양 반응 탭: Waterfall Plot (종양 크기 변화율 %), PR/SD/PD 기준선 표시
   - 치료 기간 탭: Swimmer Plot (가로 막대 차트)
   - 바이오마커 탭: KRAS, EGFR, PD-L1 상태별 반응 분석

3. **임상 3상 페이지** (시험군 코호트, N=15)
   - 탭: 생존 분석, 환자 타임라인, 바이오마커
   - 생존 분석 탭: Kaplan-Meier 곡선 (AreaChart, stepAfter), Number at Risk 테이블
   - 타임라인 탭: 개별 환자 치료 기간 (가로 막대)
   - 바이오마커 탭: 양성/음성 하위그룹 비교

## 디자인 요구사항
- 색상: Teal(#0d9488), Amber(#f59e0b), Indigo(#6366f1), Rose(#f43f5e)
- 각 페이지 상단에 4개의 요약 통계 카드 (숫자 카운터 애니메이션)
- 카드와 차트에 호버 효과 및 페이드인 애니메이션
- 반응형 디자인 (모바일/데스크톱)
- 테이블에 환자별 색상 인디케이터

## 데이터
- 더미 데이터로 시작하되, 나중에 실제 데이터로 교체 가능하도록 데이터 구조를 파일 상단에 분리해서 정의
```

---

### 프롬프트 (영어)

```
Create a clinical trial cohort results dashboard. Each page should display individual patient data for a single specific cohort.

## Tech Stack
- React + TypeScript
- Recharts (charting library)
- Tailwind CSS

## Page Structure
1. **Phase 1 Page** (400mg cohort, N=6)
   - Tabs: Pharmacokinetics (PK), Safety (AE), Pharmacogenomics
   - PK Tab: Spaghetti Plot (individual patient concentration-time curves), PK parameters table (Cmax, Tmax, AUC, t1/2, CL/F)
   - AE Tab: Adverse event matrix (Grade 0-3 color coded), DLT status
   - PGx Tab: CYP2D6 phenotype (EM/IM/PM), CYP3A4 genotype

2. **Phase 2 Page** (300mg cohort, N=12)
   - Tabs: Tumor Response, Treatment Duration, Biomarkers
   - Response Tab: Waterfall Plot (tumor size change %), PR/SD/PD reference lines
   - Duration Tab: Swimmer Plot (horizontal bar chart)
   - Biomarker Tab: KRAS, EGFR, PD-L1 status vs response analysis

3. **Phase 3 Page** (Treatment arm cohort, N=15)
   - Tabs: Survival Analysis, Patient Timeline, Biomarkers
   - Survival Tab: Kaplan-Meier curve (AreaChart, stepAfter), Number at Risk table
   - Timeline Tab: Individual patient treatment duration (horizontal bars)
   - Biomarker Tab: Positive/Negative subgroup comparison

## Design Requirements
- Colors: Teal(#0d9488), Amber(#f59e0b), Indigo(#6366f1), Rose(#f43f5e)
- 4 summary stat cards at top of each page (animated number counter)
- Hover effects and fade-in animations on cards and charts
- Responsive design (mobile/desktop)
- Color indicators per patient in tables

## Data
- Start with dummy data, but define data structures at the top of files for easy replacement with real data later
```

---

## 📎 첨부 파일 목록

Antigravity에 참고용으로 첨부할 수 있는 파일들:

| 파일명 | 설명 | 용도 |
|--------|------|------|
| `Phase1.tsx` | 임상 1상 페이지 전체 코드 | 데이터 구조, 차트 구현, 테이블 스타일 참고 |
| `Phase2.tsx` | 임상 2상 페이지 전체 코드 | Waterfall/Swimmer Plot 구현 참고 |
| `Phase3.tsx` | 임상 3상 페이지 전체 코드 | Kaplan-Meier 곡선 구현 참고 |
| `StatCard.tsx` | 통계 카드 컴포넌트 | 숫자 애니메이션 구현 참고 |
| `ChartCard.tsx` | 차트 컨테이너 컴포넌트 | 페이드인 애니메이션 참고 |
| `index.css` | 전역 스타일 | 색상 변수, 애니메이션 정의 참고 |

---

## 💡 팁

1. **데이터 분리**: 각 페이지 파일 상단에 데이터 배열이 정의되어 있어서, 나중에 API 연동 시 해당 부분만 교체하면 됩니다.

2. **차트 커스터마이징**: Recharts의 `Tooltip`, `Legend`, `ReferenceLine` 등을 활용해 임상시험 특화 시각화를 구현했습니다.

3. **색상 일관성**: `responseColors` 객체를 사용해 PR/SD/PD 등 반응 유형별 색상을 일관되게 적용했습니다.

4. **애니메이션**: `delay` prop을 통해 카드와 차트가 순차적으로 나타나는 스태거 애니메이션을 구현했습니다.
