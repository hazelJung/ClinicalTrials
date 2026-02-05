# Antigravity 마이그레이션 가이드: FDA 임상시험 시뮬레이션 대시보드

## 1. 핵심 페이지 파일 구조

### 1.1 Phase 1: Safety & PK/PD (`Phase1.tsx`)
**파일 경로**: `docs/pages/Phase1.tsx` (570줄)

**뼈대 디자인**:
- **Header**: 아이콘 + 제목 + 설명 (Framer Motion 애니메이션)
- **KPI 섹션**: 4개의 카드 (MTD, DLT Rate, 시험 대상자, t1/2)
- **Tabs 구조**: PK Profile | Safety | Pharmacogenomics
  - **PK Profile Tab**: 
    - Spaghetti Plot (개별 환자별 약물 혈중 농도 변화)
    - 용량별 AUC 분포 (Bar Chart)
    - 약동학 파라미터 요약 (Table)
  - **Safety Tab**:
    - DLT 발생 현황 (Bar Chart)
    - 이상반응 발생 현황 (Stacked Bar Chart)
    - 이상반응 요약 (Table)
  - **Pharmacogenomics Tab**:
    - CYP2C19 유전자형별 PK 비교 (Scatter Chart)
    - 유전자형별 AUC 분포 (Bar Chart)
    - 유전자형별 PK 파라미터 (Table)

**JavaScript/React 핵심 로직**:
```typescript
// 1. 데이터 구조
const pkProfileData = [
  { time: 0, p1: 0, p2: 0, ..., mean: 0 },
  { time: 0.5, p1: 45, p2: 52, ..., mean: 47.6 },
  // ... 9개 시간점 데이터
];

const aeHeatmapData = [
  { soc: "위장관계", dose50: 2, dose100: 4, dose150: 6, dose200: 8 },
  // ... 5개 기관계 데이터
];

const dltData = [
  { dose: "50mg", subjects: 3, dlt: 0, rate: 0 },
  // ... 5개 용량 단계
];

const genotypeData = [
  { genotype: "EM", auc: 1850, cmax: 195, subjects: 12 },
  // ... 3개 유전자형
];

// 2. 차트 라이브러리: Recharts 사용
// - LineChart (Spaghetti Plot)
// - BarChart (용량별 AUC, DLT, AE)
// - ScatterChart (유전자형 비교)
// - 모든 차트에 CartesianGrid, XAxis, YAxis, Tooltip, Legend 포함

// 3. 스타일링: Tailwind CSS + Framer Motion
// - motion.div로 애니메이션 (opacity, y 변환)
// - bg-card/50 backdrop-blur-sm으로 유리 모양 효과
// - text-cyan-400, text-amber-400 등 색상 강조

// 4. 컴포넌트 구성
// - DashboardLayout (네비게이션 포함)
// - KPICard (4개 메트릭 표시)
// - Tabs (3개 탭 전환)
// - Card (각 섹션 래퍼)
```

---

### 1.2 Phase 2: PoC & Dose Finding (`Phase2.tsx`)
**파일 경로**: `/client/src/pages/Phase2.tsx` (552줄)

**뼈대 디자인**:
- **Header**: 아이콘 + 제목 + 설명
- **KPI 섹션**: 4개의 카드 (ORR, DCR, 시험 대상자, RP2D)
- **Tabs 구조**: Efficacy | Dose-Response | Biomarker Analysis
  - **Efficacy Tab**:
    - Waterfall Plot (종양 크기 변화, 환자별 색상 구분)
    - Response Summary (3개 카드: ORR, DCR, 기타)
    - Swimmer Plot (환자별 치료 기간 및 반응)
  - **Dose-Response Tab**:
    - Dose-Response 곡선 (Composed Chart)
    - 용량별 효능/안전성 비교 (Bar Chart)
  - **Biomarker Analysis Tab**:
    - Biomarker+ vs Biomarker- 반응률 비교 (Bar Chart)
    - ORR/DCR 비교 (Grouped Bar Chart)

**JavaScript/React 핵심 로직**:
```typescript
// 1. 데이터 구조
const waterfallData = [
  { patient: "P01", change: -72, biomarker: "+" },
  // ... 24개 환자 데이터 (change 기준 정렬)
];

const doseResponseData = [
  { dose: 50, efficacy: 15, safety: 5 },
  // ... 6개 용량 단계
];

const swimmerData = [
  { patient: "P01", start: 0, duration: 24, response: 4, ongoing: true },
  // ... 8개 환자 데이터
];

const biomarkerComparisonData = [
  { group: "Biomarker+", orr: 62, dcr: 88, n: 42 },
  // ... 3개 그룹 (Biomarker+, -, Overall)
];

// 2. 차트 라이브러리
// - BarChart (Waterfall Plot - layout="vertical")
// - ComposedChart (Dose-Response - Line + Area)
// - Cell로 개별 바 색상 지정 (biomarker 상태에 따라)

// 3. 상호작용 요소
// - ReferenceLine (PR: -30%, PD: +20% 기준선)
// - 범례 및 설명 텍스트 (RECIST 기준)

// 4. 색상 코드
// - Biomarker+: #84CC16 (lime)
// - Biomarker-: #64748B (gray)
// - PR: #84CC16, SD: #06B6D4, PD: #DC2626
```

---

### 1.3 Phase 3: Confirmatory Trial (`Phase3.tsx`)
**파일 경로**: `/client/src/pages/Phase3.tsx` (557줄)

**뼈대 디자인**:
- **Header**: 아이콘 + 제목 + 설명
- **KPI 섹션**: 4개의 카드 (HR, p-value, 시험 대상자, PFS 중앙값)
- **Tabs 구조**: Survival Analysis | Subgroup Analysis | Precision Medicine
  - **Survival Analysis Tab**:
    - Kaplan-Meier 생존 곡선 (PFS) (Line Chart - stepAfter)
    - PFS 및 OS 중앙값 비교 (2개 카드)
  - **Subgroup Analysis Tab**:
    - Forest Plot (HR 및 신뢰 구간) (Custom Bar Chart)
    - 하위 그룹별 위험비 (9개 그룹)
  - **Precision Medicine Tab**:
    - 유전자형별 생존 곡선 (Multiple Line Chart)
    - 유전자형별 반응률 (Grouped Bar Chart)
    - CYP2C19 유전자형에 따른 효과 차이 시각화

**JavaScript/React 핵심 로직**:
```typescript
// 1. 데이터 구조
const kmData = [
  { month: 0, treatment: 100, control: 100 },
  // ... 13개 시간점 (0-36개월)
];

const forestPlotData = [
  { subgroup: "전체 인구", hr: 0.68, lower: 0.55, upper: 0.84, n: 420, favors: "treatment" },
  // ... 9개 하위 그룹 (연령, 성별, Biomarker, ECOG)
];

const pfsData = [
  { group: "치료군", median: 12.4, lower: 10.8, upper: 14.2 },
  // ... 2개 그룹
];

const genotypeSubgroupData = [
  { genotype: "전체", hr: 0.68, pfs: 12.4, os: 24.8, n: 420 },
  // ... 4개 유전자형 (EM, IM, PM)
];

// 2. 차트 라이브러리
// - LineChart (Kaplan-Meier - stepAfter 타입)
// - BarChart (Forest Plot 표현 - 수평 레이아웃)
// - ReferenceLine (y=50% 기준선, x=1.0 HR 기준선)

// 3. 통계 표시
// - HR (위험비) + 95% CI (신뢰 구간)
// - p-value 표시
// - 중앙값 및 95% CI 범위

// 4. 색상 코드
// - 치료군: #A855F7 (purple)
// - 대조군: #64748B (gray, dashed line)
// - 유의미한 효과: #84CC16 (lime)
// - 중립: #64748B (gray)
```

---

## 2. 공통 컴포넌트 및 레이아웃

### 2.1 DashboardLayout (`DashboardLayout.tsx`)
**역할**: 모든 페이지의 상단 네비게이션 및 레이아웃 제공

**구조**:
```typescript
// 네비게이션 항목
const navItems = [
  { path: "/", label: "개요", icon: Activity },
  { path: "/phase1", label: "Phase 1", icon: FlaskConical, subtitle: "Safety & PK" },
  { path: "/phase2", label: "Phase 2", icon: TestTube2, subtitle: "PoC & Dose" },
  { path: "/phase3", label: "Phase 3", icon: BarChart3, subtitle: "Confirmatory" },
  { path: "/cohort-settings", label: "Cohort", icon: Settings2, subtitle: "Advanced" },
];

// 기능
// - 데스크톱: 수평 네비게이션 (md 이상)
// - 모바일: 토글 메뉴
// - 활성 탭 표시 (밑줄 애니메이션)
// - Framer Motion으로 부드러운 전환
```

### 2.2 KPICard (`KPICard.tsx`)
**역할**: 각 페이지 상단의 주요 지표 카드

**구조**:
```typescript
interface KPICardProps {
  title: string;
  value: string | number;
  unit?: string;
  change?: number;
  changeLabel?: string;
  icon: React.ComponentType;
  color: "cyan" | "amber" | "lime" | "purple";
  delay?: number;
}

// 기능
// - 아이콘 + 제목 + 값 + 단위 표시
// - 선택적 변화율 표시 (vs 예상, vs 대조군 등)
// - 색상 테마 (cyan, amber, lime, purple)
// - Framer Motion 애니메이션 (delay 기반)
```

---

## 3. 스타일 및 디자인 시스템

### 3.1 색상 팔레트
```css
/* Primary Colors */
--primary: #06B6D4 (Cyan - 주요 강조)
--lime: #84CC16 (Lime - 긍정적 결과)
--purple: #A855F7 (Purple - 치료군)
--amber: #F59E0B (Amber - 경고/주의)
--red: #DC2626 (Red - 부정적 결과)


### 3.2 차트 스타일링
```typescript
// 모든 차트 공통 설정
const chartConfig = {
  grid: { strokeDasharray: "3 3", stroke: "#334155" },
  axis: { stroke: "#64748B", fontSize: 12 },
  tooltip: { 
    backgroundColor: "#1E293B", 
    border: "1px solid #334155",
    borderRadius: "8px"
  },
  colors: ["#06B6D4", "#84CC16", "#A855F7", "#F59E0B", "#EC4899"]
};
```

### 3.3 애니메이션
```typescript
// Framer Motion 기본 설정
const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5, delay: 0 }
};

// 카드 호버 효과
whileHover={{ scale: 1.02 }}
whileTap={{ scale: 0.98 }}
```

---

## 4. Antigravity 마이그레이션 프롬프트

### 프롬프트 템플릿

```
# Antigravity 마이그레이션: FDA 임상시험 시뮬레이션 대시보드

## 목표
기존 React/Tailwind 기반 임상시험 대시보드(Manus)를 Antigravity 플랫폼으로 마이그레이션합니다.

## 참고 파일 및 구조

### 1. Phase 1 페이지 (안전성 및 약동학 분석)
**원본 파일**: Phase1.tsx (570줄)

**뼈대 디자인**:
- 헤더: 아이콘(FlaskConical) + "Phase 1: Safety & PK/PD" 제목
- KPI 섹션: 4개 메트릭 카드 (MTD, DLT Rate, 시험 대상자, t1/2)
- 3개 탭 (PK Profile | Safety | Pharmacogenomics)

**각 탭의 차트 구성**:
1. **PK Profile Tab**:
   - Spaghetti Plot (LineChart): 개별 환자 5명 + 평균선 (9개 시간점)
   - 용량별 AUC 분포 (BarChart): 4개 용량 단계
   - 파라미터 테이블: Cmax, AUC0-∞, Tmax, t1/2

2. **Safety Tab**:
   - DLT 발생 (BarChart): 용량별 대상자 수 vs DLT 발생
   - 이상반응 (Stacked BarChart): 5개 기관계 × 4개 용량
   - 이상반응 요약 테이블: CTCAE Grade별 분류

3. **Pharmacogenomics Tab**:
   - 유전자형 비교 (ScatterChart): EM/IM/PM 3개 그룹
   - 유전자형별 AUC (BarChart): 3개 유전자형
   - 파라미터 테이블: 유전자형별 PK 값

**데이터 구조**:
- pkProfileData: 9개 시간점 × 5명 환자 + 평균
- aeHeatmapData: 5개 기관계 × 4개 용량
- dltData: 5개 용량 단계
- genotypeData: 3개 유전자형

**차트 라이브러리**: Recharts
- LineChart, BarChart, ScatterChart 사용
- 모든 차트에 CartesianGrid, XAxis, YAxis, Tooltip, Legend 포함
- 색상: #06B6D4(cyan), #84CC16(lime), #A855F7(purple), #F59E0B(amber), #DC2626(red)

---

### 2. Phase 2 페이지 (개념 증명 및 용량 탐색)
**원본 파일**: Phase2.tsx (552줄)

**뼈대 디자인**:
- 헤더: 아이콘(TestTube2) + "Phase 2: PoC & Dose Finding" 제목
- KPI 섹션: 4개 메트릭 카드 (ORR, DCR, 시험 대상자, RP2D)
- 3개 탭 (Efficacy | Dose-Response | Biomarker Analysis)

**각 탭의 차트 구성**:
1. **Efficacy Tab**:
   - Waterfall Plot (BarChart, layout="vertical"): 24명 환자, biomarker 상태별 색상
   - Response Summary: 3개 카드 (ORR 42%, DCR 78%, 기타)
   - Swimmer Plot: 8명 환자별 치료 기간 및 반응

2. **Dose-Response Tab**:
   - Dose-Response 곡선 (ComposedChart): 효능 vs 안전성
   - 용량별 비교 (BarChart): 6개 용량 단계

3. **Biomarker Analysis Tab**:
   - Biomarker 반응률 비교 (BarChart): Biomarker+/- vs Overall
   - ORR/DCR 비교 (Grouped BarChart)

**데이터 구조**:
- waterfallData: 24명 환자 (change 기준 정렬)
- doseResponseData: 6개 용량 단계
- swimmerData: 8명 환자
- biomarkerComparisonData: 3개 그룹

**차트 라이브러리**: Recharts
- BarChart (layout="vertical"), ComposedChart, Cell로 개별 색상 지정
- ReferenceLine으로 PR(-30%), PD(+20%) 기준선 표시
- 색상: Biomarker+ #84CC16, Biomarker- #64748B

---

### 3. Phase 3 페이지 (확증 임상시험 및 통계 분석)
**원본 파일**: Phase3.tsx (557줄)

**뼈대 디자인**:
- 헤더: 아이콘(BarChart3) + "Phase 3: Confirmatory Trial" 제목
- KPI 섹션: 4개 메트릭 카드 (HR, p-value, 시험 대상자, PFS 중앙값)
- 3개 탭 (Survival Analysis | Subgroup Analysis | Precision Medicine)

**각 탭의 차트 구성**:
1. **Survival Analysis Tab**:
   - Kaplan-Meier 곡선 (LineChart, type="stepAfter"): 치료군 vs 대조군 (13개 시간점)
   - PFS 중앙값 카드: 치료군 12.4개월 vs 대조군 7.2개월
   - OS 중앙값 카드: 치료군 24.8개월 vs 대조군 18.2개월

2. **Subgroup Analysis Tab**:
   - Forest Plot (BarChart, 수평): 9개 하위 그룹 (연령, 성별, Biomarker, ECOG)
   - HR + 95% CI 표시
   - 유의미한 효과 vs 중립 색상 구분

3. **Precision Medicine Tab**:
   - 유전자형별 생존 곡선 (Multiple LineChart): EM/IM/PM
   - 유전자형별 반응률 (BarChart): ORR, DCR, CR, PR
   - CYP2C19 유전자형에 따른 효과 차이

**데이터 구조**:
- kmData: 13개 시간점 (0-36개월)
- forestPlotData: 9개 하위 그룹
- pfsData, osData: 2개 그룹
- genotypeSubgroupData: 4개 유전자형

**차트 라이브러리**: Recharts
- LineChart (stepAfter), BarChart (수평 레이아웃)
- ReferenceLine으로 50% 생존율, HR=1.0 기준선 표시
- 색상: 치료군 #A855F7, 대조군 #64748B

---

## 공통 컴포넌트

### DashboardLayout
- 상단 네비게이션 (5개 항목: 개요, Phase 1/2/3, Cohort Settings)
- 데스크톱: 수평 네비게이션, 모바일: 토글 메뉴
- 활성 탭 표시 (밑줄 애니메이션)

### KPICard
- 아이콘 + 제목 + 값 + 단위 + 선택적 변화율
- 색상 테마 (cyan, amber, lime, purple)
- Framer Motion 애니메이션

---

## 스타일 가이드

### 색상
- Primary: #06B6D4 (Cyan)
- Success: #84CC16 (Lime)
- Treatment: #A855F7 (Purple)
- Warning: #F59E0B (Amber)
- Error: #DC2626 (Red)
- Background: Dark mode (oklch 기반)

### 차트 공통 설정
- Grid: strokeDasharray="3 3", stroke="#334155"
- Axis: stroke="#64748B", fontSize=12
- Tooltip: backgroundColor="#1E293B", border="1px solid #334155"

### 애니메이션
- Fade-in-up: opacity 0→1, y 20→0, duration 0.5s
- 카드 호버: scale 1→1.02
- 탭 전환: spring animation

---

## 마이그레이션 요청사항

1. 위의 3개 페이지(Phase 1/2/3)를 Antigravity 플랫폼에 맞게 재구현해주세요.
2. 각 페이지의 차트, 테이블, 카드 레이아웃을 동일하게 유지해주세요.
3. 데이터는 현재 하드코딩된 형태를 유지하되, 향후 API 연동이 가능하도록 구조화해주세요.
4. 다크 모드 스타일 및 Framer Motion 애니메이션을 가능한 한 유지해주세요.
5. 반응형 디자인(모바일/태블릿/데스크톱)을 적용해주세요.
```

---

## 5. 추가 참고사항

### 5.1 라이브러리 의존성
- **Recharts**: 모든 차트 (LineChart, BarChart, ScatterChart, ComposedChart)
- **Framer Motion**: 애니메이션 (motion.div, whileHover, whileTap)
- **Lucide React**: 아이콘 (FlaskConical, TestTube2, BarChart3, Dna 등)
- **Tailwind CSS**: 스타일링

### 5.2 성능 최적화 팁
- 차트 데이터는 `useMemo`로 메모이제이션
- 탭 전환 시 불필요한 리렌더링 방지
- 이미지 최적화 (CDN 사용)

### 5.3 접근성 고려사항
- 모든 차트에 `aria-label` 추가
- 색상만으로 정보 전달하지 않기 (패턴, 텍스트 추가)
- 키보드 네비게이션 지원

---

## 6. 파일 다운로드 링크

원본 파일들은 다음 경로에서 확인 가능합니다:
- Phase1.tsx: `docs/pages/Phase1.tsx`
- Phase2.tsx: `docs/pages/Phase2.tsx`
- Phase3.tsx: `docs/pages/Phase3.tsx`
- DashboardLayout.tsx: `docs/components/DashboardLayout.tsx`
- KPICard.tsx: `/docs/components/KPICard.tsx`
```

---

## 사용 방법

위의 프롬프트를 Antigravity에 제공할 때:

1. **전체 프롬프트 복사**: 위의 "Antigravity 마이그레이션 프롬프트" 섹션 전체를 복사
2. **Antigravity 입력**: Antigravity 플랫폼의 프롬프트 입력 창에 붙여넣기
3. **파일 첨부** (선택사항): 원본 파일들을 첨부하면 더 정확한 마이그레이션 가능
4. **실행**: 프롬프트 실행 및 결과 검토

---

## 추가 질문 시 확인사항

- **차트 상호작용**: 클릭, 호버, 드래그 기능 필요 여부
- **데이터 업데이트**: 실시간 데이터 연동 필요 여부
- **내보내기 기능**: PDF, CSV 내보내기 필요 여부
- **다국어 지원**: 영어/한국어 이중 언어 필요 여부
