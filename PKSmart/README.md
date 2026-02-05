# 🏥 PKSmart: AI-Powered Clinical Trial Design Platform

PharmaTwin의 AI 기반 임상시험 설계 및 독성 예측 플랫폼입니다.
본 문서는 프로젝트 구조와 **IND Generator 개발**을 위한 가이드라인을 포함합니다.

---

## 📂 1. Project Structure

주요 디렉토리 및 파일 구조에 대한 설명입니다.

```
PKSmart/
├── app/                        # Web Application (FastAPI)
│   ├── main.py                 # 앱 진입점 (FastAPI App 생성)
│   ├── database.py             # DB 연결 설정
│   ├── models.py               # DB 모델 (Table 정의)
│   ├── routers/                # API 라우터 (기능별 분리)
│   │   ├── auth.py             # 로그인/회원가입
│   │   ├── projects.py         # 프로젝트 및 코호트 결과 관리
│   │   ├── cohorts.py          # 코호트 생성/관리
│   │   ├── dashboard.py        # 대시보드, 사용자 관리
│   │   ├── ind_agent.py        # ⭐ IND Generator 핵심 로직
│   │   └── analysis.py         # 분석 기능
│   ├── static/                 # CSS, JS, Images
│   └── templates/              # HTML 템플릿 (Jinja2)
│       ├── base.html           # [Root] 공통 헤더/스크립트
│       ├── app_base.html       # [Dashboard Layout] 사이드바+헤더
│       ├── auth_base.html      # [Auth Layout] 중앙 카드 배치
│       ├── dashboard.html      # [Main] 대시보드
│       ├── ind_generator.html  # ⭐ [Target] IND Generator 입력 화면
│       ├── ind_report_detail.html # IND Report 상세 페이지
│       ├── project_results.html   # 프로젝트 결과 (Phase 1/2/3)
│       ├── project_detail.html    # 프로젝트 상세
│       ├── cohort_detail.html     # 코호트 상세
│       └── create_cohort.html     # 코호트 생성
├── pksmart/                    # AI Core Logic (Model, Training, Prediction)
├── models/                     # 학습된 ML 모델 파일 (.joblib)
├── docs/                       # 개발 문서
└── requirements.txt            # 의존성 패키지 목록
```

---

## 🛠️ 2. IND Generator 개발 가이드 (현재 진행중)

### 2.1 현재 진행 상황 ✅

IND Generator 페이지 (`ind_generator.html`)의 입력 폼 구조 변경 작업이 **진행 중**입니다.

**완료된 작업:**
- [x] Clinical Trial Parameters 섹션: 기본 접힘 상태로 변경
- [x] "Auto-populated from project results" 라벨 추가
- [x] Drug Candidate Info 섹션: "Auto-populated from project data" 라벨 추가
- [x] PK & Safety Data 섹션: "Auto-populated from prediction models" 라벨 추가
- [x] 모든 자동 채움 필드에 회색 텍스트 스타일 (`text-gray-500`) 적용
- [x] 기본값 설정 (placeholder → value)

### 2.2 남은 작업 📋

1. **실제 데이터 연동**: 현재는 하드코딩된 기본값 사용 중
   - `ind_agent.py`의 `form_data`에서 DB 값을 가져와서 템플릿에 전달
   - `project_results.html`의 Phase 1/2/3 결과 데이터와 연동

2. **필드별 데이터 매핑**:
   | 필드 | 현재 상태 | 목표 |
   |------|----------|------|
   | Clinical Phase | 기본값 "Phase 1" | project results에서 가져오기 |
   | Expected Patients | 기본값 30 | cohort의 n_subjects 연동 |
   | Study Duration | 기본값 "12 weeks" | project 설정에서 가져오기 |
   | Drug Name | 기본값 설정됨 | project.title 연동 |
   | SMILES | 기본값 설정됨 | prediction.smiles 연동 |
   | Cmax, AUC, t½, Vss | 기본값 설정됨 | PK prediction 결과 연동 |

---

## 🎨 3. ind_generator.html 구조 상세

### 3.1 섹션별 구성

```
┌─────────────────────────────────────────────────────────────────┐
│                      IND Generator Page                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────┐  ┌─────────────────────────┐       │
│  │    LEFT COLUMN          │  │    RIGHT COLUMN         │       │
│  │    (Input Form)         │  │    (Preview/Result)     │       │
│  ├─────────────────────────┤  ├─────────────────────────┤       │
│  │                         │  │                         │       │
│  │ Section 1: Applicant    │  │  Ready State            │       │
│  │ ✔ 빈칸, 열림            │  │  Loading State          │       │
│  │                         │  │  Success State          │       │
│  │ Section 2: Investigator │  │  Error State            │       │
│  │ ✔ 빈칸, 열림            │  │                         │       │
│  │                         │  │                         │       │
│  │ Section 3: Clinical     │  │                         │       │
│  │ ✔ 자동채움, 접힘(회색)   │  │                         │       │
│  │                         │  │                         │       │
│  │ Section 4: Drug Info    │  │                         │       │
│  │ ✔ 자동채움, 열림(회색)   │  │                         │       │
│  │                         │  │                         │       │
│  │ Section 5: PK & Safety  │  │                         │       │
│  │ ✔ 자동채움, 열림(회색)   │  │                         │       │
│  │                         │  │                         │       │
│  │ [Generate Button]       │  │                         │       │
│  └─────────────────────────┘  └─────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 HTML 구조 (라인 번호 참조)

| 라인 범위 | 섹션 | 상태 | 설명 |
|-----------|------|------|------|
| 131-167 | Section 1: Applicant Information | 열림, 빈칸 | 신청자 정보 (선택사항) |
| 169-217 | Section 2: Investigator & Institution | 열림, 빈칸 | 연구자/기관 정보 (선택사항) |
| 219-262 | Section 3: Clinical Trial Parameters | **접힘**, 자동채움 | 임상시험 파라미터 (회색 텍스트) |
| 264-378 | Section 4: Drug Candidate Info | 열림, 자동채움 | 약물 정보 (회색 텍스트) |
| 380-480 | Section 5: PK & Safety Data | 열림, 자동채움 | PK/안전성 데이터 (회색 텍스트) |
| 482-489 | Generate Button | - | IND 생성 버튼 |
| 492-600 | Right Panel (Preview) | - | 결과 미리보기 |

### 3.3 핵심 CSS 클래스

```html
<!-- 접힘 상태 (collapsed) -->
<div class="collapsible-header collapsed ...">  <!-- header에 'collapsed' 추가 -->
<div class="collapsible-content ...">            <!-- 'open' 클래스 제거 -->

<!-- 열림 상태 (open) -->
<div class="collapsible-header ...">             <!-- 'collapsed' 없음 -->
<div class="collapsible-content open ...">       <!-- 'open' 클래스 추가 -->

<!-- 회색 텍스트 (자동 채움 필드) -->
<input class="form-input text-gray-500" value="기본값">
<select class="form-input text-gray-500">
```

---

## 🔧 4. Backend 연동 (ind_agent.py)

### 4.1 데이터 흐름

```
project_id → ind_agent.py → form_data → ind_generator.html
                ↓
         [Prediction 조회]
         [Cohort 조회]
         [Project 조회]
                ↓
         form_data = {
           "smiles": prediction.smiles,
           "cmax": pk.get("human_Cmax_ng_mL_linear"),
           "auc": pk.get("human_AUC_ng_h_mL_linear"),
           "t_half": pk.get("human_thalf_linear"),
           "target_population": cohort_data["population"],
           "expected_patients": cohort_data["n_subjects"],
           ...
         }
```

### 4.2 핵심 함수

| 함수 | 파일 | 역할 |
|------|------|------|
| `ind_generator_page()` | `ind_agent.py` (라인 70-243) | IND Generator 페이지 렌더링 & form_data 구성 |
| `generate_ind()` | `ind_agent.py` (라인 248+) | IND 문서 생성 API |

---

## 🚀 5. How to Run

```bash
# 가상환경 활성화 (Windows)
.\.venv\Scripts\activate

# 서버 실행
uvicorn app.main:app --reload

# 또는 Docker 사용
.\restart_server.ps1
```

**접속 주소**: http://127.0.0.1:8000/ind-generator?project_id=8

---

## 📝 6. 작업 시 주의사항

### 6.1 템플릿 수정 시
- `{% extends "app_base.html" %}` 절대 삭제 금지 (사이드바 깨짐)
- `{% block content %}` 내부만 수정
- Jinja2 문법 주의: `{{ form_data.field or 'default' }}`

### 6.2 스타일 적용 시
- Tailwind CSS 사용 중
- 회색 자동채움 텍스트: `text-gray-500`
- 접힘/열림: `collapsed` 클래스 + `open` 클래스

### 6.3 Agent 지시 시
```
"ind_generator.html의 Section 3 (Clinical Trial Parameters)를 
project_results.html의 Phase 1 데이터와 연동해줘.
form_data에서 값을 가져오도록 수정해줘."
```

---

## 🏗️ 7. Template & Router Details

### 7.1 Template Hierarchy

| 파일명 | 역할 | 상속 관계 |
|--------|------|-----------|
| `base.html` | Root - Tailwind, Fonts | - |
| `app_base.html` | Dashboard Layout (사이드바+헤더) | extends base.html |
| `auth_base.html` | Auth Layout (중앙 카드) | extends base.html |
| `ind_generator.html` | **IND 입력 폼** | extends app_base.html |
| `project_results.html` | Phase 1/2/3 결과 | extends app_base.html |

### 7.2 Router Roles

| 파일 | 역할 | 주요 페이지 |
|------|------|------------|
| `auth.py` | 인증 (로그인/회원가입) | login.html, signup.html |
| `projects.py` | 프로젝트 관리 | project_detail.html |
| `cohorts.py` | 코호트 관리 | cohort_detail.html |
| `ind_agent.py` | **IND 생성** | ind_generator.html |
| `dashboard.py` | 대시보드 | dashboard.html |

---

## 📊 8. Current Status Summary

| 항목 | 상태 |
|------|------|
| Applicant Information | ✅ 빈칸, 열림 |
| Investigator & Institution | ✅ 빈칸, 열림 |
| Clinical Trial Parameters | ⚠️ 기본값 설정, 접힘 (DB 연동 필요) |
| Drug Candidate Info | ⚠️ 기본값 설정, 열림 (DB 연동 필요) |
| PK & Safety Data | ⚠️ 기본값 설정, 열림 (DB 연동 필요) |
| 회색 텍스트 스타일 | ✅ 적용 완료 |
| Generate 기능 | ✅ 작동 중 |

**다음 단계**: `ind_agent.py`에서 실제 DB 데이터를 `form_data`에 채워서 템플릿에 전달
