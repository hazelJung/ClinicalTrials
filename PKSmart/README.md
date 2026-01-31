# 🏥 PKSmart: AI-Powered Clinical Trial Design Platform

PharmaTwin의 AI 기반 임상시험 설계 및 독성 예측 플랫폼입니다.
본 문서는 프로젝트 구조와 **IND Report 상세 페이지 개발**을 위한 가이드라인을 포함합니다.

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
│   │   └── dashboard.py        # 대시보드, IND Report(Mock), 사용자 관리
│   ├── static/                 # CSS, JS, Images
│   └── templates/              # HTML 템플릿 (Jinja2)
│       ├── base.html           # [Root] 공통 헤더/스크립트
│       ├── app_base.html       # [Dashboard Layout] 사이드바+헤더
│       ├── auth_base.html      # [Auth Layout] 중앙 카드 배치
│       ├── dashboard.html      # [Main] 3단 대시보드 (Projects, Results, Reports)
│       └── ind_report_detail.html # [Target] IND Report 상세 페이지 
├── pksmart/                    # AI Core Logic (Model, Training, Prediction)
├── docs/                       # 개발 문서
└── Requirements.txt            # 의존성 패키지 목록
```

---

## 🛠️ 2. IND Report 개발 가이드 (For Teammates & Agents)

**목표**: `ind_report_detail.html` 내부의 콘텐츠를 고도화하여 실제 FDA 제출용 리포트처럼 보이게 만드는 것.

### 2.1 Backend (Data Source)
현재 IND Report 데이터는 **DB가 아닌 Mock Data**로 관리되고 있습니다.
- **파일**: `app/routers/dashboard.py`
- **데이터 위치**: `MOCK_IND_REPORTS` 리스트 (전역 변수)
- **라우터**: `GET /dashboard/reports/{report_id}`

> **Tip**: 추후 DB 연동이 필요하면 `models.py`에 `Report` 테이블을 만들고 `dashboard.py`의 라우터를 수정해야 합니다. 지금은 `MOCK_IND_REPORTS`의 내용을 풍성하게 수정해서 테스트하세요.

### 2.2 Frontend (Template)
- **파일**: `app/templates/ind_report_detail.html`
- **상속 구조**:
    ```html
    {% extends "app_base.html" %}  <!-- 절대 변경 금지 (사이드바 유지) -->

    {% block content %}
      <!-- 여기서부터 작업 시작 -->
      <div class="min-h-screen ...">
          <div class="max-w-2xl ..."> <!-- 중앙 카드 컨테이너 -->
              <!-- 여기에 리포트 상세 내용을 작성하세요 -->
          </div>
      </div>
    {% endblock %}
    ```

### ⚠️ 2.3 주의사항 (Critical)
1.  **Layout 유지**: `{% extends "app_base.html" %}`을 지우거나 변경하지 마세요.
    - 왼쪽 **사이드바(Compact Mode)**와 상단 **헤더**는 `app_base.html`에서 자동으로 잡아줍니다.
2.  **Card Style**: 현재 `auth_base.html` 스타일의 중앙 집중형 카드로 디자인되어 있습니다.
    - 리포트 내용이 길어지면 `.max-w-2xl`(카드 너비)를 `.max-w-4xl` 등으로 늘려서 사용해도 좋습니다.
3.  **Agent 활용 시**:
    - Agent에게 *"ind_report_detail.html의 `{% block content %}` 내부만 꾸며줘"* 라고 지시하세요.
    - Sidebar나 Header를 건드리지 않도록 명시해야 디자인이 깨지지 않습니다.

---

## 🎨 3. Dashboard Design Logic

대시보드는 크게 3가지 섹션으로 구성되어 있습니다 (`dashboard.html`).

| 섹션 | 데이터 소스 | 라우터 파일 | 설명 |
| :--- | :--- | :--- | :--- |
| **PROJECTS** | `db(Projects)` | `routers/projects.py` | 사용자 생성 프로젝트. (Collections 스타일 디자인) |
| **RESULTS** | `db(Cohorts)` | `routers/projects.py` | 코호트 시뮬레이션 결과. (Studies 스타일 디자인) |
| **IND REPORT** | `Mock Data` | `routers/dashboard.py` | **[개발 대상]** 현재는 가상 데이터. 상세 페이지 개발 중. |

---

## 🚀 4. How to Run

```bash
# 서버 실행 (터미널)
uvicorn app.main:app --reload

# Windows (Script)
./restart_server.ps1
```

접속 주소: [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)

---

## 🏗️ 5. Template & Router Details

### 5.1 Template Hierarchy (Role & Composition)

| 파일명 | 역할 및 포함 내용 | 비고 |
| :--- | :--- | :--- |
| **base.html** | **[Root]** 모든 템플릿의 부모입니다.<br>- Tailwind CSS CDN<br>- Google Fonts (Inter)<br>- 공통 Meta 태그 및 Title Block | 모든 페이지는 직간접적으로 이 파일을 상속받습니다. |
| **app_base.html** | **[Dashboard Layout]** 로그인 후 메인 화면용 레이아웃입니다.<br>- **왼쪽 사이드바** (반응형: Wide/Compact 모드)<br>- **상단 헤더** (로고, 사용자 프로필, 로그아웃)<br>- `sidebar_mode='compact'` 변수로 사이드바 너비 조절 가능 | 대시보드 및 상세 페이지(`ind_report_detail.html` 등)는 반드시 이를 상속받아야 합니다. |
| **auth_base.html** | **[Auth Layout]** 로그인/회원가입용 레이아웃입니다.<br>- 사이드바 없음<br>- 배경 이미지 + 중앙 집중형 카드 컨테이너<br>- 푸터 (Copyright) | `login.html`, `signup.html` 등에서 사용합니다. |

### 5.2 Router Roles (Backend Logic)

| 파일명 (`app/routers/`) | 담당 역할 & 주요 기능 | 연결된 페이지 |
| :--- | :--- | :--- |
| **auth.py** | **[인증]**<br>- 로그인 (`/login`), 회원가입 (`/signup`)<br>- JWT 토큰 발급 및 검증<br>- 로그아웃 (`/logout`) | `login.html`, `signup.html` |
| **projects.py** | **[프로젝트 & 시뮬레이션]**<br>- 프로젝트 생성/삭제/조회 (`/projects`)<br>- 코호트 생성 및 결과 조회 (`/cohorts`)<br>- **RESULTS 섹션**의 상세 페이지 처리 | `cohort_detail.html`<br>`create_project.html`<br>`project_results.html` |
| **dashboard.py** | **[대시보드 & 리포트]**<br>- 메인 대시보드 화면 렌더링 (`/dashboard`)<br>- **IND REPORT (Mock)** 데이터 관리 및 상세 페이지<br>- 기타 (공유, 사용자 관리) | `dashboard.html`<br>`ind_report_detail.html`<br>`share.html` |
