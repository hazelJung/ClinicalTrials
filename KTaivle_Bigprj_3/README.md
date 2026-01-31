# 🧬 AI-Driven In-silico Clinical Trial & Toxicity Prediction Platform

> **Dual-Engine Platform**: mPBPK (Drug Disposition) + QSAR (Toxicity Prediction) Integrated System
>
> An AI-based digital twin platform predicting clinical success/failure of antibody and small molecule drug candidates.

---

## 🚀 Getting Started (시작하기)

> **혜정정(Hazel)님을 위한 리드미**, 에이전트한테 여기 보여주세요 예. (헤이즐이 뭐임? 헤이즐넛?)
>
> 프로젝트 플로우는 [PROJECT_STATUS.md](PROJECT_STATUS.md)
>
> 모델 학습과 파싱 코드는 `src`, 진짜 모델(pkl)은 `models`에 있습니다.
>
> `tests` 경로는 전부 실제 데이터 기반으로 하는 검증코드인데 하도 많아서 에이전트한테 새로 모델 검증해달라고 할 때 저 경로 멘션하면 도이.
### 0. 필수 프로그램 설치 (Prerequisites)
*   **Python**: v3.10 이상 (Anaconda 권장)
*   **Node.js**: v18.0.0 이상
*   **Git**: (선택 사항)

---

### Step 1: Backend 서버 실행 (Python)

독성 예측 모델과 시뮬레이션 엔진이 돌아가는 핵심 서버입니다.

1. **`api` 폴더로 이동**
   ```bash
   cd Big_Project/api
   ```

2. **가상환경 생성 및 활성화** (권장)
   ```bash
   # 가상환경 생성
   python -m venv venv

   # 가상환경 활성화 (Windows)
   venv\Scripts\activate
   
   # 가상환경 활성화 (Mac/Linux)
   source venv/bin/activate
   ```

3. **라이브러리 설치**
   *(상위 폴더의 requirements.txt를 설치합니다)*
   ```bash
   pip install -r ../requirements.txt
   ```

4. **서버 시작**
   ```bash
   uvicorn main:app --port 8000 --reload
   ```
   ✅ 성공 시: `Application startup complete.` 및 `http://127.0.0.1:8000` 로그 확인.

---

### Step 2: Frontend 대시보드 실행 (Next.js)

사용자용 웹 인터페이스입니다.

1. **새로운 터미널**을 열고 **`web-ui` 폴더로 이동**
   ```bash
   cd Big_Project/web-ui
   ```

2. **패키지 설치**
   ```bash
   npm install
   ```
   *(약 1~2분 소요됩니다)*

3. **개발 서버 시작**
   ```bash
   npm run dev
   ```

4. **접속하기**
   *   브라우저 주소창에 **`http://localhost:3000`** 입력
   *   짜잔! 🎉 대시보드가 열립니다.

---

## 📚 주요 기능 사용법

### 1. mPBPK Simulator (항체 약물동태)
*   [mPBPK Simulator] 메뉴 클릭
*   **Drug Parameters**: 약물의 KD(결합력), Dose(투여량) 입력
*   **Target Params**: 타겟 단백질의 반감기 등 설정
*   **Cohort Config**: 가상 환자군(인종, 성별, 체중) 설정
*   **[Run Simulation]** 클릭 → 농도 그래프와 성공률 확인

### 2. QSAR Toxicity Predictor (독성 예측)
*   [QSAR Predictor] 메뉴 클릭
*   **SMILES Input**: 약물의 화학 구조(SMILES) 입력 (예: Aspirin `CC(=O)Oc1ccccc1C(=O)O`)
*   **Threshold**: 민감도 조절 (기본 20%)
*   **[Analyze Toxicity]** 클릭 → 12가지 독성 예측 결과 확인

---

## 🛠️ 문제 해결 (Troubleshooting)

**Q. `Error: listen EADDRINUSE: address already in use :::3000`**
*   이미 다른 Node.js 서버가 켜져 있는 것입니다.
*   터미널에서 `Ctrl + C`를 눌러 끄거나, `taskkill /F /IM node.exe`로 강제 종료 후 다시 실행하세요.

**Q. `Failed to fetch` 에러가 떠요.**
*   Backend 서버(Python)가 꺼져 있어서 그렇습니다. Step 1을 다시 확인하세요.
*   브라우저에서 `http://localhost:8000/api/health`가 열리는지 확인하세요.

---

## 📅 Project Status

*   **현재 버전**: v1.0 (Prototype)
*   **마지막 업데이트**: 2026-01-28
*   **상태**: Phase 6 (Web UI) 완료, Phase 7 (Reporting IND?) 진행 중

---

## 📂 폴더 구조

```
Big_Project/
├── api/                 # Python Backend (FastAPI)
│   └── main.py          # 서버 실행 파일
├── web-ui/              # Frontend (Next.js)
├── src/                 # 핵심 모델 (mPBPK, QSAR, ML)
├── data/                # 데이터셋 (DrugBank, Tox21)
├── docs/                # 프로젝트 문서
└── reports/             # 시뮬레이션 결과 리포트
```
