# IND Agent 통합 가이드

> **목적**: PKSmart와 유사한 FastAPI 기반 프로젝트에 IND Agent 파일을 추가하는 방법  
> **대상**: 동일한 폴더 구조를 가진 프로토타입을 개발 중인 팀원

---

## 📦 전달할 파일 목록

| 파일명 | 용도 |
|--------|------|
| `llm_client.py` | LLM API 호출 클라이언트 (Ollama/Gemini/OpenAI 지원) |
| `prompts.py` | IND 문서 생성용 프롬프트 템플릿 |
| `ind_generator.py` | IND 문서 생성 핵심 서비스 |
| `ind_agent.py` | FastAPI 라우터 (API 엔드포인트) |
| `docx_template.py` | 마크다운 → DOCX 파일 변환 |

---

## 📁 파일 배치 위치

```
your_project/
├── app/
│   ├── main.py                 # ⚙️ 수정 필요
│   ├── routers/
│   │   └── ind_agent.py        # ✅ 여기에 복사
│   └── services/
│       ├── llm_client.py       # ✅ 여기에 복사
│       ├── prompts.py          # ✅ 여기에 복사
│       ├── ind_generator.py    # ✅ 여기에 복사
│       └── docx_template.py    # ✅ 여기에 복사
├── .env                        # ⚙️ API 키 추가 필요
└── requirements.txt            # ⚙️ 의존성 추가 필요
```

---

## 📋 각 파일 기능 설명

### 1. `llm_client.py` → `app/services/`

| 항목 | 설명 |
|------|------|
| **역할** | LLM API 통신 담당 |
| **지원 LLM** | Ollama (기본값), Google Gemini, OpenAI |
| **핵심 함수** | `get_llm_client()` - 환경변수에 따라 적절한 클라이언트 반환 |
| **클래스** | `LLMClient` (추상 클래스), `OllamaClient`, `GeminiClient`, `OpenAIClient` |
| **의존성** | `google-generativeai`, `openai`, `ollama` (선택적) |

---

### 2. `prompts.py` → `app/services/`

| 항목 | 설명 |
|------|------|
| **역할** | LLM에게 전달할 프롬프트 템플릿 |
| **핵심 함수** | `get_system_prompt()` - LLM 역할 정의 |
| | `get_user_prompt(data)` - 약물 데이터를 프롬프트로 변환 |
| | `format_qsar_results(qsar_predictions, threshold)` - QSAR 결과 포맷팅 |
| **커스터마이징** | 필요시 프롬프트 내용 수정 가능 |

---

### 3. `ind_generator.py` → `app/services/`

| 항목 | 설명 |
|------|------|
| **역할** | IND 문서 생성 핵심 로직 |
| **클래스** | `INDGeneratorService` |
| **핵심 메서드** | `generate_ind_draft(data)` - 약물 데이터 입력 → IND 문서 출력 |
| **반환값** | `{"text": 생성된 문서, "file_path": 파일경로, "filename": 파일명}` |
| **주요 기능** | 마크다운 테이블 후처리, DOCX 파일 자동 생성 |

---

### 4. `docx_template.py` → `app/services/`

| 항목 | 설명 |
|------|------|
| **역할** | 생성된 마크다운을 DOCX 파일로 변환 |
| **핵심 함수** | `generate_docx(markdown_text, output_path)` |
| **의존성** | `python-docx` |

---

### 5. `ind_agent.py` → `app/routers/`

| 항목 | 설명 |
|------|------|
| **역할** | REST API 엔드포인트 정의 |
| **주요 엔드포인트** | 아래 표 참조 |
| **입력** | JSON (drug_name, smiles, cmax, auc 등) |
| **출력** | 생성된 IND 문서 (마크다운/DOCX) |

#### API 엔드포인트 목록

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/ind/generator` | GET | IND 생성 페이지 렌더링 |
| `/ind/generate` | POST | IND 문서 생성 API |
| `/ind/download/{filename}` | GET | 생성된 DOCX 다운로드 |
| `/ind/molecule-image` | GET | 분자 구조 이미지 생성 |

---

## ⚙️ 추가 설정 사항

### 1. `requirements.txt`에 추가

```
google-generativeai>=0.8.0
python-docx>=0.8.11
```

### 2. `.env` 파일 설정

```env
# LLM Provider 선택 (ollama, gemini, openai 중 하나)
LLM_PROVIDER=gemini

# Gemini 사용 시 (권장)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# OpenAI 사용 시 (선택)
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o-mini

# Ollama 사용 시 (로컬, 선택)
OLLAMA_MODEL=gemma3:12b
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. `app/main.py` 수정

```python
from app.routers import ind_agent  # 추가
app.include_router(ind_agent.router)  # 추가
```

---

## 🧪 테스트 방법

서버 실행 후:

```bash
# 기본 포트(8000)
curl -X POST "http://localhost:8000/ind/generate" \
  -H "Content-Type: application/json" \
  -d '{"drug_name": "TestDrug", "cmax": 1500.5, "overall_score": 75.0}'
```

또는 브라우저에서 `http://localhost:8000/ind/generator` 접속하여 UI 확인

---

## 📚 PKSmart 원본 파일 경로

| 파일 | PKSmart 경로 |
|------|--------------|
| llm_client.py | `app/services/llm_client.py` |
| prompts.py | `app/services/prompts.py` |
| ind_generator.py | `app/services/ind_generator.py` |
| docx_template.py | `app/services/docx_template.py` |
| ind_agent.py | `app/routers/ind_agent.py` |

---

## ⚠️ 주의사항

1. **GEMINI_API_KEY 필수**: Gemini 사용 시 API 키가 없으면 서버 시작 시 에러 발생
2. **python-dotenv**: `.env` 파일 로드를 위해 `python-dotenv` 패키지 설치 권장
3. **출력 디렉토리**: 생성된 DOCX 파일은 `data/generated_ind/` 폴더에 저장됨 (자동 생성)
