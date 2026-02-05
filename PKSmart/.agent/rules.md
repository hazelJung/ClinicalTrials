# Agent Rules

이 파일은 AI Agent(Antigravity)가 프로젝트를 수행할 때 따라야 할 영구적인 규칙을 정의합니다.

## 1. 언어 및 커뮤니케이션 (Communication)
- **모든 계획(Plan), 설명(Explanation), 대화(Chat)는 반드시 '한국어(Korean)'로 진행한다.**
- 사용자가 영어로 질문하더라도, 특별한 요청이 없다면 한국어로 답변한다.
- 기술 용어(Docker, Port 등)는 원어 표기를 병기하거나 그대로 사용해도 무방하다.
- 코드를 수정할 때는 기존 스타일(Prettier 설정)을 반드시 따르세요.
- 새로운 라이브러리를 설치할 때는 반드시 허락을 구하세요.
- 주석은 친절하고 구체적으로 달아주세요.

## 2. 실행 환경 (Execution Environment)
- 이 프로젝트는 `venv`와 `uvicorn`을 사용하는 로컬 Python 환경 또는 Docker 환경에서 실행될 수 있다.
- **실행 전 확인 명령어**:
  - 포트 확인: `netstat -ano | findstr :8000`
  - 도커 프로세스: `docker ps`