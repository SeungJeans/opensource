# 2026 AI·SW 융합 – 7주차 대화형 RAG Chatbot 실습

6주차 Basic RAG를 확장하여 **이전 대화를 반영하는 대화형 RAG Chatbot**을 구현합니다.
(과제 원본: [hjlee-crypto/2026-ai-sw-convergence](https://github.com/hjlee-crypto/2026-ai-sw-convergence))

## 폴더 구조

```text
.
├─ 6w/data/            # 공지1.pdf, 공지2.pdf 등 RAG 대상 문서
└─ 7w/
   ├─ step1_ollama.py       # Ollama 연결 확인
   ├─ step2_retrieval.py    # Chunking → Embedding → Chroma → 검색
   ├─ step3_basic_rag.py    # 검색 결과로 답변 생성 (1회)
   ├─ step4_chat_rag.py     # 이전 대화를 반영하는 터미널 챗봇
   ├─ app.py                # Streamlit Chat UI
   └─ requirements.txt
```

> 코드는 `7w`에서 실행하며 `../6w/data`의 `공지*.pdf`를 읽습니다. 두 폴더의 상대 위치를 바꾸지 마세요.

## 실습 흐름

| 단계 | 파일 | 하는 일 |
| --- | --- | --- |
| STEP 1 | `step1_ollama.py` | Python에서 `gemma3:4b` 호출 확인 |
| STEP 2 | `step2_retrieval.py` | PDF → Chunk(700/100) → Embedding → Chroma → 관련 Chunk 3개 검색 |
| STEP 3 | `step3_basic_rag.py` | 검색 결과를 Prompt에 넣어 문서 기반 답변 생성 |
| STEP 4 | `step4_chat_rag.py` | 직전 질문 + 현재 질문으로 검색, 대화 기록을 Prompt에 포함 |
| STEP 5 | `app.py` | Streamlit 웹 Chat UI (검색 근거 확인, 대화 초기화) |

## 환경 설정

1. [Python 3.12](https://www.python.org/downloads/), [VS Code](https://code.visualstudio.com/), [Ollama](https://ollama.com/download) 설치
2. 모델 다운로드 및 확인

   ```bash
   ollama pull gemma3:4b
   ollama list
   ```

3. 가상환경 생성 및 패키지 설치 (`7w` 폴더에서)

   ```powershell
   cd 7w
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1      # macOS/Linux: source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## 빠른 실행 (Windows PowerShell, `7w` 폴더)

```powershell
.\setup.ps1      # 가상환경 + 패키지 + gemma3:4b 설치
.\run_all.ps1    # STEP 1~4 실행, 출력은 outputs\*.txt 에 저장
```

macOS/Linux는 가상환경 활성화 후 `./run_all.sh`. 결과는 `7w/실행및결과.md`에 정리합니다.

## 단계별 실행

```bash
python step1_ollama.py
python step2_retrieval.py
python step3_basic_rag.py
python step4_chat_rag.py      # 종료: exit
streamlit run app.py
```

예시 (STEP 4):

```text
사용자: 인턴십 관련 정보 알려줘.
AI: ...

사용자: 그럼 제출 기한은?     ← 이전 질문의 맥락을 이어받음
AI: ...
```

## 모델 없이 실행해 보기 (오프라인 버전)

Ollama / Hugging Face를 쓸 수 없을 때를 위해 `7w/offline/`에 TF-IDF 검색 + 문장 추출 버전을 넣었습니다.
원본과 결과가 다르므로 **구조 확인용**이며, 실제 실행 결과와 한계는 [`7w/offline/실행및결과_오프라인.md`](7w/offline/실행및결과_오프라인.md)에 있습니다.

```bash
cd 7w/offline
pip install -r requirements-offline.txt
./run_all_offline.sh && streamlit run app_offline.py
```

## 제출용 스크린샷 체크리스트

각 단계 실행 결과를 캡처해 두면 됩니다.

- [ ] STEP 1: 질문/답변 출력
- [ ] STEP 2: 검색 결과 3개 출력
- [ ] STEP 3: 검색된 문서 + RAG 답변
- [ ] STEP 4: 후속 질문이 이어지는 대화
- [ ] STEP 5: 브라우저 Chat UI 및 "검색된 문서 확인"
