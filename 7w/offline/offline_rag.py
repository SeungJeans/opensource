# ==========================================================
# 오프라인 RAG 공통 모듈 (Ollama / Hugging Face 없이 동작)
#
# 원본 실습과 다른 점
#   Embedding : sentence-transformers  ->  TF-IDF (글자 2~3-gram)
#   LLM       : gemma3:4b (Ollama)      ->  검색된 문서에서 관련 문장 추출
# 나머지(PDF 로딩, Chunk 700/100, 검색 k=3, 대화형 Query 구성)는 동일하다.
# ==========================================================

import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "6w" / "data"


def load_chunks(chunk_size=700, chunk_overlap=100):
    """공지*.pdf -> 페이지 -> Chunk. 각 Chunk에 파일명/페이지 Metadata 포함."""
    pdf_files = sorted(DATA_DIR.glob("공지*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("6w/data 폴더에서 공지 PDF를 찾을 수 없습니다.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    pages, chunks = 0, []

    for pdf_file in pdf_files:
        for page_no, page in enumerate(PdfReader(str(pdf_file)).pages):
            pages += 1
            for text in splitter.split_text(page.extract_text() or ""):
                chunks.append(
                    {"file_name": pdf_file.name, "page": page_no, "content": text}
                )

    return pages, chunks


class TfidfRetriever:
    """Embedding + Vector DB + Retriever 역할을 TF-IDF 하나로 대체."""

    def __init__(self, chunks, k=3):
        self.chunks = chunks
        self.k = k
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
        self.matrix = self.vectorizer.fit_transform(c["content"] for c in chunks)

    def invoke(self, query):
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        top = scores.argsort()[::-1][: self.k]
        return [dict(self.chunks[i], score=float(scores[i])) for i in top]


# 질문의 의미와 무관한 어미/요청 표현 (줄 선택 시 점수에서 제외)
QUESTION_NOISE = re.compile(
    r"(무엇인가요|무엇이야|알려\s?줘|알려주세요|언제인가요|언제야|어떻게\s?되나요|인가요|입니까|은\?|는\?|\?)"
)


def extractive_answer(question, docs, max_lines=4):
    """LLM 대신, 검색된 문서에서 질문과 가장 가까운 줄을 골라 답변으로 구성한다."""
    question = QUESTION_NOISE.sub(" ", question)

    lines, seen = [], set()
    for doc in docs:
        for line in doc["content"].splitlines():
            line = line.strip().lstrip("-*※ ").strip()
            if len(line) >= 8 and line not in seen:  # 중복 줄 제거
                seen.add(line)
                lines.append((doc["file_name"], line))

    if not lines:
        return "제공된 문서에서 확인할 수 없습니다."

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
    matrix = vec.fit_transform([q for _, q in lines] + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1])[0]

    if scores.max() < 0.05:
        return "제공된 문서에서 확인할 수 없습니다."

    best = sorted(scores.argsort()[::-1][:max_lines])  # 문서 내 원래 순서 유지
    return "\n".join(f"- {lines[i][1]}  ({lines[i][0]})" for i in best)


def build_search_query(question, chat_history):
    """직전 사용자 질문 + 현재 질문으로 검색 Query 구성 (원본 STEP 4와 동일)."""
    previous = ""
    for message in reversed(chat_history):
        if message["role"] == "user":
            previous = message["content"]
            break

    if previous:
        return f"이전 질문:\n{previous}\n\n현재 질문:\n{question}\n"
    return question
