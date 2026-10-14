# ==========================================================
# STEP 2. 문서 검색 (Retrieval)
# PDF 불러오기 -> Chunking -> Embedding -> Chroma -> 검색
# 아직 LLM은 사용하지 않는다.
# ==========================================================

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ----------------------------------------------------------
# 1. 데이터 폴더: ../6w/data
# ----------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR.parent / "6w" / "data"

print("데이터 폴더:")
print(DATA_DIR)

# ----------------------------------------------------------
# 2. 공지 PDF 불러오기
# ----------------------------------------------------------
pdf_files = sorted(DATA_DIR.glob("공지*.pdf"))

if not pdf_files:
    raise FileNotFoundError("6w/data 폴더에서 공지 PDF를 찾을 수 없습니다.")

documents = []

for pdf_file in pdf_files:
    print(f"\n문서 불러오기: {pdf_file.name}")

    pages = PyPDFLoader(str(pdf_file)).load()

    # 검색 결과에서 파일명을 확인할 수 있도록 Metadata 추가
    for page in pages:
        page.metadata["file_name"] = pdf_file.name

    documents.extend(pages)

print("\n불러온 전체 페이지 수:", len(documents))

# ----------------------------------------------------------
# 3. Chunking
# ----------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
)
chunks = splitter.split_documents(documents)

print("생성된 Chunk 수:", len(chunks))

# ----------------------------------------------------------
# 4. Embedding 모델 (다국어 지원)
# ----------------------------------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ----------------------------------------------------------
# 5. Chroma Vector DB + Retriever (관련 Chunk 3개)
# ----------------------------------------------------------
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="week7_step2",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ----------------------------------------------------------
# 6. 검색 테스트
# ----------------------------------------------------------
question = "인턴십 관련 공지의 주요 내용은 무엇인가요?"

print("\n질문:")
print(question)

docs = retriever.invoke(question)

for i, doc in enumerate(docs, start=1):
    print(f"\n========== 검색 결과 {i} ==========")
    print("파일:", doc.metadata.get("file_name", "확인 불가"))
    print("페이지:", doc.metadata.get("page", "확인 불가"))
    print("\n문서 내용:")
    print(doc.page_content[:800])
