# ==========================================================
# STEP 5. Streamlit 대화형 RAG Chatbot
# 실행: streamlit run app.py
# ==========================================================

from pathlib import Path

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# ----------------------------------------------------------
# 1. 페이지 설정
# ----------------------------------------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
st.title("한성대학교 학사공지 RAG Chatbot")


# ----------------------------------------------------------
# 2. RAG 환경 생성 (최초 1회만 실행되도록 캐시)
# ----------------------------------------------------------
@st.cache_resource
def build_rag():
    data_dir = Path(__file__).resolve().parent.parent / "6w" / "data"

    pdf_files = sorted(data_dir.glob("공지*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("6w/data 폴더에서 공지 PDF를 찾을 수 없습니다.")

    documents = []

    for pdf_file in pdf_files:
        pages = PyPDFLoader(str(pdf_file)).load()

        for page in pages:
            page.metadata["file_name"] = pdf_file.name

        documents.extend(pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="week7_streamlit",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOllama(model="gemma3:4b", temperature=0, num_predict=120)

    return retriever, llm


retriever, llm = build_rag()


# ----------------------------------------------------------
# 3. 대화형 RAG 함수
# ----------------------------------------------------------
def ask_chat_rag(question, chat_history):
    # 직전 사용자 질문
    previous_question = None

    for message in reversed(chat_history):
        if message["role"] == "user":
            previous_question = message["content"]
            break

    # 검색 Query
    if previous_question:
        search_query = f"""
이전 질문:
{previous_question}

현재 질문:
{question}
"""
    else:
        search_query = question

    docs = retriever.invoke(search_query)

    # 검색된 문서를 하나의 문자열로 결합
    context_parts = []

    for i, doc in enumerate(docs, start=1):
        file_name = doc.metadata.get("file_name", "확인 불가")
        context_parts.append(
            f"""
[문서 {i}]

파일명:
{file_name}

내용:
{doc.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    # 이전 대화
    history_parts = []

    for message in chat_history:
        role = "사용자" if message["role"] == "user" else "AI"
        history_parts.append(f"{role}: {message['content']}")

    history_text = "\n".join(history_parts)

    prompt = f"""
당신은 한성대학교 학사공지 문서를 근거로
질문에 답하는 AI입니다.

다음 규칙을 지키세요.

1. 이전 대화는 현재 질문의 의미를 이해하는 데 사용하세요.
2. 실제 답변은 검색된 문서의 내용을 근거로 작성하세요.
3. 문서에 없는 내용은 추측하지 마세요.
4. 질문과 문서의 표현이 조금 다르더라도
   의미가 같으면 관련 정보를 사용하세요.
5. 날짜, 기간, 제출 방법 등 구체적인 정보가 있으면
   문서의 내용을 정확하게 사용하세요.
6. 검색된 문서에서 확인할 수 없는 내용은
   "제공된 문서에서 확인할 수 없습니다."
   라고 답하세요.
7. 답변은 한국어로 간결하게 작성하세요.

[이전 대화]

{history_text}

[검색된 문서]

{context}

[현재 질문]

{question}

[답변]
"""

    response = llm.invoke(prompt)
    return response.content, docs, search_query


# ----------------------------------------------------------
# 4. Session State 초기화
# ----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ----------------------------------------------------------
# 5. 대화 초기화 버튼
# ----------------------------------------------------------
if st.button("대화 초기화"):
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.rerun()

# ----------------------------------------------------------
# 6. 기존 대화 출력
# ----------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # AI 답변에는 검색 근거를 함께 표시
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("검색된 문서 확인"):
                st.write("검색 Query")
                st.code(message["search_query"])

                for i, source in enumerate(message["sources"], start=1):
                    st.write(f"검색 결과 {i}")
                    st.write("파일:", source["file_name"])
                    st.write(source["content"])
                    st.divider()

# ----------------------------------------------------------
# 7. 답변 생성 / 질문 입력
#    답변 생성 중에는 입력창이 나타나지 않도록
#    pending_question 상태로 두 단계로 나눠 처리한다.
# ----------------------------------------------------------
if st.session_state.pending_question is not None:
    question = st.session_state.pending_question

    # 현재 질문을 제외한 이전 대화
    previous_history = st.session_state.messages[:-1]

    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하고 있습니다..."):
            answer, docs, search_query = ask_chat_rag(question, previous_history)

    sources = [
        {
            "file_name": doc.metadata.get("file_name", "확인 불가"),
            "content": doc.page_content[:800],
        }
        for doc in docs
    ]

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "search_query": search_query,
            "sources": sources,
        }
    )

    st.session_state.pending_question = None
    st.rerun()

else:
    question = st.chat_input("질문을 입력하세요.")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.pending_question = question
        st.rerun()
