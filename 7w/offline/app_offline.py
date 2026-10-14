# ==========================================================
# STEP 5 (오프라인). Streamlit 대화형 RAG Chatbot
# 실행: streamlit run app_offline.py
# 원본 app.py와 같은 구조(pending_question 2단계 처리)이며
# Ollama/Hugging Face 대신 TF-IDF 검색 + 문장 추출을 사용한다.
# ==========================================================

import streamlit as st

from offline_rag import (
    TfidfRetriever,
    build_search_query,
    extractive_answer,
    load_chunks,
)

st.set_page_config(page_title="RAG Chatbot (오프라인)", page_icon="💬")
st.title("한성대학교 학사공지 RAG Chatbot")
st.caption("오프라인 버전: TF-IDF 검색 + 관련 문장 추출 (LLM 미사용)")


@st.cache_resource
def build_rag():
    _, chunks = load_chunks()
    return TfidfRetriever(chunks, k=3)


retriever = build_rag()


def ask_chat_rag(question, chat_history):
    search_query = build_search_query(question, chat_history)
    docs = retriever.invoke(search_query)
    return extractive_answer(question, docs), docs, search_query


if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if st.button("대화 초기화"):
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            with st.expander("검색된 문서 확인"):
                st.write("검색 Query")
                st.code(message["search_query"])

                for i, source in enumerate(message["sources"], start=1):
                    st.write(f"검색 결과 {i}")
                    st.write("파일:", source["file_name"])
                    st.write(source["content"])
                    st.divider()

if st.session_state.pending_question is not None:
    question = st.session_state.pending_question
    previous_history = st.session_state.messages[:-1]

    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하고 있습니다..."):
            answer, docs, search_query = ask_chat_rag(question, previous_history)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "search_query": search_query,
            "sources": [
                {"file_name": d["file_name"], "content": d["content"][:800]}
                for d in docs
            ],
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
