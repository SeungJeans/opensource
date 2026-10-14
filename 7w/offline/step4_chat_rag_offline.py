# STEP 4 (오프라인). 이전 질문 + 현재 질문으로 검색하는 대화형 RAG
# 사용: python step4_chat_rag_offline.py            (직접 입력, exit로 종료)
#       python step4_chat_rag_offline.py < ../questions.txt
from offline_rag import (
    TfidfRetriever,
    build_search_query,
    extractive_answer,
    load_chunks,
)

_, chunks = load_chunks()
retriever = TfidfRetriever(chunks, k=3)


def ask_chat_rag(question, chat_history):
    search_query = build_search_query(question, chat_history)
    docs = retriever.invoke(search_query)
    # 후속 질문은 짧아서 단독으로는 검색이 어렵기 때문에, 답변 추출은 현재 질문 기준으로 한다
    return extractive_answer(question, docs), docs, search_query


chat_history = []

print("대화형 RAG Chatbot (오프라인 버전)")
print("종료하려면 exit를 입력하세요.")

while True:
    try:
        question = input("\n사용자: ")
    except EOFError:
        break

    if question.strip().lower() == "exit":
        break

    if not question.strip():
        continue

    print(question)  # 파일 입력 시에도 질문이 출력에 남도록 에코

    answer, docs, search_query = ask_chat_rag(question, chat_history)

    print("\n[검색 Query]")
    print(search_query)

    print("[검색된 문서]", ", ".join(f"{d['file_name']} p.{d['page']}" for d in docs))
    print("\nAI:")
    print(answer)

    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})
