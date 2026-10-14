# STEP 3 (오프라인). 검색 -> 관련 문장 추출 답변 (LLM 없음, 이전 대화 미사용)
from offline_rag import TfidfRetriever, extractive_answer, load_chunks

_, chunks = load_chunks()
retriever = TfidfRetriever(chunks, k=3)


def ask_rag(question):
    docs = retriever.invoke(question)

    print("\n========== 검색된 실제 내용 ==========")
    for i, doc in enumerate(docs, start=1):
        print(f"\n[검색 결과 {i}]")
        print("파일:", doc["file_name"])
        print("페이지:", doc["page"])
        print(doc["content"][:1000])
    print("======================================")

    return extractive_answer(question, docs), docs


question = "인턴십 신청 기간은 언제인가요?"
answer, docs = ask_rag(question)

print("\n질문:")
print(question)

print("\n검색된 문서:")
for doc in docs:
    print("-", doc["file_name"])

print("\nRAG 답변:")
print(answer)
