# STEP 2 (오프라인). PDF -> Chunk -> TF-IDF -> 관련 Chunk 3개 검색
from offline_rag import DATA_DIR, TfidfRetriever, load_chunks

print("데이터 폴더:")
print(DATA_DIR)

pages, chunks = load_chunks()
print("\n불러온 전체 페이지 수:", pages)
print("생성된 Chunk 수:", len(chunks))

retriever = TfidfRetriever(chunks, k=3)

question = "인턴십 관련 공지의 주요 내용은 무엇인가요?"
print("\n질문:")
print(question)

for i, doc in enumerate(retriever.invoke(question), start=1):
    print(f"\n========== 검색 결과 {i} (유사도 {doc['score']:.3f}) ==========")
    print("파일:", doc["file_name"])
    print("페이지:", doc["page"])
    print("\n문서 내용:")
    print(doc["content"][:800])
