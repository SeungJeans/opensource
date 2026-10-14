# ==========================================================
# STEP 1. Ollama 연결 확인
# Python에서 로컬 LLM(gemma3:4b)을 호출할 수 있는지만 확인한다.
# ==========================================================

from langchain_ollama import ChatOllama

# 1. 로컬 Ollama 모델 연결
llm = ChatOllama(
    model="gemma3:4b",
    temperature=0,
)

# 2. 테스트 질문
question = "RAG가 무엇인지 한 문장으로 설명해줘."
response = llm.invoke(question)

# 3. 결과 출력
print("질문:")
print(question)

print("\n답변:")
print(response.content)
