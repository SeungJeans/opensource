#!/usr/bin/env bash
# macOS/Linux용: ./run_all.sh  (7w 폴더, 가상환경 .venv 활성화 후)
set -u
mkdir -p outputs
for s in step1_ollama step2_retrieval step3_basic_rag; do
  echo "=== $s ==="; python "$s.py" 2>&1 | tee "outputs/$s.txt"
done
echo "=== step4_chat_rag ==="
python step4_chat_rag.py < questions.txt 2>&1 | tee outputs/step4_chat_rag.txt
