#!/usr/bin/env bash
# 7w/offline 폴더에서 실행 (Ollama / Hugging Face 불필요)
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements-offline.txt
set -u
mkdir -p outputs
python step2_retrieval_offline.py   2>&1 | tee outputs/step2.txt
python step3_basic_rag_offline.py   2>&1 | tee outputs/step3.txt
python step4_chat_rag_offline.py < questions.txt 2>&1 | tee outputs/step4.txt
