# 7w 폴더에서 실행: .\run_all.ps1
# STEP 1~4를 차례로 실행하고 출력을 outputs\ 폴더에 저장한다.
# (처음 실행 시 임베딩 모델 다운로드로 시간이 걸린다)
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force outputs | Out-Null

foreach ($s in "step1_ollama", "step2_retrieval", "step3_basic_rag") {
    Write-Host "=== $s ==="
    & $py "$s.py" 2>&1 | Tee-Object -FilePath "outputs\$s.txt"
}

Write-Host "=== step4_chat_rag (questions.txt 로 자동 질문) ==="
Get-Content questions.txt -Encoding UTF8 | & $py step4_chat_rag.py 2>&1 | Tee-Object -FilePath "outputs\step4_chat_rag.txt"

Write-Host "`n완료. outputs\ 폴더의 txt 파일 내용과 스크린샷을 확인하세요."
Write-Host "웹 UI는 직접 실행:  .\.venv\Scripts\streamlit.exe run app.py"
