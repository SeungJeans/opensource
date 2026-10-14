# 7w 폴더에서 실행: .\setup.ps1
# 가상환경 생성 + 패키지 설치 + Gemma 모델 다운로드
$ErrorActionPreference = "Stop"

python --version
ollama --version

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

ollama pull gemma3:4b
ollama list
Write-Host "`n설치 완료. 다음: .\run_all.ps1"
