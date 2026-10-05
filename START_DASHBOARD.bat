@echo off
setlocal
cd /d "%~dp0"
if exist "..\.venv\Scripts\python.exe" (
  "..\.venv\Scripts\python.exe" -c "import streamlit, torch, torchvision, plotly" >nul 2>nul
  if not errorlevel 1 (
    "..\.venv\Scripts\python.exe" -m streamlit run app.py
    goto :end
  )
)
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo Install Python 3.12, then open this file again.
    goto :end
  )
)
".venv\Scripts\python.exe" -c "import torch, torchvision" >nul 2>nul
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 goto :end
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :end
".venv\Scripts\python.exe" -m streamlit run app.py
:end
pause
