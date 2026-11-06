@echo off
chcp 65001 >nul
cd /d D:\PaperQA2_FullStack_System
call .\venv\Scripts\activate.bat

echo =====================================================================
echo   KHOI DONG HE THONG PAPERQA2 SCIENTIFIC RAG (2 GIAO DIEN SONG SONG)
echo   1. Giao dien Nguoi dung (End-User Frontend): http://localhost:8000
echo   2. Giao dien Test Backend (Developer Lab)  : http://localhost:8501
echo =====================================================================

start "SciScholar User UI - Port 8000" cmd /k "cd /d D:\PaperQA2_FullStack_System && call .\venv\Scripts\activate.bat && python user_server.py"
start "PaperQA2 Backend Test Lab - Port 8501" cmd /k "cd /d D:\PaperQA2_FullStack_System && call .\venv\Scripts\activate.bat && streamlit run backend_test_ui.py --server.port 8501"