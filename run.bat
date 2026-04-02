@echo off
REM MDroid Attendance System - Dual Platform Launcher

echo.
echo ======================================
echo  MDroid Attendance System
echo ======================================
echo.
echo Choose a platform to run:
echo.
echo   1) Flask (Traditional Web App)
echo   2) Streamlit (Modern Dashboard)
echo   3) Exit
echo.

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Starting Flask application...
    echo Opening at http://127.0.0.1:5000
    echo.
    python app.py
) else if "%choice%"=="2" (
    echo.
    echo Starting Streamlit application...
    echo Opening at http://localhost:8501
    echo.
    streamlit run streamlit_app.py
) else if "%choice%"=="3" (
    exit /b 0
) else (
    echo Invalid choice. Please try again.
    pause
    goto :eof
)

pause
