@echo off
echo ========================================
echo   Voice Clone Studio DramaBox Edition
echo ========================================
echo.

call venv\Scripts\activate.bat

echo Starting Voice Clone Studio...
echo Checking available engines...
echo.

python voice_clone_studio.py

pause
