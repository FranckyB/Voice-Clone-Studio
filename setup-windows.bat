@echo off
echo ========================================
echo Voice Clone Studio - Simplified Setup
echo ========================================
echo.
echo This setup installs the DramaBox build stack:
echo - Core app + MMAudio dependencies
echo - DramaBox finetune dependencies
echo - Optional Qwen3 ASR, Whisper, DeepFilterNet, llama.cpp
echo.

echo Select CUDA version for PyTorch (press number):
echo   1. CUDA 13.0 (default)
echo   2. CUDA 12.8
echo   3. CUDA 12.1
echo.
choice /C 123 /T 99 /D 1 /M "Enter choice"
set CUDA_CHOICE=%errorlevel%
echo.

echo Optional: Install Qwen3 ASR?
echo   1. Yes (default)
echo   2. No
choice /C 12 /T 99 /D 1 /M "Install Qwen3 ASR?"
set QWEN3ASR_CHOICE=%errorlevel%
echo.

echo Optional: Install Whisper?
echo   1. Yes (default)
echo   2. No
choice /C 12 /T 99 /D 1 /M "Install Whisper?"
set WHISPER_CHOICE=%errorlevel%
echo.

echo Optional: Install DeepFilterNet for audio denoising?
echo   1. Yes (default)
echo   2. No
choice /C 12 /T 99 /D 1 /M "Install DeepFilterNet?"
set DEEPFILTER_CHOICE=%errorlevel%
echo.

echo Optional: Install llama.cpp?
echo   1. Yes (default)
echo   2. No
choice /C 12 /T 99 /D 1 /M "Install llama.cpp?"
set LLAMA_CHOICE=%errorlevel%
echo.

echo ========================================
echo Optional: Install Flash Attention 2 for faster inference?
echo Flash Attention 2 provides fast attention for supported models.
echo Requires CUDA GPU. Cannot be used with multilingual Chatterbox.
echo ========================================
echo.
echo   1. Yes - Install Flash Attention 2 (DEFAULT)
echo   2. No  - Skip
echo.
choice /C 12 /T 99 /D 1 /M "Install Flash Attention 2?"
set FLASH_CHOICE=%errorlevel%
echo.
echo All questions answered - installing now...
echo.

echo [1/6] Checking Python installation...
set PYTHON_CMD=
where py >nul 2>&1
if %errorlevel% equ 0 (
    for %%V in (3.11 3.10) do (
        if not defined PYTHON_CMD (
            py -%%V --version >nul 2>&1
            if not errorlevel 1 (
                set PYTHON_CMD=py -%%V
            )
        )
    )
)

if not defined PYTHON_CMD (
    python --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=python
    )
)

if not defined PYTHON_CMD (
    echo ERROR: Python 3.10 or 3.11 is required.
    pause
    exit /b 1
)

for /f "tokens=2" %%a in ('%PYTHON_CMD% --version 2^>^&1') do set PYVER=%%a
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)
if not "%PYMAJOR%"=="3" (
    echo ERROR: Python 3.10-3.11 is required. Detected: %PYVER%
    pause
    exit /b 1
)
if %PYMINOR% LSS 10 (
    echo ERROR: Python 3.10-3.11 is required. Detected: %PYVER%
    pause
    exit /b 1
)
if %PYMINOR% GTR 11 (
    echo ERROR: Python 3.12+ is not supported by this setup.
    pause
    exit /b 1
)

echo Using: %PYTHON_CMD% (Python %PYVER%)
echo.

echo [2/6] Installing ffmpeg and sox...
winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements >nul 2>&1
winget install -e --id ChrisBagwell.SoX --accept-source-agreements --accept-package-agreements >nul 2>&1
echo.

echo [3/6] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    %PYTHON_CMD% -m venv venv
    if not exist venv (
        echo ERROR: Failed to create venv.
        pause
        exit /b 1
    )
)
echo.

echo [4/6] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate venv.
    pause
    exit /b 1
)
python.exe -m pip install --upgrade pip
echo.

echo [5/6] Installing PyTorch...
setlocal enabledelayedexpansion
if "%CUDA_CHOICE%"=="1" (
    set CUDA_VER=cu130
    set TORCH_VER=2.9.1
    set TORCHAUDIO_VER=2.9.1
    set TORCHVISION_VER=0.24.1
)
if "%CUDA_CHOICE%"=="2" (
    set CUDA_VER=cu128
    set TORCH_VER=2.9.1
    set TORCHAUDIO_VER=2.9.1
    set TORCHVISION_VER=0.24.1
)
if "%CUDA_CHOICE%"=="3" (
    set CUDA_VER=cu121
    set TORCH_VER=2.5.1
    set TORCHAUDIO_VER=2.5.1
    set TORCHVISION_VER=0.20.1
)

pip install torch==!TORCH_VER! torchaudio==!TORCHAUDIO_VER! torchvision==!TORCHVISION_VER! --index-url https://download.pytorch.org/whl/!CUDA_VER!
if !errorlevel! neq 0 (
    echo ERROR: Failed to install PyTorch.
    pause
    exit /b 1
)
endlocal
echo.

echo [6/6] Installing application dependencies...
if not exist requirements_windows.txt (
    echo ERROR: requirements_windows.txt not found.
    pause
    exit /b 1
)
pip install -r requirements_windows.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements.
    pause
    exit /b 1
)

if not "%DEEPFILTER_CHOICE%"=="1" goto :skip_deepfilter
echo Installing DeepFilterNet (audio denoising)...
pip install deepfilternet
if %errorlevel% neq 0 (
    echo WARNING: DeepFilterNet installation failed. Requires Rust compiler to build from source.
    echo Audio denoising will not be available, but all other features will work normally.
    echo To install later: Install Rust from https://rustup.rs then run: pip install deepfilternet
)
echo.
:skip_deepfilter

echo Installing DramaBox finetune dependencies...
pip install accelerate safetensors sentencepiece pydantic==2.10.6
pip install "resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master"
if %errorlevel% neq 0 (
    echo WARNING: Some DramaBox dependencies failed to install.
)

if not "%QWEN3ASR_CHOICE%"=="1" goto :skip_qwen3asr
echo Installing Qwen3 ASR...
pip install nagisa soynlp qwen-omni-utils pytz flask
pip install qwen-asr
:skip_qwen3asr

if not "%WHISPER_CHOICE%"=="1" goto :skip_whisper
echo Installing Whisper...
pip install openai-whisper
:skip_whisper

if not "%LLAMA_CHOICE%"=="1" goto :skip_llama
echo Installing llama.cpp...
winget install llama.cpp --accept-source-agreements --accept-package-agreements >nul 2>&1
:skip_llama

REM Flash Attention 2
if not "%FLASH_CHOICE%"=="1" goto :skip_flash

echo.
echo Installing Flash Attention 2...
setlocal enabledelayedexpansion

REM Flash Attention needs Python-version-specific wheels (cp310, cp311, cp312)
set FLASH_PY=cp3%PYMINOR%

if "%CUDA_CHOICE%"=="1" (
    set FLASH_WHL=flash_attn-2.8.3+torch2.9.1.cuda13.1-!FLASH_PY!-!FLASH_PY!-win_amd64.whl
) else (
    set FLASH_WHL=flash_attn-2.8.2-!FLASH_PY!-!FLASH_PY!-win_amd64.whl
)
set FLASH_URL=https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/!FLASH_WHL!
echo Downloading: !FLASH_WHL!
pip install "!FLASH_URL!"
if !errorlevel! neq 0 (
    echo WARNING: Flash Attention 2 installation failed.
    echo Wheel may not exist for Python 3.%PYMINOR% with your CUDA version.
    echo You can browse available wheels at: https://huggingface.co/MonsterMMORPG/Wan_GGUF/tree/main
) else (
    echo Flash Attention 2 installed successfully!
)
endlocal
goto :flash_done

:skip_flash
echo Skipping Flash Attention 2 installation.
:flash_done
echo.

echo.
echo ========================================
echo Setup Complete
echo ========================================
echo.
echo Launch with:
echo   venv\Scripts\activate
echo   python voice_clone_studio.py
echo.
pause
