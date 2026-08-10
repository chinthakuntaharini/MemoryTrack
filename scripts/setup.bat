@echo off
REM MemoryTrack Setup Script for Windows

echo 🚀 MemoryTrack Setup Script
echo ==========================

REM Check Python version
echo 📋 Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python not found
    echo Please install Python 3.11 or later from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo ✅ Python %python_version% found

REM Create virtual environment
echo 🐍 Creating virtual environment...
if not exist venv (
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ℹ️  Virtual environment already exists
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo 📁 Creating directories...
mkdir logs 2>nul
mkdir weights 2>nul
mkdir database 2>nul
mkdir videos 2>nul
mkdir results 2>nul

REM Initialize database
echo 🗄️  Initializing database...
python -c "from utils.db_manager import DatabaseManager; db = DatabaseManager(); db.init_db()"

REM Download YOLO model (optional)
echo 🤖 Downloading YOLO model...
python -c "try: from ultralytics import YOLO; model = YOLO('yolo11n.pt'); print('✅ YOLOv11 model downloaded'); except Exception as e: print(f'⚠️  YOLO model will be downloaded on first run: {e}')"

REM Run basic tests
echo 🧪 Running basic tests...
python -c "import sys; sys.path.append('.'); from core import PersonDetector, AdaptiveMemoryBank; from utils import ConfigLoader, DatabaseManager; config = ConfigLoader(); config.validate(); print('✅ All modules loaded successfully')"

echo.
echo 🎉 Setup completed successfully!
echo.
echo 🚀 Quick Start:
echo    1. Activate environment: venv\Scripts\activate.bat
echo    2. Run CLI: python main.py --help
echo    3. Run dashboard: streamlit run dashboard/app.py
echo    4. Run tests: pytest tests/
echo.
echo 📖 See README.md for detailed usage instructions
pause