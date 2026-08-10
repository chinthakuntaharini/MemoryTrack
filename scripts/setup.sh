#!/bin/bash
# MemoryTrack Setup Script for Linux/MacOS

set -e

echo "🚀 MemoryTrack Setup Script"
echo "=========================="

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.11+ required, found Python $python_version"
    echo "Please install Python 3.11 or later"
    exit 1
fi

echo "✅ Python $python_version found"

# Create virtual environment
echo "🐍 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "ℹ️  Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs weights database videos results

# Initialize database
echo "🗄️  Initializing database..."
python -c "from utils.db_manager import DatabaseManager; db = DatabaseManager(); db.init_db()"

# Download YOLO model (optional)
echo "🤖 Downloading YOLO model..."
python -c "
try:
    from ultralytics import YOLO
    model = YOLO('yolo11n.pt')
    print('✅ YOLOv11 model downloaded')
except Exception as e:
    print(f'⚠️  YOLO model will be downloaded on first run: {e}')
"

# Run basic tests
echo "🧪 Running basic tests..."
python -c "
import sys
sys.path.append('.')

# Test imports
try:
    from core import PersonDetector, AdaptiveMemoryBank
    from utils import ConfigLoader, DatabaseManager
    print('✅ All core modules imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)

# Test configuration
try:
    config = ConfigLoader()
    config.validate()
    print('✅ Configuration loaded successfully')
except Exception as e:
    print(f'❌ Configuration error: {e}')
    sys.exit(1)
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "🚀 Quick Start:"
echo "   1. Activate environment: source venv/bin/activate"
echo "   2. Run CLI: python main.py --help"
echo "   3. Run dashboard: streamlit run dashboard/app.py"
echo "   4. Run tests: pytest tests/"
echo ""
echo "📖 See README.md for detailed usage instructions"