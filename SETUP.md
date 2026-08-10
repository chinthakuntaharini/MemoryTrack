# MemoryTrack - Setup Guide

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize Database

```bash
python -c "from utils.db_manager import DatabaseManager; db = DatabaseManager(); db.init_db()"
```

### 3. Download Models

The system will automatically download models on first run, but you can pre-download them:

- **YOLOv11**: Auto-downloaded by Ultralytics
- **MediaPipe**: Auto-downloaded on first use
- **TorchReID**: Download from [TorchReID GitHub](https://github.com/kaiyangzhou/deep-person-reid)

### 4. Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_memory.py
pytest tests/test_matching.py

# Run with coverage
pytest --cov=core --cov=utils --cov-report=html
```

### 5. Run CLI Pipeline

```bash
# Process a video file
python main.py --video path/to/video.mp4 --output results/output.mp4

# Process with GPU (if available)
python main.py --video path/to/video.mp4 --device cuda

# Process webcam
python main.py --webcam 0

# Process without display (headless)
python main.py --video video.mp4 --no-display
```

### 6. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## Configuration

Edit `config/settings.yaml` to customize:

- Detection thresholds
- Tracking parameters
- Feature extraction settings
- Memory bank configuration
- Fusion weights

## Troubleshooting

### CUDA Not Available

If you see "CUDA not available" warnings:
1. Check if you have an NVIDIA GPU
2. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
3. Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Model Loading Errors

If models fail to load:
1. Check your internet connection (for auto-download)
2. Manually download models and update paths in `config/settings.yaml`
3. Use CPU mode by setting `device: "cpu"` in configuration

### Import Errors

If you get import errors:
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check Python version (requires 3.11+)
3. Try reinstalling: `pip install --upgrade -r requirements.txt`

## Project Structure

```
MemoryTrack/
├── config/
│   └── settings.yaml           # Configuration file
├── core/
│   ├── detector.py             # YOLOv11 + ByteTrack
│   ├── pose_extractor.py       # MediaPipe pose
│   ├── reid_extractor.py       # TorchReID appearance
│   ├── color_extractor.py      # OpenCV HSV color
│   ├── feature_fusion.py       # Multi-modal fusion
│   └── memory_bank.py          # FAISS memory bank
├── database/
│   └── schema.sql              # Database schema
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── utils/
│   ├── config_loader.py        # Config management
│   ├── db_manager.py           # Database operations
│   ├── video_loader.py         # Video stream handling
│   └── visualization.py        # Frame visualization
├── tests/
│   ├── test_memory.py          # Memory bank tests
│   └── test_matching.py        # Fusion tests
├── requirements.txt            # Python dependencies
├── main.py                     # CLI runner
└── README.md                   # Project documentation
```

## Next Steps

1. **Test with sample video**: Download a sample video and run `python main.py --video sample.mp4`
2. **Explore dashboard**: Run `streamlit run dashboard/app.py` and explore the UI
3. **Customize configuration**: Edit `config/settings.yaml` for your use case
4. **Add missing persons**: Use the dashboard to add missing person profiles
5. **Run multi-camera**: Configure multiple cameras in settings and process simultaneously

## Performance Tips

- **GPU Acceleration**: Use CUDA for 2-3x speedup
- **Frame Skipping**: Set `frame_skip` in config to process every Nth frame
- **Resolution**: Reduce `resize_width/height` in config for faster processing
- **Model Selection**: Use `yolo11n.pt` for speed, `yolo11x.pt` for accuracy

## Research Extensions

To extend this for research:

1. **Gait Recognition**: Integrate OpenGait for gait features
2. **Custom Dataset**: Create your own multi-camera dataset
3. **Ablation Studies**: Test different fusion strategies
4. **Temporal Analysis**: Analyze long-term tracking patterns
5. **Paper Writing**: Document methodology and results
