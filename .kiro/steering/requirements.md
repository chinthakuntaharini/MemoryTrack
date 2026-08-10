# MemoryTrack - Requirements Specification

## Project Overview
MemoryTrack is an adaptive memory-based multi-camera missing person tracking system that fuses multiple computer vision features (appearance, pose, gait, color, accessories) into a unified vector representation stored in an adaptive memory bank.

## Core Requirements

### Functional Requirements
1. **Multi-Camera Video Processing**: Process multiple video streams simultaneously
2. **Person Detection**: Detect persons in video frames using YOLOv11
3. **Object Tracking**: Maintain consistent person IDs across frames using ByteTrack
4. **Feature Extraction**: Extract multiple feature modalities:
   - Appearance/ReID embeddings (TorchReID)
   - Pose keypoints and body ratios (MediaPipe/MMPose)
   - Color histograms (OpenCV HSV)
   - Accessory detection (YOLO)
   - Motion/trajectory features (ByteTrack)
5. **Adaptive Memory Bank**: Store and manage temporal feature snapshots with decay
6. **Feature Fusion**: Combine multiple modalities into unified embeddings
7. **Similarity Matching**: Fast vector similarity search using FAISS
8. **Explainable Matching**: Provide match explanations with modality contributions
9. **Dashboard UI**: Streamlit interface for real-time monitoring and search

### Non-Functional Requirements
1. **Performance**: Real-time processing capability (≥15 FPS on CPU, ≥30 FPS on GPU)
2. **Scalability**: Support 1000+ person profiles in memory bank
3. **Robustness**: Graceful fallback to CPU when GPU unavailable
4. **Modularity**: Clean separation of concerns with well-defined interfaces
5. **Testability**: Unit tests for core components
6. **Documentation**: Type hints and docstrings for all modules

## Technical Stack

### Core Framework
- **Python**: 3.11+
- **PyTorch**: 2.0+ (with CUDA support optional)
- **OpenCV**: 4.8+

### Detection & Tracking
- **Ultralytics YOLO**: 8.0+ (YOLOv11 support)
- **ByteTrack**: Latest stable release

### Feature Extraction
- **TorchReID**: deep-person-reid (for appearance embeddings)
- **MediaPipe**: 0.10+ (pose estimation)
- **MMPose**: 1.0+ (alternative pose solution)
- **OpenCV**: Built-in color histogram extraction

### Vector Storage
- **FAISS**: 1.7+ (Facebook AI Similarity Search)

### Dashboard
- **Streamlit**: 1.28+

### Database
- **SQLite**: Built-in (for metadata storage)
- **PostgreSQL**: Optional (for production deployments)

### Testing
- **pytest**: 7.4+
- **pytest-cov**: For coverage reporting

## Dependency Versions

```
# Core
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0

# Detection & Tracking
ultralytics>=8.0.0
bytetrack>=1.0.0

# Feature Extraction
torchreid>=0.2.5
mediapipe>=0.10.0
mmpose>=1.0.0
mmdet>=3.0.0

# Vector Storage
faiss-cpu>=1.7.0  # Use faiss-gpu for CUDA support

# Dashboard
streamlit>=1.28.0
plotly>=5.17.0

# Database
sqlalchemy>=2.0.0

# Utilities
pyyaml>=6.0
tqdm>=4.66.0
pillow>=10.0.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
```

## Hardware Requirements

### Minimum (CPU-only)
- CPU: 4 cores, 2.5 GHz+
- RAM: 16 GB
- Storage: 50 GB SSD

### Recommended (GPU-accelerated)
- CPU: 8 cores, 3.0 GHz+
- RAM: 32 GB
- GPU: NVIDIA RTX 3060+ (8 GB VRAM)
- Storage: 100 GB SSD

## System Architecture

```
Video Input (Multi-camera)
    ↓
YOLOv11 Person Detection
    ↓
ByteTrack Tracking
    ↓
┌─────────────────────────────────┐
│  Multi-Modal Feature Extraction  │
├─────────────────────────────────┤
│ • TorchReID (Appearance)        │
│ • MediaPipe (Pose/Body Ratios)   │
│ • OpenCV (Color Histograms)      │
│ • YOLO (Accessories)            │
│ • ByteTrack (Motion/Trajectory) │
└─────────────────────────────────┘
    ↓
Feature Fusion Module
    ↓
Adaptive Memory Bank (FAISS)
    ↓
Similarity Matching & XAI
    ↓
Streamlit Dashboard
```

## Data Flow

1. **Input**: Video frames from multiple cameras
2. **Detection**: YOLOv11 detects persons → bounding boxes
3. **Tracking**: ByteTrack assigns persistent IDs
4. **Feature Extraction**: Parallel extraction of 5 modalities
5. **Fusion**: Weighted concatenation into unified vector
6. **Memory Update**: Store in FAISS with temporal decay
7. **Matching**: Query against memory bank for missing persons
8. **Output**: Match results with explanations to dashboard

## Key Algorithms

### Temporal Feature Decay
```
W(t) = e^(-λ * Δt)
```
Where λ is decay rate and Δt is time elapsed

### Feature Fusion
```
F_combined = [α * F_ReID | β * F_Pose | γ * F_Color | δ * F_Accessory | ε * F_Motion]
```
Dynamic weights adjusted based on occlusion/confidence

### Similarity Matching
- Cosine similarity for normalized vectors
- FAISS IndexFlatIP for fast retrieval
- Top-K matches with confidence scores

## Success Criteria

1. **Detection Accuracy**: ≥95% mAP on person detection
2. **Tracking Stability**: ≥90% ID consistency over 30 seconds
3. **Matching Accuracy**: ≥85% precision on missing person identification
4. **Performance**: ≥15 FPS real-time processing on CPU
5. **Scalability**: Support 1000+ profiles with <100ms query time
