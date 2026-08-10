# MemoryTrack - Design Document

## System Architecture

### High-Level Architecture
MemoryTrack follows a modular pipeline architecture with clear separation between detection, tracking, feature extraction, fusion, memory management, and visualization.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Video Input Layer                             │
│  (Multi-camera streams, RTSP, files, webcam)                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Detection & Tracking Layer                        │
│  ┌──────────────┐         ┌──────────────┐                       │
│  │   YOLOv11    │────────→│  ByteTrack   │                       │
│  │  Detection   │         │   Tracking   │                       │
│  └──────────────┘         └──────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Modal Feature Extraction Layer                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  TorchReID   │  │  MediaPipe   │  │   OpenCV     │           │
│  │  Appearance  │  │   Pose       │  │   Color      │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│  ┌──────────────┐  ┌──────────────┐                           │
│  │  YOLO Acc.   │  │ ByteTrack    │                           │
│  │  Detection   │  │   Motion      │                           │
│  └──────────────┘  └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Feature Fusion Layer                           │
│  Weighted concatenation with dynamic weight adjustment           │
│  Normalization + dimensionality reduction (PCA optional)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 Adaptive Memory Bank Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   FAISS      │  │  Temporal    │  │  Dynamic     │           │
│  │   Index      │  │   Decay      │  │   Update     │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 Matching & XAI Layer                              │
│  Similarity search + modality contribution analysis               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Presentation Layer                             │
│  Streamlit Dashboard + Visualization + API Endpoints              │
└─────────────────────────────────────────────────────────────────┘
```

## Component Design

### 1. Detector Module (`core/detector.py`)

**Purpose**: Wrapper for YOLOv11 person detection and ByteTrack tracking

**Key Classes**:
```python
class PersonDetector:
    """YOLOv11-based person detection"""
    - __init__(model_path: str, confidence_threshold: float)
    - detect(frame: np.ndarray) -> List[BoundingBox]
    
class MultiObjectTracker:
    """ByteTrack wrapper for multi-object tracking"""
    - __init__(max_age: int, min_hits: int)
    - update(detections: List[BoundingBox]) -> Dict[int, Track]
```

**Design Decisions**:
- Use YOLOv11n for real-time performance, YOLOv11x for accuracy
- ByteTrack for robust tracking with Kalman filtering
- Configurable confidence thresholds for different scenarios

### 2. Pose Extractor Module (`core/pose_extractor.py`)

**Purpose**: Extract pose keypoints and body structural features

**Key Classes**:
```python
class PoseExtractor:
    """MediaPipe-based pose estimation"""
    - __init__(model_complexity: int)
    - extract(frame: np.ndarray, bbox: BoundingBox) -> PoseFeatures
    
class BodyRatioCalculator:
    """Calculate body proportions from keypoints"""
    - calculate_ratios(keypoints: np.ndarray) -> Dict[str, float]
```

**Output Features**:
- 33 pose keypoints (MediaPipe format)
- Shoulder-to-hip ratio
- Leg-to-torso ratio
- Body height proportions
- Keypoint confidence scores

### 3. ReID Extractor Module (`core/reid_extractor.py`)

**Purpose**: Extract appearance embeddings for person re-identification

**Key Classes**:
```python
class ReIDExtractor:
    """TorchReID-based appearance embedding"""
    - __init__(model_name: str, weights_path: str)
    - extract(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray
    - preprocess(image: np.ndarray) -> torch.Tensor
```

**Model Options**:
- OSNet (lightweight, good for real-time)
- ResNet-50 (balanced accuracy/speed)
- Vision Transformer (highest accuracy, slower)

**Output**: 512-dimensional appearance embedding

### 4. Color Extractor Module (`core/color_extractor.py`)

**Purpose**: Extract color distribution features from clothing regions

**Key Classes**:
```python
class ColorExtractor:
    """HSV color histogram extraction"""
    - __init__(hist_bins: int, regions: List[str])
    - extract(frame: np.ndarray, bbox: BoundingBox, pose: PoseFeatures) -> np.ndarray
    - get_dominant_colors(histogram: np.ndarray) -> List[Tuple[int, int, int]]
```

**Output Features**:
- HSV histogram for upper body (shirt/jacket)
- HSV histogram for lower body (pants/skirt)
- Dominant color values (top 3 per region)

### 5. Feature Fusion Module (`core/feature_fusion.py`)

**Purpose**: Combine multiple modalities into unified embedding

**Key Classes**:
```python
class FeatureFusion:
    """Multi-modal feature fusion with dynamic weighting"""
    - __init__(weights: Dict[str, float], dimensions: Dict[str, int])
    - fuse(features: Dict[str, np.ndarray], confidences: Dict[str, float]) -> np.ndarray
    - normalize(vector: np.ndarray) -> np.ndarray
    - adjust_weights(occlusion_flags: Dict[str, bool]) -> Dict[str, float]
```

**Fusion Strategy**:
```
F_combined = [
    α * F_ReID (512-dim),
    β * F_Pose (64-dim),
    γ * F_Color (96-dim),
    δ * F_Accessory (32-dim),
    ε * F_Motion (16-dim)
]
```
Total: ~720-dimensional vector

**Dynamic Weight Adjustment**:
- Low pose confidence → reduce β, increase α
- Occlusion detected → reduce affected modality weights
- Night mode → reduce color weight γ

### 6. Memory Bank Module (`core/memory_bank.py`)

**Purpose**: Adaptive storage and retrieval of person feature vectors

**Key Classes**:
```python
class AdaptiveMemoryBank:
    """FAISS-backed memory with temporal decay"""
    - __init__(embedding_dim: int, decay_rate: float)
    - add_profile(person_id: int, features: np.ndarray, metadata: Dict)
    - update_profile(person_id: int, features: np.ndarray, metadata: Dict)
    - search(query: np.ndarray, top_k: int) -> List[MatchResult]
    - apply_temporal_decay()
    - get_profile(person_id: int) -> PersonProfile
    
class TemporalSnapshot:
    """Single temporal snapshot of a person"""
    - features: np.ndarray
    - timestamp: datetime
    - camera_id: str
    - confidence: float
```

**Key Algorithms**:

**Temporal Decay**:
```python
def calculate_weight(snapshot: TemporalSnapshot, current_time: datetime) -> float:
    delta_t = (current_time - snapshot.timestamp).total_seconds()
    return math.exp(-decay_rate * delta_t)
```

**Dynamic Update**:
```python
def update_profile(person_id: int, new_features: np.ndarray, confidence: float):
    if confidence > UPDATE_THRESHOLD:
        # Add new snapshot without removing old ones
        memory_bank.add_snapshot(person_id, new_features)
    else:
        # Average with existing snapshots
        existing = memory_bank.get_profile(person_id)
        updated = (existing * 0.7) + (new_features * 0.3)
        memory_bank.update_snapshot(person_id, updated)
```

**FAISS Configuration**:
- Index: IndexFlatIP (inner product for cosine similarity)
- Normalization: L2 normalization before indexing
- Batch size: 100 for bulk operations

### 7. Video Loader Module (`utils/video_loader.py`)

**Purpose**: Multi-camera video stream management

**Key Classes**:
```python
class VideoLoader:
    """Multi-camera video stream loader"""
    - __init__(sources: List[str])
    - __iter__() -> Iterator[Tuple[int, np.ndarray]]
    - get_frame(camera_id: int) -> Optional[np.ndarray]
    - release()
```

**Supported Sources**:
- Video files (MP4, AVI, MOV)
- RTSP streams
- Webcam devices
- Image sequences

### 8. Visualization Module (`utils/visualization.py`)

**Purpose**: Render bounding boxes, features, and match overlays

**Key Classes**:
```python
class Visualizer:
    """Video frame visualization"""
    - __init__(show_features: bool = True)
    - draw_bbox(frame: np.ndarray, bbox: BoundingBox, track_id: int)
    - draw_pose(frame: np.ndarray, keypoints: np.ndarray)
    - draw_match_info(frame: np.ndarray, match: MatchResult)
    - draw_features(frame: np.ndarray, features: Dict)
```

### 9. Dashboard Module (`dashboard/app.py`)

**Purpose**: Streamlit-based real-time monitoring interface

**Key Components**:
```python
# Main Pages
- Live Monitoring: Real-time video feeds with tracking
- Missing Person Search: Query memory bank with image/video
- Memory Bank Management: View/manage stored profiles
- Analytics: System performance metrics

# UI Components
- Video player with frame-by-frame navigation
- Match confidence visualization
- Feature contribution charts
- Timeline view across cameras
```

## Database Schema

### SQLite Schema (`database/schema.sql`)

```sql
-- Person profiles
CREATE TABLE persons (
    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    status TEXT DEFAULT 'missing',  -- missing, found, safe
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_camera TEXT,
    last_seen_time TIMESTAMP
);

-- Feature snapshots
CREATE TABLE feature_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    features BLOB,  -- Serialized numpy array
    camera_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence REAL,
    FOREIGN KEY (person_id) REFERENCES persons(person_id)
);

-- Match records
CREATE TABLE matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_person_id INTEGER,
    matched_person_id INTEGER,
    confidence REAL,
    match_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    camera_id TEXT,
    explanation TEXT,
    FOREIGN KEY (query_person_id) REFERENCES persons(person_id),
    FOREIGN KEY (matched_person_id) REFERENCES persons(person_id)
);

-- Accessory detections
CREATE TABLE accessories (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER,
    accessory_type TEXT,  -- backpack, cap, handbag, etc.
    confidence REAL,
    FOREIGN KEY (snapshot_id) REFERENCES feature_snapshots(snapshot_id)
);

-- Camera metadata
CREATE TABLE cameras (
    camera_id TEXT PRIMARY KEY,
    location TEXT,
    rtsp_url TEXT,
    is_active BOOLEAN DEFAULT TRUE
);
```

## Configuration Management

### Settings Structure (`config/settings.yaml`)

```yaml
# Detection settings
detection:
  model_path: "yolo11n.pt"
  confidence_threshold: 0.5
  nms_threshold: 0.45
  device: "cuda"  # or "cpu"

# Tracking settings
tracking:
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3

# Feature extraction
feature_extraction:
  reid:
    model_name: "osnet_x0_25"
    weights_path: "weights/osnet_x0_25_market1501.pth"
    embedding_dim: 512
  
  pose:
    model_complexity: 1  # 0, 1, or 2
    min_detection_confidence: 0.5
  
  color:
    hist_bins: 16
    regions: ["upper_body", "lower_body"]

# Memory bank
memory_bank:
  embedding_dim: 720
  decay_rate: 0.0001  # Per second
  max_snapshots_per_person: 10
  update_threshold: 0.8

# Feature fusion weights
fusion_weights:
  reid: 0.4
  pose: 0.2
  color: 0.15
  accessory: 0.15
  motion: 0.1

# Dashboard
dashboard:
  refresh_interval: 100  # ms
  max_display_cameras: 4
  show_confidence_threshold: 0.6

# Database
database:
  type: "sqlite"  # or "postgresql"
  path: "database/memorytrack.db"
  # For PostgreSQL:
  # host: "localhost"
  # port: 5432
  # name: "memorytrack"
  # user: "postgres"
  # password: "password"
```

## Error Handling Strategy

### Graceful Degradation
1. **GPU Unavailable**: Automatically fall back to CPU
2. **Model Loading Failure**: Use backup lightweight models
3. **Camera Connection Lost**: Continue with available cameras
4. **Feature Extraction Failure**: Use available modalities only

### Logging Strategy
```python
import logging

# Log levels
DEBUG: Detailed feature extraction logs
INFO: System status, tracking updates
WARNING: Degraded performance, fallback modes
ERROR: Critical failures requiring intervention
```

## Performance Optimization

### Parallel Processing
- Feature extraction runs in parallel across modalities
- Multi-camera processing using thread pool
- Batch FAISS operations for efficiency

### Memory Management
- Limit snapshots per person (LRU eviction)
- Periodic memory bank cleanup
- Efficient numpy array serialization

### GPU Utilization
- Batch inference for YOLO and ReID
- CUDA streams for overlapping operations
- Mixed precision training/inference (FP16)

## Testing Strategy

### Unit Tests
- Test each module independently
- Mock external dependencies (models, cameras)
- Validate algorithm correctness

### Integration Tests
- Test full pipeline with sample videos
- Validate feature fusion output dimensions
- Test memory bank CRUD operations

### Performance Tests
- Measure FPS on different hardware
- Profile memory usage
- Test scalability with large memory banks

## Security Considerations

1. **Data Privacy**: Encrypt stored feature vectors
2. **Access Control**: Role-based dashboard access
3. **Input Validation**: Validate all video inputs
4. **SQL Injection**: Use parameterized queries
5. **Model Security**: Verify model checksums

## Deployment Architecture

### Development
- Local execution with sample videos
- SQLite database
- CPU-only mode

### Production
- GPU server for inference
- PostgreSQL database
- Multi-camera RTSP streams
- Load balancing for multiple instances
