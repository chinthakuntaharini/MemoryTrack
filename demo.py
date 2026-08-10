#!/usr/bin/env python3
"""
MemoryTrack Demo Script
Simple demonstration of the MemoryTrack system functionality.
"""

import sys
import os
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def demo_configuration():
    """Demonstrate configuration loading."""
    print("🔧 Configuration Loading Demo")
    print("-" * 40)
    
    try:
        from utils.config_loader import ConfigLoader
        
        config = ConfigLoader()
        print("✅ Configuration loaded successfully")
        
        # Show some config values
        det_config = config.get_detection_config()
        print(f"Detection model: {det_config.get('model_path', 'yolo11n.pt')}")
        print(f"Confidence threshold: {det_config.get('confidence_threshold', 0.5)}")
        
        memory_config = config.get_memory_bank_config()
        print(f"Memory bank embedding dim: {memory_config.get('embedding_dim', 720)}")
        print(f"Decay rate: {memory_config.get('decay_rate', 0.0001)}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration demo failed: {e}")
        return False

def demo_database():
    """Demonstrate database operations."""
    print("\n🗄️  Database Operations Demo")
    print("-" * 40)
    
    try:
        from utils.db_manager import DatabaseManager
        import numpy as np
        
        # Use in-memory database for demo
        db = DatabaseManager(":memory:")
        db.init_db()
        print("✅ Database initialized")
        
        # Create a test person
        person_data = {
            'name': 'John Doe',
            'status': 'missing',
            'notes': 'Demo person for testing'
        }
        
        person_id = db.create_person(person_data)
        print(f"✅ Created person with ID: {person_id}")
        
        # Add a feature snapshot
        features = np.random.random(720).astype(np.float32)
        snapshot_data = {
            'person_id': person_id,
            'features': features,
            'camera_id': 'demo_cam_001',
            'confidence': 0.95
        }
        
        snapshot_id = db.add_feature_snapshot(snapshot_data)
        print(f"✅ Added feature snapshot with ID: {snapshot_id}")
        
        # Retrieve person
        person = db.get_person(person_id)
        print(f"✅ Retrieved person: {person['name']} (Status: {person['status']})")
        
        # Get snapshots
        snapshots = db.get_person_snapshots(person_id)
        print(f"✅ Person has {len(snapshots)} feature snapshots")
        
        return True
    except Exception as e:
        print(f"❌ Database demo failed: {e}")
        return False

def demo_memory_bank():
    """Demonstrate memory bank operations."""
    print("\n🧠 Memory Bank Demo")
    print("-" * 40)
    
    try:
        from core.memory_bank import AdaptiveMemoryBank
        from utils.db_manager import DatabaseManager
        import numpy as np
        
        # Initialize memory bank with in-memory database
        db = DatabaseManager(":memory:")
        db.init_db()
        
        memory_bank = AdaptiveMemoryBank(
            embedding_dim=512,  # Smaller for demo
            db_manager=db,
            decay_rate=0.001
        )
        print("✅ Memory bank initialized")
        
        # Add some test profiles
        for i in range(5):
            features = np.random.random(512).astype(np.float32)
            memory_bank.add_profile(
                person_id=i + 1,
                features=features,
                metadata={
                    'camera_id': f'cam_{i % 3 + 1}',
                    'timestamp': f'2024-01-01T{10+i}:00:00'
                }
            )
        
        print("✅ Added 5 test profiles to memory bank")
        
        # Perform a search
        query_features = np.random.random(512).astype(np.float32)
        results = memory_bank.search(query_features, top_k=3)
        
        print(f"✅ Search returned {len(results)} matches:")
        for i, result in enumerate(results, 1):
            print(f"   {i}. Person ID: {result.person_id}, Similarity: {result.similarity:.3f}")
        
        # Get memory bank statistics
        stats = memory_bank.get_statistics()
        print(f"✅ Memory bank statistics:")
        print(f"   Total profiles: {stats.get('total_profiles', 0)}")
        print(f"   Total snapshots: {stats.get('total_snapshots', 0)}")
        
        return True
    except Exception as e:
        print(f"❌ Memory bank demo failed: {e}")
        return False

def demo_feature_fusion():
    """Demonstrate feature fusion without heavy ML dependencies."""
    print("\n🔗 Feature Fusion Demo")
    print("-" * 40)
    
    try:
        from core.feature_fusion import FeatureFusion
        import numpy as np
        
        fusion = FeatureFusion()
        print("✅ Feature fusion initialized")
        
        # Create mock features
        features = {
            'reid': np.random.random(512).astype(np.float32),
            'pose': np.random.random(64).astype(np.float32),
            'color': np.random.random(96).astype(np.float32),
            'accessory': np.random.random(32).astype(np.float32),
            'motion': np.random.random(16).astype(np.float32)
        }
        
        confidences = {
            'reid': 0.9,
            'pose': 0.8,
            'color': 0.7,
            'accessory': 0.6,
            'motion': 0.85
        }
        
        print("✅ Created mock features for all modalities")
        
        # Fuse features
        result = fusion.fuse(features, confidences)
        
        print(f"✅ Feature fusion completed:")
        print(f"   Fused vector dimension: {len(result.fused_features)}")
        print(f"   Overall confidence: {result.overall_confidence:.3f}")
        print(f"   Modality weights: {result.modality_weights}")
        
        return True
    except Exception as e:
        print(f"❌ Feature fusion demo failed: {e}")
        return False

def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("\n🎨 Visualization Demo")
    print("-" * 40)
    
    try:
        from utils.visualization import Visualizer
        from core.detector import BoundingBox
        import numpy as np
        
        # Create a mock frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (50, 100, 150)  # Fill with blue-ish color
        
        visualizer = Visualizer()
        print("✅ Visualizer initialized")
        
        # Create mock bounding box
        bbox = BoundingBox(
            x1=100, y1=100, x2=300, y2=400,
            confidence=0.85, class_id=0
        )
        
        # Visualize (this would normally draw on the frame)
        print("✅ Mock bounding box created for visualization")
        print(f"   Bbox: ({bbox.x1}, {bbox.y1}) to ({bbox.x2}, {bbox.y2})")
        print(f"   Confidence: {bbox.confidence:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Visualization demo failed: {e}")
        return False

def demo_dashboard_info():
    """Show information about the dashboard."""
    print("\n📊 Dashboard Information")
    print("-" * 40)
    
    print("✅ Streamlit dashboard is available at:")
    print("   Command: streamlit run dashboard/app.py")
    print("   URL: http://localhost:8501")
    print("")
    print("📋 Dashboard features:")
    print("   • Live monitoring with multi-camera support")
    print("   • Missing person search interface")
    print("   • Memory bank management")
    print("   • Analytics and system metrics")
    print("   • Real-time visualization")
    
    return True

def main():
    """Run the complete demo."""
    print("🎯 MemoryTrack System Demo")
    print("=" * 50)
    print("This demo showcases the core functionality of MemoryTrack")
    print("without requiring heavy ML dependencies.\n")
    
    demos = [
        ("Configuration System", demo_configuration),
        ("Database Operations", demo_database),
        ("Memory Bank", demo_memory_bank),
        ("Feature Fusion", demo_feature_fusion),
        ("Visualization", demo_visualization),
        ("Dashboard Info", demo_dashboard_info),
    ]
    
    successful = 0
    total = len(demos)
    
    for demo_name, demo_func in demos:
        try:
            if demo_func():
                successful += 1
                print(f"✅ {demo_name} demo completed successfully")
            else:
                print(f"❌ {demo_name} demo failed")
        except Exception as e:
            print(f"❌ {demo_name} demo failed with exception: {e}")
        
        time.sleep(0.5)  # Small delay for readability
    
    print("\n" + "=" * 50)
    print(f"📊 Demo Results: {successful}/{total} demos completed successfully")
    
    if successful == total:
        print("🎉 All demos completed! MemoryTrack is working correctly.")
        print("\n🚀 Next steps:")
        print("   1. Install full dependencies: pip install -r requirements.txt")
        print("   2. Run the dashboard: streamlit run dashboard/app.py")
        print("   3. Process videos: python main.py --help")
    else:
        print(f"⚠️  {total - successful} demos failed. Check the output above.")
    
    return successful == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)