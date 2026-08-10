#!/usr/bin/env python3
"""
Simple MemoryTrack Demo - ASCII only for Windows compatibility
"""

import sys
import os
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_configuration():
    """Test configuration loading."""
    print("Configuration Test:")
    try:
        from utils.config_loader import ConfigLoader
        config = ConfigLoader()
        det_config = config.get_detection_config()
        print(f"  Model: {det_config.get('model_path', 'yolo11n.pt')}")
        print(f"  Confidence: {det_config.get('confidence_threshold', 0.5)}")
        print("  [PASS] Configuration loaded")
        return True
    except Exception as e:
        print(f"  [FAIL] Configuration test: {e}")
        return False

def test_database():
    """Test database operations."""
    print("Database Test:")
    try:
        from utils.db_manager import DatabaseManager
        import numpy as np
        
        db = DatabaseManager(":memory:")
        db.init_db()
        print("  [PASS] Database initialized")
        
        # Create test person
        person_data = {'name': 'Test Person', 'status': 'missing'}
        person_id = db.create_person(person_data)
        print(f"  [PASS] Created person ID: {person_id}")
        
        # Add feature snapshot
        features = np.random.random(720).astype(np.float32)
        snapshot_data = {
            'person_id': person_id,
            'features': features,
            'camera_id': 'test_cam',
            'confidence': 0.9
        }
        snapshot_id = db.add_feature_snapshot(snapshot_data)
        print(f"  [PASS] Added snapshot ID: {snapshot_id}")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Database test: {e}")
        return False

def test_memory_bank():
    """Test memory bank operations."""
    print("Memory Bank Test:")
    try:
        from core.memory_bank import AdaptiveMemoryBank
        from utils.db_manager import DatabaseManager
        import numpy as np
        
        db = DatabaseManager(":memory:")
        db.init_db()
        
        memory_bank = AdaptiveMemoryBank(embedding_dim=512, db_manager=db)
        print("  [PASS] Memory bank initialized")
        
        # Add test profiles
        for i in range(3):
            features = np.random.random(512).astype(np.float32)
            memory_bank.add_profile(
                person_id=i + 1,
                features=features,
                metadata={'camera_id': f'cam_{i + 1}'}
            )
        
        print("  [PASS] Added 3 test profiles")
        
        # Search test
        query = np.random.random(512).astype(np.float32)
        results = memory_bank.search(query, top_k=2)
        print(f"  [PASS] Search returned {len(results)} results")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Memory bank test: {e}")
        return False

def test_feature_fusion():
    """Test feature fusion."""
    print("Feature Fusion Test:")
    try:
        from core.feature_fusion import FeatureFusion
        import numpy as np
        
        fusion = FeatureFusion()
        
        features = {
            'reid': np.random.random(512).astype(np.float32),
            'pose': np.random.random(64).astype(np.float32),
            'color': np.random.random(96).astype(np.float32)
        }
        
        confidences = {'reid': 0.9, 'pose': 0.8, 'color': 0.7}
        
        result = fusion.fuse(features, confidences)
        print(f"  [PASS] Fused {len(result.fused_features)} dimensions")
        print(f"  [PASS] Overall confidence: {result.overall_confidence:.3f}")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Feature fusion test: {e}")
        return False

def main():
    """Run simple demo."""
    print("MemoryTrack Simple Demo")
    print("=" * 40)
    print("Testing core functionality...\n")
    
    tests = [
        ("Configuration", test_configuration),
        ("Database", test_database),
        ("Memory Bank", test_memory_bank),
        ("Feature Fusion", test_feature_fusion)
    ]
    
    passed = 0
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
        print()
    
    print("=" * 40)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("SUCCESS: All core features working!")
        print("\nNext steps:")
        print("1. Start dashboard: python run_memorytrack.py --dashboard")
        print("2. View at: http://localhost:8501")
    else:
        print("Some tests failed. Check output above.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)