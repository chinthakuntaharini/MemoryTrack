#!/usr/bin/env python3
"""
Basic test script for MemoryTrack - tests core functionality without heavy ML dependencies.
This script can run in CI environments where torch/ultralytics/etc. are not available.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that core modules can be imported."""
    print("Testing core imports...")
    
    try:
        from utils.config_loader import ConfigLoader
        print("✅ ConfigLoader imported successfully")
    except ImportError as e:
        print(f"❌ ConfigLoader import failed: {e}")
        return False
    
    try:
        from utils.db_manager import DatabaseManager
        print("✅ DatabaseManager imported successfully")
    except ImportError as e:
        print(f"❌ DatabaseManager import failed: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from utils.config_loader import ConfigLoader
        config = ConfigLoader()
        
        # Test basic config sections
        assert config.get_detection_config() is not None
        assert config.get_tracking_config() is not None
        assert config.get_feature_extraction_config() is not None
        assert config.get_memory_bank_config() is not None
        
        print("✅ Configuration loaded and validated")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_database():
    """Test database initialization."""
    print("\nTesting database...")
    
    try:
        from utils.db_manager import DatabaseManager
        
        # Use in-memory database for testing
        db = DatabaseManager(":memory:")
        db.init_db()
        
        # Test basic operations
        person_data = {
            'name': 'Test Person',
            'status': 'missing',
            'notes': 'Test case'
        }
        
        person_id = db.create_person(person_data)
        assert person_id is not None
        
        person = db.get_person(person_id)
        assert person is not None
        assert person['name'] == 'Test Person'
        
        print("✅ Database operations successful")
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_core_modules():
    """Test that core modules have expected classes."""
    print("\nTesting core module structure...")
    
    try:
        # Test that we can import the main classes (even if we can't instantiate them)
        import core
        
        # Check that key classes are available
        expected_classes = [
            'PersonDetector', 'MultiObjectTracker', 'BoundingBox', 'Track',
            'PoseExtractor', 'PoseFeatures', 'BodyRatioCalculator',
            'ReIDExtractor', 'ColorExtractor', 'ColorFeatures',
            'FeatureFusion', 'AdaptiveMemoryBank', 'PersonProfile',
            'TemporalSnapshot', 'MatchResult'
        ]
        
        available_classes = []
        for class_name in expected_classes:
            if hasattr(core, class_name):
                available_classes.append(class_name)
        
        print(f"✅ {len(available_classes)}/{len(expected_classes)} core classes available")
        
        if len(available_classes) >= len(expected_classes) * 0.8:  # 80% threshold
            return True
        else:
            print("❌ Too many core classes missing")
            return False
            
    except Exception as e:
        print(f"❌ Core module test failed: {e}")
        return False

def test_project_structure():
    """Test that key project files exist."""
    print("\nTesting project structure...")
    
    required_files = [
        'README.md',
        'requirements.txt',
        'config/settings.yaml',
        'database/schema.sql',
        'main.py',
    ]
    
    required_dirs = [
        'core',
        'utils',
        'dashboard',
        'tests',
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if not os.path.isdir(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
    
    if not missing_files and not missing_dirs:
        print("✅ All required files and directories present")
        return True
    else:
        return False

def main():
    """Run all basic tests."""
    print("🚀 MemoryTrack Basic Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration), 
        ("Database Test", test_database),
        ("Core Modules Test", test_core_modules),
        ("Project Structure Test", test_project_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MemoryTrack basic functionality is working.")
        sys.exit(0)
    else:
        print(f"⚠️  {total - passed} tests failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()