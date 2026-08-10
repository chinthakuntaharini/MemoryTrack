#!/usr/bin/env python3
"""
MemoryTrack CLI Runner - Lightweight version for demo purposes
"""

import argparse
import sys
import os
import logging
from pathlib import Path

def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='MemoryTrack - Adaptive Memory-Based Multi-Camera Missing Person Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --demo                              # Run system demo
  %(prog)s --test                              # Run basic tests  
  %(prog)s --dashboard                         # Start Streamlit dashboard
  %(prog)s --video input.mp4                  # Process video file
  %(prog)s --webcam 0                         # Process webcam feed
  %(prog)s --config config/custom.yaml        # Use custom config
  %(prog)s --video input.mp4 --output out.mp4 # Save processed output

For more information, see README.md
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--video', type=str, help='Input video file path')
    input_group.add_argument('--webcam', type=int, help='Webcam device ID (usually 0)')
    input_group.add_argument('--rtsp', type=str, help='RTSP stream URL')
    input_group.add_argument('--demo', action='store_true', help='Run system demo')
    input_group.add_argument('--test', action='store_true', help='Run basic tests')
    input_group.add_argument('--dashboard', action='store_true', help='Start Streamlit dashboard')
    
    # Processing options
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--output', type=str, help='Output video file path')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Processing device (default: auto)')
    parser.add_argument('--no-display', action='store_true', help='Run without video display')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing')
    
    # Debug options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    return parser

def run_demo():
    """Run the system demo."""
    print("🎯 Running MemoryTrack Demo...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, 'demo.py'], 
                              capture_output=True, text=True, timeout=60)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Demo timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"Demo failed: {e}")
        return False

def run_tests():
    """Run basic tests."""
    print("🧪 Running Basic Tests...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, 'test_basic.py'], 
                              capture_output=True, text=True, timeout=120)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Tests timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"Tests failed: {e}")
        return False

def run_dashboard():
    """Start the Streamlit dashboard."""
    print("📊 Starting MemoryTrack Dashboard...")
    print("Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    
    try:
        import subprocess
        # Use streamlit run directly
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'dashboard/app.py', 
            '--server.port=8501'
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"Dashboard failed to start: {e}")
        return False
    
    return True

def run_full_pipeline(args):
    """Run the full MemoryTrack pipeline."""
    print("🚀 Starting MemoryTrack Pipeline...")
    print("⚠️  Note: This requires full ML dependencies to be installed")
    
    try:
        # Try to import the full pipeline
        from main import MemoryTrackPipeline
        
        # Initialize pipeline
        pipeline = MemoryTrackPipeline(args.config)
        
        # Process based on input type
        if args.video:
            print(f"📹 Processing video: {args.video}")
            pipeline.process_video(args.video, args.output)
        elif args.webcam is not None:
            print(f"📸 Processing webcam: {args.webcam}")
            pipeline.process_webcam(args.webcam, args.output)
        elif args.rtsp:
            print(f"📡 Processing RTSP stream: {args.rtsp}")
            pipeline.process_rtsp(args.rtsp, args.output)
        else:
            print("❌ No input source specified")
            return False
            
    except ImportError as e:
        print(f"❌ Pipeline dependencies not available: {e}")
        print("💡 Try installing full dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return False
    
    return True

def check_system_status():
    """Check system status and available features."""
    print("🔍 MemoryTrack System Status")
    print("=" * 50)
    
    # Check Python version
    print(f"Python: {sys.version}")
    
    # Check core dependencies
    dependencies = {
        'numpy': 'Core numerical operations',
        'opencv-cv2': 'Computer vision (as cv2)',
        'streamlit': 'Dashboard interface',
        'sqlalchemy': 'Database operations',
        'pyyaml': 'Configuration management',
        'faiss': 'Vector similarity search',
        'torch': 'Deep learning framework',
        'ultralytics': 'YOLO detection',
        'mediapipe': 'Pose estimation'
    }
    
    available = []
    missing = []
    
    for pkg, description in dependencies.items():
        try:
            if pkg == 'opencv-cv2':
                import cv2
            elif pkg == 'faiss':
                import faiss
            else:
                __import__(pkg)
            available.append(f"✅ {pkg}: {description}")
        except ImportError:
            missing.append(f"❌ {pkg}: {description}")
    
    print("\n📦 Available Dependencies:")
    for dep in available:
        print(f"  {dep}")
    
    if missing:
        print("\n⚠️  Missing Dependencies:")
        for dep in missing:
            print(f"  {dep}")
        print("\n💡 Install with: pip install -r requirements.txt")
    
    # Check project structure
    required_files = [
        'config/settings.yaml',
        'database/schema.sql', 
        'dashboard/app.py',
        'core/__init__.py',
        'utils/__init__.py'
    ]
    
    print(f"\n📁 Project Structure:")
    for file in required_files:
        status = "✅" if os.path.exists(file) else "❌"
        print(f"  {status} {file}")
    
    print(f"\n🎯 Available Features:")
    print(f"  ✅ Demo mode (--demo)")
    print(f"  ✅ Basic tests (--test)")
    print(f"  ✅ Dashboard (--dashboard)")
    
    if len(missing) == 0:
        print(f"  ✅ Full pipeline (--video, --webcam, --rtsp)")
    else:
        print(f"  ⚠️  Full pipeline (install missing dependencies)")

def main():
    """Main entry point."""
    parser = create_parser()
    
    # If no arguments provided, show help and system status
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n")
        check_system_status()
        return 0
    
    args = parser.parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.debug else logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Route to appropriate function
        if args.demo:
            success = run_demo()
        elif args.test:
            success = run_tests()
        elif args.dashboard:
            success = run_dashboard()
        else:
            success = run_full_pipeline(args)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())