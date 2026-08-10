"""
Simple test dashboard to diagnose issues.
"""

import streamlit as st

st.set_page_config(
    page_title="MemoryTrack Test",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 MemoryTrack - Test Dashboard")

st.write("If you can see this, Streamlit is working correctly.")

st.header("Test Components")

# Test basic imports
try:
    import numpy as np
    st.success("✅ NumPy imported successfully")
except ImportError as e:
    st.error(f"❌ NumPy import failed: {e}")

try:
    import cv2
    st.success("✅ OpenCV imported successfully")
except ImportError as e:
    st.error(f"❌ OpenCV import failed: {e}")

try:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    st.success("✅ Path configured successfully")
except Exception as e:
    st.error(f"❌ Path configuration failed: {e}")

# Test core imports
try:
    from core.memory_bank import AdaptiveMemoryBank
    st.success("✅ Memory bank imported successfully")
    
    # Test basic functionality
    bank = AdaptiveMemoryBank(embedding_dim=720)
    stats = bank.get_statistics()
    st.write(f"Memory bank stats: {stats}")
except Exception as e:
    st.error(f"❌ Memory bank test failed: {e}")

st.header("Interactive Test")
if st.button("Test Button"):
    st.write("Button clicked successfully!")

st.info("If you see all green checkmarks, the system is working correctly.")
