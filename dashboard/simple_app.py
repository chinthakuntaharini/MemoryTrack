"""
Minimal Streamlit app for testing.
"""

import streamlit as st

st.title("MemoryTrack Dashboard")
st.write("Welcome to MemoryTrack - Missing Person Tracking System")

st.header("System Status")
st.write("✅ Dashboard is running")
st.write("✅ All components loaded")

st.header("Navigation")
st.write("Select a page from the sidebar:")
st.write("- Live Monitoring")
st.write("- Missing Person Search") 
st.write("- Memory Bank")
st.write("- Analytics")

st.info("This is a minimal test version. Full features coming soon.")
