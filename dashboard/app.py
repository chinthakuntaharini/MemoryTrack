"""
Streamlit Dashboard for MemoryTrack system.
Provides real-time monitoring, missing person search, and memory bank management.
"""

import streamlit as st
import cv2
import numpy as np
from typing import Optional, List
import tempfile
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="MemoryTrack Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import MemoryTrack components
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import with error handling
try:
    from utils.config_loader import ConfigLoader
    from utils.db_manager import DatabaseManager
    from core.memory_bank import AdaptiveMemoryBank
    UTILS_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing core components: {e}")
    UTILS_AVAILABLE = False

# Heavy components - import only when needed
DETECTOR_AVAILABLE = False


# Initialize session state
if UTILS_AVAILABLE:
    if 'memory_bank' not in st.session_state:
        st.session_state.memory_bank = AdaptiveMemoryBank(embedding_dim=720)

    if 'db' not in st.session_state:
        st.session_state.db = DatabaseManager()

    if 'config' not in st.session_state:
        st.session_state.config = ConfigLoader()
else:
    st.error("Core components not available. Please check installation.")


def main():
    """Main dashboard application."""
    st.title("🔍 MemoryTrack - Missing Person Tracking System")
    
    if not UTILS_AVAILABLE:
        st.error("Dashboard cannot function without core components. Please check the installation.")
        st.info("Run: pip install -r requirements.txt")
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["Live Monitoring", "Missing Person Search", "Memory Bank", "Analytics"]
        )
        
        st.header("System Status")
        try:
            stats = st.session_state.memory_bank.get_statistics()
            st.metric("Total Profiles", stats['total_profiles'])
            st.metric("Total Snapshots", stats['total_snapshots'])
            st.metric("Avg Snapshots/Profile", f"{stats['avg_snapshots_per_profile']:.1f}")
            
            st.header("Status Distribution")
            if stats['status_distribution']:
                for status, count in stats['status_distribution'].items():
                    st.write(f"{status}: {count}")
        except Exception as e:
            st.error(f"Error getting statistics: {e}")
    
    # Render selected page
    if page == "Live Monitoring":
        render_live_monitoring()
    elif page == "Missing Person Search":
        render_missing_person_search()
    elif page == "Memory Bank":
        render_memory_bank()
    elif page == "Analytics":
        render_analytics()


def render_live_monitoring():
    """Render live monitoring page."""
    st.header("📹 Live Monitoring")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Video Feed")
        
        # Video source selection
        video_source = st.selectbox(
            "Select Video Source",
            ["Upload Video File", "Webcam", "Sample Video"]
        )
        
        if video_source == "Upload Video File":
            uploaded_file = st.file_uploader(
                "Upload a video file",
                type=['mp4', 'avi', 'mov']
            )
            
            if uploaded_file:
                # Save uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
                    f.write(uploaded_file.read())
                    video_path = f.name
                
                st.success(f"Video uploaded: {uploaded_file.name}")
                
                # Process video button
                if st.button("Start Processing"):
                    process_video_display(video_path)
        
        elif video_source == "Webcam":
            if st.button("Start Webcam"):
                process_webcam_display()
        
        elif video_source == "Sample Video":
            st.info("Upload a sample video to see the processing pipeline in action.")
    
    with col2:
        st.subheader("Detection Info")
        
        # Placeholder for detection statistics
        st.metric("Active Tracks", "0")
        st.metric("FPS", "0")
        st.metric("Persons Detected", "0")
        
        st.subheader("Recent Matches")
        # Placeholder for recent matches
        st.write("No recent matches")


def process_video_display(video_path: str):
    """Process and display video with tracking."""
    st.info("Processing video... This may take a few moments.")
    
    # Import heavy components only when needed
    try:
        from core.detector import PersonDetector, MultiObjectTracker
        from core.pose_extractor import PoseExtractor
        from core.reid_extractor import ReIDExtractor
        from core.color_extractor import ColorExtractor
        from core.feature_fusion import FeatureFusion
        
        # Initialize components
        detector = PersonDetector(model_path="yolo11n.pt", device="cpu")
        tracker = MultiObjectTracker()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Failed to open video file")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.info(f"Video: {total_frames} frames at {fps:.1f} FPS")
        
        # Process frames
        frame_count = 0
        progress_bar = st.progress(0)
        
        # Placeholder for video display
        video_placeholder = st.empty()
        
        while cap.isOpened() and frame_count < 100:  # Limit to 100 frames for demo
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Detection
            detections = detector.detect(frame)
            
            # Tracking
            tracks = tracker.update(detections)
            
            # Draw bounding boxes
            for track_id, track in tracks.items():
                x1, y1, x2, y2 = int(track.bbox.x1), int(track.bbox.y1), int(track.bbox.x2), int(track.bbox.y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            frame_count += 1
            progress_bar.progress(frame_count / min(100, total_frames))
            
            # Small delay for display
            time.sleep(0.05)
        
        cap.release()
        detector = None  # Cleanup
        
        st.success(f"Processed {frame_count} frames")
        
    except ImportError as e:
        st.error(f"Missing required components: {e}")
        st.info("Please install all dependencies: pip install -r requirements.txt")
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")


def process_webcam_display():
    """Process and display webcam feed."""
    st.info("Starting webcam... Press 'q' in the video window to stop.")
    
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Failed to open webcam")
            return
        
        video_placeholder = st.empty()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Simple detection placeholder
            # In real implementation, use detector
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        st.success("Webcam stopped")
        
    except Exception as e:
        st.error(f"Error with webcam: {str(e)}")


def render_missing_person_search():
    """Render missing person search page."""
    st.header("🔎 Missing Person Search")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Query Input")
        
        query_type = st.radio(
            "Query Type",
            ["Upload Image", "Upload Video", "Person ID"]
        )
        
        if query_type == "Upload Image":
            uploaded_image = st.file_uploader(
                "Upload an image of the missing person",
                type=['jpg', 'jpeg', 'png']
            )
            
            if uploaded_image:
                st.image(uploaded_image, caption="Query Image", use_container_width=True)
                
                if st.button("Search"):
                    st.info("Searching memory bank...")
                    # In real implementation, extract features and search
                    time.sleep(1)
                    st.success("Search complete")
        
        elif query_type == "Upload Video":
            uploaded_video = st.file_uploader(
                "Upload a video of the missing person",
                type=['mp4', 'avi']
            )
            
            if uploaded_video:
                st.video(uploaded_video)
                
                if st.button("Search"):
                    st.info("Searching memory bank...")
                    time.sleep(1)
                    st.success("Search complete")
        
        elif query_type == "Person ID":
            person_id = st.number_input("Enter Person ID", min_value=1, value=1)
            
            if st.button("Search"):
                profile = st.session_state.memory_bank.get_profile(person_id)
                
                if profile:
                    st.success(f"Found profile: {profile.name or 'Unknown'}")
                    st.write(f"Status: {profile.status}")
                    st.write(f"Snapshots: {len(profile.snapshots)}")
                else:
                    st.warning("Profile not found")
    
    with col2:
        st.subheader("Search Results")
        
        # Placeholder for search results
        st.info("Upload a query to see search results")


def render_memory_bank():
    """Render memory bank management page."""
    st.header("💾 Memory Bank Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("All Profiles")
        
        # Get all profiles
        profiles = st.session_state.memory_bank.profiles
        
        if profiles:
            # Display profiles in a table
            profile_data = []
            for person_id, profile in profiles.items():
                profile_data.append({
                    "ID": person_id,
                    "Name": profile.name or "Unknown",
                    "Status": profile.status,
                    "Snapshots": len(profile.snapshots),
                    "Last Seen": profile.last_seen_camera or "N/A"
                })
            
            st.dataframe(profile_data)
        else:
            st.info("No profiles in memory bank")
        
        st.subheader("Add New Profile")
        
        with st.form("add_profile_form"):
            person_id = st.number_input("Person ID", min_value=1, value=1)
            name = st.text_input("Name (optional)")
            status = st.selectbox("Status", ["missing", "found", "safe"])
            
            submitted = st.form_submit_button("Add Profile")
            
            if submitted:
                # In real implementation, extract features from image
                features = np.random.rand(720).astype(np.float32)
                
                st.session_state.memory_bank.add_profile(
                    person_id=person_id,
                    features=features,
                    camera_id="manual",
                    confidence=1.0,
                    name=name
                )
                
                st.session_state.memory_bank.update_status(person_id, status)
                
                st.success(f"Profile {person_id} added successfully")
                st.rerun()
    
    with col2:
        st.subheader("Quick Actions")
        
        # Update status
        st.write("Update Person Status")
        update_id = st.number_input("Person ID", min_value=1, key="update_id")
        new_status = st.selectbox("New Status", ["missing", "found", "safe"], key="new_status")
        
        if st.button("Update Status"):
            if st.session_state.memory_bank.update_status(update_id, new_status):
                st.success(f"Updated person {update_id} to {new_status}")
            else:
                st.warning("Person not found")
        
        # Delete profile
        st.write("Delete Profile")
        delete_id = st.number_input("Person ID", min_value=1, key="delete_id")
        
        if st.button("Delete Profile"):
            if st.session_state.memory_bank.delete_profile(delete_id):
                st.success(f"Deleted profile {delete_id}")
                st.rerun()
            else:
                st.warning("Person not found")


def render_analytics():
    """Render analytics page."""
    st.header("📊 System Analytics")
    
    # Memory bank statistics
    stats = st.session_state.memory_bank.get_statistics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Profiles", stats['total_profiles'])
    
    with col2:
        st.metric("Total Snapshots", stats['total_snapshots'])
    
    with col3:
        st.metric("Index Size", stats['index_size'])
    
    # Status distribution
    st.subheader("Status Distribution")
    
    if stats['status_distribution']:
        status_labels = list(stats['status_distribution'].keys())
        status_values = list(stats['status_distribution'].values())
        
        st.bar_chart(dict(zip(status_labels, status_values)))
    else:
        st.info("No status data available")
    
    # Memory bank configuration
    st.subheader("Memory Bank Configuration")
    
    st.json({
        "embedding_dim": stats['embedding_dim'] if 'embedding_dim' in stats else 720,
        "decay_rate": stats['decay_rate'],
        "max_snapshots_per_person": stats['max_snapshots_per_person']
    })
    
    # Database statistics
    st.subheader("Database Statistics")
    
    try:
        persons = st.session_state.db.list_persons()
        st.metric("Database Profiles", len(persons))
        
        recent_matches = st.session_state.db.get_recent_matches(limit=10)
        st.metric("Recent Matches", len(recent_matches))
    except Exception as e:
        st.warning(f"Could not fetch database statistics: {str(e)}")


if __name__ == "__main__":
    main()
