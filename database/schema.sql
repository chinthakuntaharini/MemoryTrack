-- MemoryTrack Database Schema
-- SQLite Schema (compatible with PostgreSQL with minor modifications)

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Person profiles table
CREATE TABLE IF NOT EXISTS persons (
    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    status TEXT DEFAULT 'missing' CHECK(status IN ('missing', 'found', 'safe')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_camera TEXT,
    last_seen_time TIMESTAMP,
    notes TEXT
);

-- Feature snapshots table
CREATE TABLE IF NOT EXISTS feature_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    features BLOB,  -- Serialized numpy array (720-dimensional vector)
    camera_id TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence REAL NOT NULL,
    frame_number INTEGER,
    FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE
);

-- Individual modality features (for XAI explanation)
CREATE TABLE IF NOT EXISTS modality_features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL,
    modality TEXT NOT NULL CHECK(modality IN ('reid', 'pose', 'color', 'accessory', 'motion')),
    features BLOB NOT NULL,
    confidence REAL DEFAULT 1.0,
    FOREIGN KEY (snapshot_id) REFERENCES feature_snapshots(snapshot_id) ON DELETE CASCADE
);

-- Accessory detections table
CREATE TABLE IF NOT EXISTS accessories (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL,
    accessory_type TEXT NOT NULL,  -- backpack, cap, handbag, umbrella, suitcase, bottle
    confidence REAL NOT NULL,
    bbox_x1 REAL,
    bbox_y1 REAL,
    bbox_x2 REAL,
    bbox_y2 REAL,
    FOREIGN KEY (snapshot_id) REFERENCES feature_snapshots(snapshot_id) ON DELETE CASCADE
);

-- Match records table
CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_person_id INTEGER,
    matched_person_id INTEGER NOT NULL,
    confidence REAL NOT NULL,
    match_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    camera_id TEXT NOT NULL,
    explanation TEXT,
    reid_similarity REAL,
    pose_similarity REAL,
    color_similarity REAL,
    accessory_similarity REAL,
    motion_similarity REAL,
    FOREIGN KEY (query_person_id) REFERENCES persons(person_id) ON DELETE SET NULL,
    FOREIGN KEY (matched_person_id) REFERENCES persons(person_id) ON DELETE CASCADE
);

-- Camera metadata table
CREATE TABLE IF NOT EXISTS cameras (
    camera_id TEXT PRIMARY KEY,
    location TEXT,
    rtsp_url TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tracking history table
CREATE TABLE IF NOT EXISTS tracking_history (
    track_id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    camera_id TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    frame_count INTEGER DEFAULT 0,
    avg_confidence REAL,
    FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE SET NULL
);

-- System events table (for debugging and monitoring)
CREATE TABLE IF NOT EXISTS system_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    camera_id TEXT,
    details TEXT,
    severity TEXT DEFAULT 'INFO' CHECK(severity IN ('DEBUG', 'INFO', 'WARNING', 'ERROR'))
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_snapshots_person_id ON feature_snapshots(person_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON feature_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_snapshots_camera ON feature_snapshots(camera_id);
CREATE INDEX IF NOT EXISTS idx_matches_query_person ON matches(query_person_id);
CREATE INDEX IF NOT EXISTS idx_matches_matched_person ON matches(matched_person_id);
CREATE INDEX IF NOT EXISTS idx_matches_time ON matches(match_time);
CREATE INDEX IF NOT EXISTS idx_tracking_person ON tracking_history(person_id);
CREATE INDEX IF NOT EXISTS idx_tracking_camera ON tracking_history(camera_id);
CREATE INDEX IF NOT EXISTS idx_events_time ON system_events(event_time);

-- Create trigger to update updated_at timestamp on persons
CREATE TRIGGER IF NOT EXISTS update_person_timestamp
AFTER UPDATE ON persons
BEGIN
    UPDATE persons SET updated_at = CURRENT_TIMESTAMP WHERE person_id = NEW.person_id;
END;
