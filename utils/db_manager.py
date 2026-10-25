"""
Database manager for MemoryTrack system.
Handles SQLite/PostgreSQL database operations for storing person profiles,
feature snapshots, and match records.
"""

import sqlite3
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage database operations for MemoryTrack."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default path.
        """
        if db_path is None:
            project_root = Path(__file__).parent.parent
            db_path = project_root / "database" / "memorytrack.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.connection: Optional[sqlite3.Connection] = None
        self.connect()
    
    def connect(self) -> None:
        """Establish database connection and ensure tables exist."""
        try:
            self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            self._ensure_schema()
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _ensure_schema(self) -> None:
        """Ensure all required tables and investigative columns exist."""
        if not self.connection:
            return
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen_camera TEXT,
                    last_seen_time TIMESTAMP,
                    upper_color TEXT,
                    lower_color TEXT,
                    accessories TEXT,
                    posture TEXT,
                    notes TEXT
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feature_snapshots (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    features BLOB,
                    camera_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL NOT NULL,
                    frame_number INTEGER,
                    FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    camera_id TEXT,
                    details TEXT,
                    severity TEXT DEFAULT 'INFO'
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_person_id INTEGER,
                    matched_person_id INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    match_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    camera_id TEXT NOT NULL,
                    explanation TEXT
                );
            """)
            # Check for newly added columns if table already existed
            cursor.execute("PRAGMA table_info(persons)")
            cols = [row['name'] for row in cursor.fetchall()]
            for col in ['upper_color', 'lower_color', 'accessories', 'posture']:
                if col not in cols:
                    try:
                        cursor.execute(f"ALTER TABLE persons ADD COLUMN {col} TEXT")
                    except Exception:
                        pass
            self.connection.commit()
        except Exception as e:
            logger.warning(f"Error ensuring schema: {e}")
    
    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def init_db(self, schema_path: Optional[str] = None) -> None:
        """
        Initialize database with schema.
        
        Args:
            schema_path: Path to SQL schema file. If None, uses default path.
        """
        if schema_path is None:
            project_root = Path(__file__).parent.parent
            schema_path = project_root / "database" / "schema.sql"
        
        if not Path(schema_path).exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        try:
            cursor = self.connection.cursor()
            cursor.executescript(schema_sql)
            self.connection.commit()
            logger.info("Database initialized with schema")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    # Person Profile Operations
    
    def add_person(self, name: Optional[str] = None, status: str = "active",
                   notes: Optional[str] = None,
                   upper_color: Optional[str] = None,
                   lower_color: Optional[str] = None,
                   accessories: Optional[str] = None,
                   posture: Optional[str] = None) -> int:
        """
        Add a new person profile with appearance attributes.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO persons (name, status, notes, upper_color, lower_color, accessories, posture)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (name, status, notes, upper_color, lower_color, accessories, posture)
        )
        self.connection.commit()
        person_id = cursor.lastrowid
        logger.info(f"Added person profile: {person_id}")
        return person_id

    def add_or_update_tracked_person(self, person_id: int,
                                     name: Optional[str] = None,
                                     status: str = "active",
                                     last_seen_camera: str = "cam_0",
                                     last_seen_time: Optional[str] = None,
                                     upper_color: Optional[str] = None,
                                     lower_color: Optional[str] = None,
                                     accessories: Optional[str] = None,
                                     posture: Optional[str] = None,
                                     notes: Optional[str] = None) -> int:
        """Insert or update a tracked person profile with forensic appearance attributes."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT person_id FROM persons WHERE person_id = ?", (person_id,))
        existing = cursor.fetchone()
        
        now_str = last_seen_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if existing:
            cursor.execute("""
                UPDATE persons SET
                    last_seen_camera = ?,
                    last_seen_time = ?,
                    upper_color = COALESCE(?, upper_color),
                    lower_color = COALESCE(?, lower_color),
                    accessories = COALESCE(?, accessories),
                    posture = COALESCE(?, posture),
                    notes = COALESCE(?, notes),
                    updated_at = CURRENT_TIMESTAMP
                WHERE person_id = ?
            """, (last_seen_camera, now_str, upper_color, lower_color, accessories, posture, notes, person_id))
        else:
            cursor.execute("""
                INSERT INTO persons (person_id, name, status, last_seen_camera, last_seen_time, upper_color, lower_color, accessories, posture, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (person_id, name or f"Person #{person_id}", status, last_seen_camera, now_str, upper_color, lower_color, accessories, posture, notes))

        self.connection.commit()
        return person_id

    def create_person(self, person_data: Dict[str, Any]) -> int:
        """Convenience method to create a person from a dictionary."""
        return self.add_person(
            name=person_data.get('name', 'Unknown'),
            status=person_data.get('status', 'active'),
            notes=person_data.get('notes'),
            upper_color=person_data.get('upper_color'),
            lower_color=person_data.get('lower_color'),
            accessories=person_data.get('accessories'),
            posture=person_data.get('posture')
        )

    def add_feature_snapshot(self, snapshot_data: Dict[str, Any]) -> int:
        """Convenience method to add a snapshot from a dictionary."""
        return self.add_snapshot(
            person_id=snapshot_data['person_id'],
            features=snapshot_data['features'],
            camera_id=snapshot_data.get('camera_id', 'unknown'),
            confidence=snapshot_data.get('confidence', 1.0),
            frame_number=snapshot_data.get('frame_number')
        )
    
    def get_person(self, person_id: int) -> Optional[Dict[str, Any]]:
        """
        Get person profile by ID.
        
        Args:
            person_id: Person ID
            
        Returns:
            Person profile dictionary or None if not found
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM persons WHERE person_id = ?", (person_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def update_person_status(self, person_id: int, status: str,
                            last_seen_camera: Optional[str] = None,
                            last_seen_TIME: Optional[datetime] = None) -> None:
        """
        Update person status.
        
        Args:
            person_id: Person ID
            status: New status
            last_seen_camera: Last seen camera ID
            last_seen_time: Last seen timestamp
        """
        cursor = self.connection.cursor()
        
        if last_seen_camera and last_seen_time:
            cursor.execute(
                """
                UPDATE persons 
                SET status = ?, last_seen_camera = ?, last_seen_time = ?
                WHERE person_id = ?
                """,
                (status, last_seen_camera, last_seen_time, person_id)
            )
        else:
            cursor.execute(
                "UPDATE persons SET status = ? WHERE person_id = ?",
                (status, person_id)
            )
        
        self.connection.commit()
        logger.info(f"Updated person {person_id} status to {status}")
    
    def list_persons(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all person profiles, optionally filtered by status.
        
        Args:
            status: Filter by status (optional)
            
        Returns:
            List of person profiles
        """
        cursor = self.connection.cursor()
        
        if status:
            cursor.execute("SELECT * FROM persons WHERE status = ?", (status,))
        else:
            cursor.execute("SELECT * FROM persons")
        
        return [dict(row) for row in cursor.fetchall()]
    
    # Feature Snapshot Operations
    
    def add_snapshot(self, person_id: int, features: np.ndarray,
                    camera_id: str, confidence: float,
                    frame_number: Optional[int] = None) -> int:
        """
        Add a feature snapshot for a person.
        
        Args:
            person_id: Person ID
            features: Feature vector (numpy array)
            camera_id: Camera ID
            confidence: Detection confidence
            frame_number: Frame number (optional)
            
        Returns:
            snapshot_id of the created snapshot
        """
        cursor = self.connection.cursor()
        
        # Serialize numpy array to bytes
        features_blob = pickle.dumps(features)
        
        cursor.execute(
            """
            INSERT INTO feature_snapshots 
            (person_id, features, camera_id, confidence, frame_number)
            VALUES (?, ?, ?, ?, ?)
            """,
            (person_id, features_blob, camera_id, confidence, frame_number)
        )
        self.connection.commit()
        snapshot_id = cursor.lastrowid
        logger.debug(f"Added snapshot {snapshot_id} for person {person_id}")
        return snapshot_id
    
    def get_snapshot(self, snapshot_id: int) -> Optional[Dict[str, Any]]:
        """
        Get feature snapshot by ID.
        
        Args:
            snapshot_id: Snapshot ID
            
        Returns:
            Snapshot dictionary with deserialized features
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM feature_snapshots WHERE snapshot_id = ?", (snapshot_id,))
        row = cursor.fetchone()
        
        if row:
            snapshot = dict(row)
            # Deserialize features
            snapshot['features'] = pickle.loads(snapshot['features'])
            return snapshot
        return None
    
    def get_person_snapshots(self, person_id: int,
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all feature snapshots for a person.
        
        Args:
            person_id: Person ID
            limit: Maximum number of snapshots to return
            
        Returns:
            List of snapshot dictionaries with deserialized features
        """
        cursor = self.connection.cursor()
        
        if limit:
            cursor.execute(
                """
                SELECT * FROM feature_snapshots 
                WHERE person_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
                """,
                (person_id, limit)
            )
        else:
            cursor.execute(
                """
                SELECT * FROM feature_snapshots 
                WHERE person_id = ? 
                ORDER BY timestamp DESC
                """,
                (person_id,)
            )
        
        snapshots = []
        for row in cursor.fetchall():
            snapshot = dict(row)
            snapshot['features'] = pickle.loads(snapshot['features'])
            snapshots.append(snapshot)
        
        return snapshots
    
    def delete_old_snapshots(self, days: int = 30) -> int:
        """
        Delete snapshots older than specified days.
        
        Args:
            days: Number of days to keep snapshots
            
        Returns:
            Number of deleted snapshots
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            DELETE FROM feature_snapshots 
            WHERE timestamp < datetime('now', '-' || ? || ' days')
            """,
            (days,)
        )
        deleted_count = cursor.rowcount
        self.connection.commit()
        logger.info(f"Deleted {deleted_count} old snapshots")
        return deleted_count
    
    # Modality Features Operations
    
    def add_modality_features(self, snapshot_id: int, modality: str,
                             features: np.ndarray, confidence: float = 1.0) -> int:
        """
        Add individual modality features for a snapshot.
        
        Args:
            snapshot_id: Snapshot ID
            modality: Modality name ('reid', 'pose', 'color', 'accessory', 'motion')
            features: Feature vector
            confidence: Feature confidence
            
        Returns:
            feature_id of the created modality features
        """
        cursor = self.connection.cursor()
        features_blob = pickle.dumps(features)
        
        cursor.execute(
            """
            INSERT INTO modality_features (snapshot_id, modality, features, confidence)
            VALUES (?, ?, ?, ?)
            """,
            (snapshot_id, modality, features_blob, confidence)
        )
        self.connection.commit()
        feature_id = cursor.lastrowid
        return feature_id
    
    # Match Operations
    
    def add_match(self, query_person_id: Optional[int], matched_person_id: int,
                 confidence: float, camera_id: str, explanation: str,
                 modality_similarities: Dict[str, float]) -> int:
        """
        Add a match record.
        
        Args:
            query_person_id: Query person ID (optional)
            matched_person_id: Matched person ID
            confidence: Match confidence
            camera_id: Camera ID where match occurred
            explanation: Match explanation text
            modality_similarities: Dictionary of modality similarity scores
            
        Returns:
            match_id of the created match record
        """
        cursor = self.connection.cursor()
        
        cursor.execute(
            """
            INSERT INTO matches 
            (query_person_id, matched_person_id, confidence, camera_id, explanation,
             reid_similarity, pose_similarity, color_similarity, 
             accessory_similarity, motion_similarity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_person_id, matched_person_id, confidence, camera_id, explanation,
                modality_similarities.get('reid'),
                modality_similarities.get('pose'),
                modality_similarities.get('color'),
                modality_similarities.get('accessory'),
                modality_similarities.get('motion')
            )
        )
        self.connection.commit()
        match_id = cursor.lastrowid
        return match_id

    def clear_database(self) -> None:
        """Clear all records from database tables for a clean session."""
        if not self.connection:
            return
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM modality_features")
        cursor.execute("DELETE FROM accessories")
        cursor.execute("DELETE FROM feature_snapshots")
        cursor.execute("DELETE FROM matches")
        cursor.execute("DELETE FROM tracking_history")
        cursor.execute("DELETE FROM persons")
        cursor.execute("DELETE FROM system_events")
        self.connection.commit()
        logger.info("Cleared all database tables")
    
    def get_recent_matches(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent match records.
        
        Args:
            limit: Maximum number of matches to return
            
        Returns:
            List of match dictionaries
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM matches 
            ORDER BY match_time DESC 
            LIMIT ?
            """,
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    # Camera Operations
    
    def add_camera(self, camera_id: str, location: Optional[str] = None,
                  rtsp_url: Optional[str] = None) -> None:
        """
        Add a camera to the database.
        
        Args:
            camera_id: Camera ID
            location: Camera location
            rtsp_url: RTSP stream URL
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO cameras (camera_id, location, rtsp_url)
            VALUES (?, ?, ?)
            """,
            (camera_id, location, rtsp_url)
        )
        self.connection.commit()
        logger.info(f"Added camera: {camera_id}")
    
    def get_cameras(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all cameras.
        
        Args:
            active_only: Only return active cameras
            
        Returns:
            List of camera dictionaries
        """
        cursor = self.connection.cursor()
        
        if active_only:
            cursor.execute("SELECT * FROM cameras WHERE is_active = TRUE")
        else:
            cursor.execute("SELECT * FROM cameras")
        
        return [dict(row) for row in cursor.fetchall()]
    
    # System Events
    
    def log_event(self, event_type: str, details: str,
                 camera_id: Optional[str] = None, severity: str = "INFO") -> None:
        """
        Log a system event.
        
        Args:
            event_type: Type of event
            details: Event details
            camera_id: Related camera ID (optional)
            severity: Event severity ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO system_events (event_type, details, camera_id, severity)
            VALUES (?, ?, ?, ?)
            """,
            (event_type, details, camera_id, severity)
        )
        self.connection.commit()
    
    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, 'connection') and self.connection:
            try:
                self.connection.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

