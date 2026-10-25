"""
Forensic Spatio-Temporal Database & FAISS Vector Index
Handles robust SQLite storage and 512-D vector similarity matching.
"""

import sqlite3
import json
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from forensic_app.core.signature import ForensicSignature

logger = logging.getLogger(__name__)


class ForensicDatabase:
    """Unified Forensic Storage: SQLite relational store + FAISS vector indexing."""

    def __init__(self, db_path: Optional[str] = None, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent / "database"
            base_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(base_dir / "forensic_records.db")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread synchronization lock
        self._lock = threading.Lock()

        # Thread-safe SQLite connection
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_sqlite_schema()

        # FAISS Index Initialization
        self.index = None
        self.index_to_uid: Dict[int, str] = {}
        self.uid_to_sig: Dict[str, ForensicSignature] = {}
        self.person_mean_embeddings: Dict[str, np.ndarray] = {}
        self._init_faiss()
        self._sync_faiss_from_sqlite()

        logger.info(f"ForensicDatabase initialized at {self.db_path}")

    def _init_sqlite_schema(self) -> None:
        """Create relational tables with indexes."""
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                person_uid TEXT PRIMARY KEY,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_camera TEXT,
                upper_color TEXT,
                lower_color TEXT,
                accessories_json TEXT,
                posture TEXT,
                notes TEXT,
                status TEXT DEFAULT 'ACTIVE'
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signatures (
                sig_id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_uid TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                track_id INTEGER,
                upper_color TEXT,
                upper_hex TEXT,
                lower_color TEXT,
                lower_hex TEXT,
                accessories_json TEXT,
                biometrics_json TEXT,
                posture TEXT,
                confidence REAL,
                bbox_json TEXT,
                crop_path TEXT,
                embedding_blob BLOB,
                FOREIGN KEY (person_uid) REFERENCES persons(person_uid) ON DELETE CASCADE
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT,
                details TEXT,
                severity TEXT DEFAULT 'INFO'
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sig_uid ON signatures(person_uid);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sig_time ON signatures(timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sig_colors ON signatures(upper_color, lower_color);")
        self.conn.commit()

    def _init_faiss(self) -> None:
        """Initialize FAISS cosine similarity (inner product) index."""
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            self.index = None
            logger.warning("FAISS not installed; vector search will fallback to NumPy cosine distance.")

    def _sync_faiss_from_sqlite(self) -> None:
        """Load stored signatures into FAISS index on startup."""
        if not self.index:
            return
        cursor = self.conn.cursor()
        cursor.execute("SELECT person_uid, camera_id, timestamp, upper_color, upper_hex, lower_color, lower_hex, accessories_json, biometrics_json, posture, confidence, bbox_json, crop_path, embedding_blob FROM signatures ORDER BY sig_id ASC")
        rows = cursor.fetchall()
        vectors = []
        for idx, row in enumerate(rows):
            uid = row["person_uid"]
            emb_blob = row["embedding_blob"]
            if emb_blob:
                emb = np.frombuffer(emb_blob, dtype=np.float32)
                if len(emb) == self.embedding_dim:
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    vectors.append(emb)
                    self.index_to_uid[self.index.ntotal + len(vectors) - 1] = uid

                    # Reconstruct signature cache
                    sig = ForensicSignature(
                        person_uid=uid,
                        camera_id=row["camera_id"],
                        timestamp=row["timestamp"],
                        upper_garment={"color": row["upper_color"], "hex": row["upper_hex"]},
                        lower_garment={"color": row["lower_color"], "hex": row["lower_hex"]},
                        accessories=json.loads(row["accessories_json"] or "[]"),
                        biometrics=json.loads(row["biometrics_json"] or "{}"),
                        posture=row["posture"] or "Standing",
                        embedding=emb.tolist(),
                        crop_path=row["crop_path"],
                        confidence=float(row["confidence"] or 1.0)
                    )
                    self.uid_to_sig[uid] = sig

                    # Build mean embedding
                    if uid in self.person_mean_embeddings:
                        m_curr = self.person_mean_embeddings[uid]
                        m_up = 0.85 * m_curr + 0.15 * emb
                        m_norm = np.linalg.norm(m_up)
                        if m_norm > 0:
                            m_up = m_up / m_norm
                        self.person_mean_embeddings[uid] = m_up
                    else:
                        self.person_mean_embeddings[uid] = emb

        if vectors:
            mat = np.array(vectors, dtype=np.float32)
            self.index.add(mat)
            logger.info(f"Loaded {len(vectors)} signatures into FAISS index.")

    def save_signature(self, sig: ForensicSignature) -> bool:
        """Persist signature into SQLite and index into FAISS (thread-safe)."""
        with self._lock:
            try:
                cursor = self.conn.cursor()
                emb_np = sig.get_embedding_numpy()
                emb_blob = emb_np.tobytes()

                # 1. Update or Insert Person Master Record
                cursor.execute("""
                    INSERT INTO persons (person_uid, first_seen, last_seen, last_camera, upper_color, lower_color, accessories_json, posture, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE')
                    ON CONFLICT(person_uid) DO UPDATE SET
                        last_seen = excluded.last_seen,
                        last_camera = excluded.last_camera,
                        upper_color = excluded.upper_color,
                        lower_color = excluded.lower_color,
                        accessories_json = excluded.accessories_json,
                        posture = excluded.posture;
                """, (
                    sig.person_uid,
                    sig.timestamp,
                    sig.timestamp,
                    sig.camera_id,
                    sig.upper_garment.get("color", "Unknown"),
                    sig.lower_garment.get("color", "Not Detected"),
                    json.dumps(sig.accessories),
                    sig.posture
                ))

                # 2. Insert Signature Snapshot
                cursor.execute("""
                    INSERT INTO signatures (person_uid, camera_id, timestamp, track_id, upper_color, upper_hex, lower_color, lower_hex, accessories_json, biometrics_json, posture, confidence, bbox_json, crop_path, embedding_blob)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sig.person_uid,
                    sig.camera_id,
                    sig.timestamp,
                    sig.track_id,
                    sig.upper_garment.get("color", "Unknown"),
                    sig.upper_garment.get("hex", "#808080"),
                    sig.lower_garment.get("color", "Not Detected"),
                    sig.lower_garment.get("hex", "#404040"),
                    json.dumps(sig.accessories),
                    json.dumps(sig.biometrics),
                    sig.posture,
                    sig.confidence,
                    json.dumps(sig.bbox),
                    sig.crop_path,
                    emb_blob
                ))

                self.conn.commit()

                # 3. Add to FAISS in-memory index
                if self.index:
                    idx_pos = self.index.ntotal
                    self.index.add(emb_np.reshape(1, -1))
                    self.index_to_uid[idx_pos] = sig.person_uid
                    self.uid_to_sig[sig.person_uid] = sig

                # 4. Update running EMA mean embedding for person
                if sig.person_uid in self.person_mean_embeddings:
                    curr_emb = self.person_mean_embeddings[sig.person_uid]
                    up_emb = 0.85 * curr_emb + 0.15 * emb_np
                    norm = np.linalg.norm(up_emb)
                    if norm > 0:
                        up_emb = up_emb / norm
                    self.person_mean_embeddings[sig.person_uid] = up_emb
                else:
                    self.person_mean_embeddings[sig.person_uid] = emb_np

                return True
            except Exception as e:
                logger.error(f"Error saving signature for {sig.person_uid}: {e}")
                return False

    def clear_database(self) -> bool:
        """
        Safely wipe all forensic records, reset FAISS index, and clean crop images.
        Completely immune to crashes and unhandled exceptions (thread-safe).
        """
        with self._lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute("DELETE FROM signatures;")
                cursor.execute("DELETE FROM persons;")
                cursor.execute("DELETE FROM system_logs;")
                self.conn.commit()

                # Reset FAISS vector index
                if self.index:
                    self.index.reset()
                self.index_to_uid.clear()
                self.uid_to_sig.clear()
                self.person_mean_embeddings.clear()

                # Safely delete disk crops
                crops_dir = self.db_path.parent / "crops"
                if crops_dir.exists():
                    for f in crops_dir.glob("*.jpg"):
                        try:
                            f.unlink()
                        except Exception:
                            pass

                self.log_event("DATABASE_CLEARED", "All person profiles and vector indices cleared successfully.")
                logger.info("Forensic database and FAISS index cleared safely.")
                return True
            except Exception as e:
                logger.error(f"Failed to clear database: {e}")
                return False

    def match_person(self,
                     query_embedding: np.ndarray,
                     upper_color: Optional[str] = None,
                     lower_color: Optional[str] = None,
                     min_similarity: float = 0.58) -> Tuple[Optional[str], float]:
        """
        Query FAISS vector store and attribute index to match a detected person
        with an existing global Person UID.
        
        Returns:
            (matched_person_uid, similarity_score) or (None, 0.0)
        """
        with self._lock:
            if query_embedding is None or len(query_embedding) != self.embedding_dim:
                return None, 0.0

            q = np.array(query_embedding, dtype=np.float32)
            q_norm = np.linalg.norm(q)
            if q_norm > 0:
                q = q / q_norm

            candidate_scores: Dict[str, float] = {}

            # 1. Search FAISS index snapshots
            if self.index is not None and self.index.ntotal > 0:
                search_k = min(15, self.index.ntotal)
                D, I = self.index.search(q.reshape(1, -1), search_k)
                for dist, idx in zip(D[0], I[0]):
                    if idx < 0:
                        continue
                    uid = self.index_to_uid.get(int(idx))
                    if uid:
                        sim = float(dist)
                        if uid not in candidate_scores or sim > candidate_scores[uid]:
                            candidate_scores[uid] = sim

            # 2. Check Person Mean Embeddings (EMA)
            for uid, mean_emb in self.person_mean_embeddings.items():
                sim = float(np.dot(q, mean_emb))
                if uid not in candidate_scores or sim > candidate_scores[uid]:
                    candidate_scores[uid] = sim

            if not candidate_scores:
                return None, 0.0

            # 3. Multimodal scoring (Embedding Similarity + Color Agreement)
            best_uid = None
            best_score = 0.0

            for uid, sim in candidate_scores.items():
                sig = self.uid_to_sig.get(uid)
                effective_score = sim

                if sig:
                    st_upper = (sig.upper_garment.get("color") or "").lower()
                    st_lower = (sig.lower_garment.get("color") or "").lower()
                    qu_upper = (upper_color or "").lower()
                    qu_lower = (lower_color or "").lower()

                    # Upper clothing color comparison
                    if qu_upper and st_upper and qu_upper != "unknown" and st_upper != "unknown":
                        if qu_upper == st_upper:
                            effective_score += 0.08  # Strong color agreement bonus
                        elif qu_upper in ["black", "charcoal"] and st_upper in ["black", "charcoal"]:
                            effective_score += 0.04
                        elif qu_upper in ["beige", "khaki"] and st_upper in ["beige", "khaki"]:
                            effective_score += 0.04
                        else:
                            effective_score -= 0.10  # Different shirt color penalty

                    # Lower clothing comparison
                    if qu_lower and st_lower and qu_lower != "unknown" and st_lower != "unknown":
                        if qu_lower == st_lower:
                            effective_score += 0.04
                        elif "bare" in qu_lower and "bare" in st_lower:
                            effective_score += 0.06

                if effective_score > best_score:
                    best_score = effective_score
                    best_uid = uid

            if best_uid is not None and best_score >= min_similarity:
                return best_uid, float(min(1.0, max(0.0, best_score)))

            return None, 0.0

    def query_by_attributes(self,
                            upper_color: Optional[str] = None,
                            lower_color: Optional[str] = None,
                            accessories: Optional[List[str]] = None,
                            query_vector: Optional[np.ndarray] = None,
                            top_k: int = 15) -> List[Dict[str, Any]]:
        """
        Composite Multi-Modal Query combining attribute filtering and vector ranking (thread-safe).
        """
        with self._lock:
            cursor = self.conn.cursor()
            sql = "SELECT person_uid, last_camera, last_seen, upper_color, lower_color, accessories_json, posture FROM persons WHERE 1=1"
            params = []

            if upper_color and upper_color.lower() not in ["any", "all", ""]:
                sql += " AND LOWER(upper_color) LIKE LOWER(?)"
                params.append(f"%{upper_color}%")

            if lower_color and lower_color.lower() not in ["any", "all", ""]:
                if "none" in lower_color.lower() or "bare" in lower_color.lower():
                    sql += " AND (lower_color LIKE '%Not Detected%' OR lower_color LIKE '%Bare%' OR lower_color LIKE '%None%' OR lower_color LIKE '%Occluded%')"
                else:
                    sql += " AND LOWER(lower_color) LIKE LOWER(?)"
                    params.append(f"%{lower_color}%")

            cursor.execute(sql, params)
            candidates = [dict(row) for row in cursor.fetchall()]

            # Filter by accessories in Python for flexible subset containment
            if accessories and len(accessories) > 0:
                target_accs = [a.lower() for a in accessories]
                filtered = []
                for c in candidates:
                    try:
                        cand_accs = [a.lower() for a in json.loads(c.get("accessories_json") or "[]")]
                    except Exception:
                        cand_accs = []
                    if any(t in cand_accs for t in target_accs):
                        filtered.append(c)
                candidates = filtered

            # If vector query provided, compute Re-ID similarity scores
            results = []
            for c in candidates:
                uid = c["person_uid"]
                sig = self.uid_to_sig.get(uid)
                vector_score = 0.5
                if sig and query_vector is not None:
                    stored_emb = sig.get_embedding_numpy()
                    q_norm = query_vector / max(np.linalg.norm(query_vector), 1e-6)
                    vector_score = float(np.dot(stored_emb, q_norm))
                    vector_score = max(0.0, min(1.0, (vector_score + 1.0) / 2.0))

                # Fetch snapshot details
                cursor.execute("SELECT crop_path, upper_hex, lower_hex, biometrics_json FROM signatures WHERE person_uid = ? ORDER BY sig_id DESC LIMIT 1", (uid,))
                last_sig = cursor.fetchone()

                crop_path = last_sig["crop_path"] if last_sig else None
                upper_hex = last_sig["upper_hex"] if last_sig else "#808080"
                lower_hex = last_sig["lower_hex"] if last_sig else "#404040"

                acc_list = []
                try:
                    acc_list = json.loads(c.get("accessories_json") or "[]")
                except Exception:
                    acc_list = []

                results.append({
                    "person_uid": uid,
                    "camera_id": c.get("last_camera", "CAM_01"),
                    "timestamp": c.get("last_seen", ""),
                    "upper_color": c.get("upper_color", "Unknown"),
                    "upper_hex": upper_hex,
                    "lower_color": c.get("lower_color", "Not Detected"),
                    "lower_hex": lower_hex,
                    "accessories": acc_list,
                    "posture": c.get("posture", "Standing"),
                    "similarity_score": vector_score,
                    "crop_path": crop_path
                })

            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:top_k]

    def log_event(self, event_type: str, details: str, severity: str = "INFO") -> None:
        """Insert system log entry."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO system_logs (event_type, details, severity) VALUES (?, ?, ?)",
                           (event_type, details, severity))
            self.conn.commit()
        except Exception:
            pass

    def get_stats(self) -> Dict[str, int]:
        """Return total person profiles and snapshot counts (thread-safe)."""
        with self._lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM persons;")
                total_persons = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM signatures;")
                total_signatures = cursor.fetchone()[0]
                return {
                    "total_persons": total_persons,
                    "total_signatures": total_signatures,
                    "faiss_indexed": self.index.ntotal if self.index else 0
                }
            except Exception:
                return {"total_persons": 0, "total_signatures": 0, "faiss_indexed": 0}

    def close(self) -> None:
        """Close SQLite connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
