"""
Forensic Spatio-Temporal Database & FAISS Vector Index
Handles robust SQLite storage, 512-D vector similarity matching, and persistent identity profiles.
"""

import re
import sqlite3
import json
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Iterable
from datetime import datetime, timezone
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from forensic_app.core.signature import ForensicSignature
from forensic_app.core.catalog import normalize_accessory

logger = logging.getLogger(__name__)

# Trailing digits of a person uid ("UID-0007" -> 7, legacy "UID-CAM_01-0036" -> 36)
_UID_SEQ_RE = re.compile(r"(\d+)\s*$")

# Lower-garment strings that mean "no measurable pants colour"
_NON_PANTS_TOKENS = ("not detected", "bare", "none", "occluded", "shorts")

# Keyword fallback for colour families (used until/unless utils.color_naming.color_family exists)
_FAMILY_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("neutral-dark", ("black", "charcoal")),
    ("neutral-light", ("white", "cream", "off-white", "ivory")),
    ("neutral-mid", ("grey", "gray", "silver")),
    ("blue", ("blue", "navy", "denim", "teal", "cyan", "turquoise")),
    ("red", ("red", "maroon", "burgundy", "crimson")),
    ("pink", ("pink", "magenta")),
    ("purple", ("purple", "violet", "lavender")),
    ("green", ("green", "olive")),
    ("yellow", ("yellow", "gold", "mustard")),
    ("orange", ("orange",)),
    ("brown", ("brown", "khaki", "beige", "tan", "camel")),
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def color_family_of(name: Optional[str]) -> str:
    """Colour family of a colour name (delegates to utils.color_naming.color_family when available)."""
    if not name:
        return "unknown"
    try:
        from forensic_app.utils.color_naming import color_family  # provided by the attributes module
        fam = color_family(name)
        if fam:
            return str(fam).lower()
    except Exception:
        pass
    low = name.strip().lower()
    for fam, keys in _FAMILY_KEYWORDS:
        if any(k in low for k in keys):
            return fam
    return "unknown"


def _is_unknown_color(name: Optional[str]) -> bool:
    low = (name or "").strip().lower()
    return low in ("", "unknown", "any", "all", "not detected")


def colors_match_tolerant(query: Optional[str], stored: Optional[str]) -> bool:
    """Tolerant colour name match: substring either way ("Blue" ~ "Navy Blue") or same colour family."""
    q = (query or "").strip().lower()
    s = (stored or "").strip().lower()
    if not q or q in ("any", "all"):
        return True
    if not s:
        return False
    if q in s or s in q:
        return True
    fq, fs = color_family_of(q), color_family_of(s)
    return fq != "unknown" and fq == fs


def _blob_to_matrix(blob: Optional[bytes], dim: int) -> Optional[np.ndarray]:
    """Decode a float32 BLOB into a (K, dim) matrix (None when empty/corrupt)."""
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size == 0 or arr.size % dim != 0:
        return None
    return arr.reshape(-1, dim).copy()


def _lab_to_json(lab: Any) -> Optional[str]:
    if lab is None:
        return None
    try:
        vals = [float(v) for v in list(lab)[:3]]
        return json.dumps(vals) if len(vals) == 3 else None
    except Exception:
        return None


def _json_to_lab(txt: Optional[str]) -> Optional[List[float]]:
    if not txt:
        return None
    try:
        vals = json.loads(txt)
        return [float(v) for v in vals[:3]] if vals and len(vals) >= 3 else None
    except Exception:
        return None


class ForensicDatabase:
    """Unified Forensic Storage: SQLite relational store + FAISS vector indexing."""

    def __init__(self, db_path: Optional[str] = None, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent / "database"
            base_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(base_dir / "forensic_records.db")

        self.is_memory = str(db_path) == ":memory:"
        self.db_path = Path(db_path)
        if not self.is_memory:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Re-entrant lock: public methods may call each other (e.g. clear_database -> log_event)
        self._lock = threading.RLock()
        # Incremented by every successful clear_database(): lets long-running workers notice a wipe
        self.clear_generation = 0

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

    @property
    def crops_dir(self) -> Optional[Path]:
        """Directory holding per-person thumbnail crops (None for in-memory databases)."""
        if self.is_memory:
            return None
        return self.db_path.parent / "crops"

    def _init_sqlite_schema(self) -> None:
        """Create relational tables with indexes."""
        with self._lock:
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
            # Persistent appearance + attribute gallery used by the identity manager
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS identity_profiles (
                    person_uid TEXT PRIMARY KEY,
                    embedding_model TEXT,
                    mean_embedding BLOB,
                    prototypes BLOB,
                    proto_count INTEGER DEFAULT 0,
                    upper_color TEXT,
                    upper_hex TEXT,
                    upper_lab TEXT,
                    lower_color TEXT,
                    lower_hex TEXT,
                    lower_lab TEXT,
                    lower_type TEXT,
                    accessory_stats TEXT,
                    n_observations INTEGER DEFAULT 0,
                    best_crop_quality REAL DEFAULT 0.0,
                    last_seen TEXT,
                    updated_at TEXT,
                    extra_json TEXT
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
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT person_uid, camera_id, timestamp, upper_color, upper_hex, lower_color, lower_hex, accessories_json, biometrics_json, posture, confidence, bbox_json, crop_path, embedding_blob FROM signatures ORDER BY sig_id ASC")
            rows = cursor.fetchall()
            vectors = []
            for row in rows:
                uid = row["person_uid"]
                emb_blob = row["embedding_blob"]
                if not emb_blob:
                    continue
                emb = np.frombuffer(emb_blob, dtype=np.float32)
                if len(emb) != self.embedding_dim:
                    continue
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

                # Reconstruct signature cache
                try:
                    accessories = json.loads(row["accessories_json"] or "[]")
                    biometrics = json.loads(row["biometrics_json"] or "{}")
                except Exception:
                    accessories, biometrics = [], {}
                self.uid_to_sig[uid] = ForensicSignature(
                    person_uid=uid,
                    camera_id=row["camera_id"],
                    timestamp=row["timestamp"],
                    upper_garment={"color": row["upper_color"], "hex": row["upper_hex"]},
                    lower_garment={"color": row["lower_color"], "hex": row["lower_hex"]},
                    accessories=accessories,
                    biometrics=biometrics,
                    posture=row["posture"] or "Standing",
                    embedding=emb.tolist(),
                    crop_path=row["crop_path"],
                    confidence=float(row["confidence"] or 1.0)
                )
                self._update_mean_embedding(uid, emb)

                if self.index is not None and norm > 0:
                    vectors.append(emb)
                    self.index_to_uid[len(vectors) - 1] = uid

            if vectors and self.index is not None:
                mat = np.array(vectors, dtype=np.float32)
                self.index.add(mat)
                logger.info(f"Loaded {len(vectors)} signatures into FAISS index.")

    def _update_mean_embedding(self, uid: str, emb: np.ndarray) -> None:
        """Running EMA of a person's (raw) embedding, unit-normalised."""
        if float(np.linalg.norm(emb)) == 0.0:
            return
        if uid in self.person_mean_embeddings:
            up = 0.85 * self.person_mean_embeddings[uid] + 0.15 * emb
            n = np.linalg.norm(up)
            self.person_mean_embeddings[uid] = (up / n if n > 0 else up).astype(np.float32)
        else:
            self.person_mean_embeddings[uid] = np.asarray(emb, dtype=np.float32)

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

                # 3. Add to FAISS in-memory index (zero embeddings carry no appearance information)
                if self.index is not None and float(np.linalg.norm(emb_np)) > 0:
                    idx_pos = self.index.ntotal
                    self.index.add(emb_np.reshape(1, -1))
                    self.index_to_uid[idx_pos] = sig.person_uid
                self.uid_to_sig[sig.person_uid] = sig

                # 4. Update running EMA mean embedding for person
                self._update_mean_embedding(sig.person_uid, emb_np)
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
                cursor.execute("DELETE FROM identity_profiles;")
                cursor.execute("DELETE FROM system_logs;")
                self.conn.commit()
                self.clear_generation += 1

                # Reset FAISS vector index
                if self.index is not None:
                    self.index.reset()
                self.index_to_uid.clear()
                self.uid_to_sig.clear()
                self.person_mean_embeddings.clear()

                # Safely delete disk crops
                crops_dir = self.crops_dir
                if crops_dir is not None and crops_dir.exists():
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

    # ------------------------------------------------------------------ legacy one-shot matcher
    @staticmethod
    def _color_name_adjustment(query_upper: Optional[str], stored_upper: Optional[str],
                               query_lower: Optional[str], stored_lower: Optional[str]) -> float:
        """Score adjustment from colour NAMES (exact > same family > conflicting)."""
        adj = 0.0
        qu, su = (query_upper or "").strip().lower(), (stored_upper or "").strip().lower()
        if not _is_unknown_color(qu) and not _is_unknown_color(su):
            if qu == su:
                adj += 0.08                                   # strong colour agreement
            elif color_family_of(qu) == color_family_of(su) != "unknown":
                adj += 0.03                                   # same family (lighting tolerant)
            else:
                adj -= 0.10                                   # different shirt colour

        ql, sl = (query_lower or "").strip().lower(), (stored_lower or "").strip().lower()
        if not _is_unknown_color(ql) and not _is_unknown_color(sl):
            q_np = any(t in ql for t in _NON_PANTS_TOKENS)
            s_np = any(t in sl for t in _NON_PANTS_TOKENS)
            if ql == sl:
                adj += 0.06 if ("bare" in ql) else 0.04
            elif "bare" in ql and "bare" in sl:
                adj += 0.06
            elif not q_np and not s_np and color_family_of(ql) == color_family_of(sl) != "unknown":
                adj += 0.02
            elif ("bare" in ql) != ("bare" in sl) and not ("occluded" in ql or "occluded" in sl):
                adj -= 0.05                                   # pants vs bare legs
        return adj

    def match_person(self,
                     query_embedding: np.ndarray,
                     upper_color: Optional[str] = None,
                     lower_color: Optional[str] = None,
                     min_similarity: float = 0.58) -> Tuple[Optional[str], float]:
        """
        Query FAISS vector store and attribute index to match a detected person
        with an existing global Person UID (single-shot; the live pipeline uses IdentityManager).

        Returns:
            (matched_person_uid, similarity_score) or (None, 0.0)
        """
        with self._lock:
            if query_embedding is None or len(query_embedding) != self.embedding_dim:
                return None, 0.0

            q = np.array(query_embedding, dtype=np.float32)
            q_norm = np.linalg.norm(q)
            if q_norm == 0:
                return None, 0.0
            q = q / q_norm

            candidate_scores: Dict[str, float] = {}

            # 1. Search FAISS index snapshots (wide k so one heavily-sampled person cannot hide others)
            if self.index is not None and self.index.ntotal > 0:
                search_k = min(max(64, 4 * len(self.person_mean_embeddings)), self.index.ntotal)
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
            if self.person_mean_embeddings:
                uids = list(self.person_mean_embeddings.keys())
                means = np.stack([self.person_mean_embeddings[u] for u in uids])
                sims = means @ q
                for uid, sim in zip(uids, sims):
                    sim = float(sim)
                    if uid not in candidate_scores or sim > candidate_scores[uid]:
                        candidate_scores[uid] = sim

            if not candidate_scores:
                return None, 0.0

            # 3. Multimodal scoring (Embedding Similarity + Colour Agreement)
            best_uid = None
            best_score = 0.0
            for uid, sim in candidate_scores.items():
                sig = self.uid_to_sig.get(uid)
                effective_score = sim
                if sig:
                    effective_score += self._color_name_adjustment(
                        upper_color, sig.upper_garment.get("color"),
                        lower_color, sig.lower_garment.get("color"))
                if effective_score > best_score:
                    best_score = effective_score
                    best_uid = uid

            if best_uid is not None and best_score >= min_similarity:
                return best_uid, float(min(1.0, max(0.0, best_score)))

            return None, 0.0

    # ------------------------------------------------------------------ attribute search
    def query_by_attributes(self,
                            upper_color: Optional[str] = None,
                            lower_color: Optional[str] = None,
                            accessories: Optional[List[str]] = None,
                            query_vector: Optional[np.ndarray] = None,
                            top_k: int = 15) -> List[Dict[str, Any]]:
        """
        Composite Multi-Modal Query combining attribute filtering and vector ranking (thread-safe).
        Colours match tolerantly ("Blue" matches "Navy Blue" and other blues); ALL requested
        accessories must be present on a person (names normalised to the canonical catalog).
        """
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT person_uid, last_camera, last_seen, upper_color, lower_color, accessories_json, posture FROM persons")
            candidates = [dict(row) for row in cursor.fetchall()]

            if upper_color and upper_color.strip().lower() not in ("any", "all", ""):
                candidates = [c for c in candidates if colors_match_tolerant(upper_color, c.get("upper_color"))]

            if lower_color and lower_color.strip().lower() not in ("any", "all", ""):
                low_q = lower_color.lower()
                if "none" in low_q or "bare" in low_q:
                    def _no_pants(val: Optional[str]) -> bool:
                        v = (val or "").lower()
                        return any(t in v for t in ("not detected", "bare", "none", "occluded"))
                    candidates = [c for c in candidates if _no_pants(c.get("lower_color"))]
                else:
                    candidates = [c for c in candidates if colors_match_tolerant(lower_color, c.get("lower_color"))]

            # Accessories: every requested (canonical) accessory must be present
            if accessories:
                targets = {normalize_accessory(a) for a in accessories if a and str(a).strip()}
                filtered = []
                for c in candidates:
                    try:
                        cand_accs = {normalize_accessory(a) for a in json.loads(c.get("accessories_json") or "[]")}
                    except Exception:
                        cand_accs = set()
                    if targets.issubset(cand_accs):
                        filtered.append(c)
                candidates = filtered

            q_unit = None
            if query_vector is not None:
                qv = np.asarray(query_vector, dtype=np.float32).ravel()
                if qv.size == self.embedding_dim and float(np.linalg.norm(qv)) > 0:
                    q_unit = qv / float(np.linalg.norm(qv))

            results = []
            for c in candidates:
                uid = c["person_uid"]
                vector_score = 0.5
                if q_unit is not None:
                    ref = self.person_mean_embeddings.get(uid)
                    if ref is None and uid in self.uid_to_sig:
                        ref = self.uid_to_sig[uid].get_embedding_numpy()
                    if ref is not None and float(np.linalg.norm(ref)) > 0:
                        vector_score = float(np.dot(ref, q_unit))
                        vector_score = max(0.0, min(1.0, (vector_score + 1.0) / 2.0))

                # Fetch latest snapshot details (prefer a snapshot that has a crop on disk)
                cursor.execute("SELECT crop_path, upper_hex, lower_hex, biometrics_json FROM signatures WHERE person_uid = ? ORDER BY (crop_path IS NULL), sig_id DESC LIMIT 1", (uid,))
                last_sig = cursor.fetchone()

                crop_path = last_sig["crop_path"] if last_sig else None
                upper_hex = last_sig["upper_hex"] if last_sig else "#808080"
                lower_hex = last_sig["lower_hex"] if last_sig else "#404040"

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

            # Highest similarity first, most recently seen first among ties
            results.sort(key=lambda x: str(x.get("timestamp") or ""), reverse=True)
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:top_k]

    # ------------------------------------------------------------------ identity profiles
    def save_identity_profile(self, profile: Dict[str, Any]) -> bool:
        """
        Upsert one identity profile. Keys: person_uid (required), embedding_model, mean_embedding (512,),
        prototypes (K, 512), upper_color/upper_hex/upper_lab, lower_color/lower_hex/lower_lab, lower_type,
        accessory_stats {name: {"hits", "checks"}}, n_observations, best_crop_quality, last_seen, extra (dict).
        """
        uid = (profile or {}).get("person_uid")
        if not uid:
            return False
        with self._lock:
            try:
                mean = profile.get("mean_embedding")
                mean_blob = None
                if mean is not None:
                    mean_arr = np.asarray(mean, dtype=np.float32).ravel()
                    if mean_arr.size == self.embedding_dim:
                        mean_blob = mean_arr.tobytes()
                protos = profile.get("prototypes")
                proto_blob, proto_count = None, 0
                if protos is not None:
                    proto_arr = np.asarray(protos, dtype=np.float32)
                    if proto_arr.ndim == 1:
                        proto_arr = proto_arr.reshape(1, -1)
                    if proto_arr.size and proto_arr.shape[1] == self.embedding_dim:
                        proto_blob = np.ascontiguousarray(proto_arr).tobytes()
                        proto_count = int(proto_arr.shape[0])

                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT INTO identity_profiles (person_uid, embedding_model, mean_embedding, prototypes, proto_count,
                        upper_color, upper_hex, upper_lab, lower_color, lower_hex, lower_lab, lower_type,
                        accessory_stats, n_observations, best_crop_quality, last_seen, updated_at, extra_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(person_uid) DO UPDATE SET
                        embedding_model = excluded.embedding_model,
                        mean_embedding = excluded.mean_embedding,
                        prototypes = excluded.prototypes,
                        proto_count = excluded.proto_count,
                        upper_color = excluded.upper_color,
                        upper_hex = excluded.upper_hex,
                        upper_lab = excluded.upper_lab,
                        lower_color = excluded.lower_color,
                        lower_hex = excluded.lower_hex,
                        lower_lab = excluded.lower_lab,
                        lower_type = excluded.lower_type,
                        accessory_stats = excluded.accessory_stats,
                        n_observations = excluded.n_observations,
                        best_crop_quality = excluded.best_crop_quality,
                        last_seen = excluded.last_seen,
                        updated_at = excluded.updated_at,
                        extra_json = excluded.extra_json;
                """, (
                    uid,
                    profile.get("embedding_model") or "unknown",
                    mean_blob,
                    proto_blob,
                    proto_count,
                    profile.get("upper_color"),
                    profile.get("upper_hex"),
                    _lab_to_json(profile.get("upper_lab")),
                    profile.get("lower_color"),
                    profile.get("lower_hex"),
                    _lab_to_json(profile.get("lower_lab")),
                    profile.get("lower_type"),
                    json.dumps(profile.get("accessory_stats") or {}),
                    int(profile.get("n_observations") or 0),
                    float(profile.get("best_crop_quality") or 0.0),
                    profile.get("last_seen") or _now_iso(),
                    _now_iso(),
                    json.dumps(profile.get("extra") or {}, default=float),
                ))
                self.conn.commit()
                return True
            except Exception as e:
                logger.error(f"Error saving identity profile {uid}: {e}")
                return False

    def load_identity_profiles(self) -> List[Dict[str, Any]]:
        """Load every persisted identity profile (same dict layout as save_identity_profile)."""
        with self._lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute("SELECT * FROM identity_profiles ORDER BY person_uid ASC")
                rows = cursor.fetchall()
            except Exception as e:
                logger.error(f"Error loading identity profiles: {e}")
                return []

        profiles = []
        for row in rows:
            try:
                mean = _blob_to_matrix(row["mean_embedding"], self.embedding_dim)
                protos = _blob_to_matrix(row["prototypes"], self.embedding_dim)
                try:
                    acc_stats = json.loads(row["accessory_stats"] or "{}")
                except Exception:
                    acc_stats = {}
                try:
                    extra = json.loads(row["extra_json"] or "{}")
                except Exception:
                    extra = {}
                profiles.append({
                    "person_uid": row["person_uid"],
                    "embedding_model": row["embedding_model"] or "unknown",
                    "mean_embedding": mean[0] if mean is not None else None,
                    "prototypes": protos,
                    "proto_count": int(row["proto_count"] or 0),
                    "upper_color": row["upper_color"],
                    "upper_hex": row["upper_hex"],
                    "upper_lab": _json_to_lab(row["upper_lab"]),
                    "lower_color": row["lower_color"],
                    "lower_hex": row["lower_hex"],
                    "lower_lab": _json_to_lab(row["lower_lab"]),
                    "lower_type": row["lower_type"],
                    "accessory_stats": acc_stats if isinstance(acc_stats, dict) else {},
                    "n_observations": int(row["n_observations"] or 0),
                    "best_crop_quality": float(row["best_crop_quality"] or 0.0),
                    "last_seen": row["last_seen"],
                    "updated_at": row["updated_at"],
                    "extra": extra if isinstance(extra, dict) else {},
                })
            except Exception as e:
                logger.debug(f"Skipping corrupt identity profile row: {e}")
        return profiles

    def delete_identity_profile(self, uid: str) -> bool:
        """Remove one identity profile row."""
        with self._lock:
            try:
                self.conn.execute("DELETE FROM identity_profiles WHERE person_uid = ?", (uid,))
                self.conn.commit()
                return True
            except Exception as e:
                logger.error(f"Error deleting identity profile {uid}: {e}")
                return False

    def next_person_sequence(self) -> int:
        """Next free numeric uid suffix: 1 + max trailing digits over persons AND identity profiles."""
        with self._lock:
            max_seq = 0
            try:
                cursor = self.conn.cursor()
                for table in ("persons", "identity_profiles"):
                    cursor.execute(f"SELECT person_uid FROM {table}")
                    for (uid,) in cursor.fetchall():
                        m = _UID_SEQ_RE.search(str(uid or ""))
                        if m:
                            max_seq = max(max_seq, int(m.group(1)))
            except Exception as e:
                logger.error(f"Error computing next person sequence: {e}")
            return max_seq + 1

    def known_person_uids(self) -> List[str]:
        """All person uids present in the persons table or identity profiles."""
        with self._lock:
            uids = set()
            try:
                cursor = self.conn.cursor()
                for table in ("persons", "identity_profiles"):
                    cursor.execute(f"SELECT person_uid FROM {table}")
                    uids.update(str(r[0]) for r in cursor.fetchall() if r[0])
            except Exception:
                pass
            return sorted(uids)

    def merge_identities(self, src_uid: str, dst_uid: str) -> bool:
        """
        Merge identity `src_uid` into `dst_uid`: signatures + persons rows, identity profile row,
        FAISS id map, EMA mean embeddings, signature cache and the thumbnail crop on disk.
        The caller (identity manager) saves the merged dst profile itself.
        """
        if not src_uid or not dst_uid or src_uid == dst_uid:
            return False
        with self._lock:
            try:
                cursor = self.conn.cursor()
                src_row = cursor.execute("SELECT * FROM persons WHERE person_uid = ?", (src_uid,)).fetchone()
                dst_row = cursor.execute("SELECT * FROM persons WHERE person_uid = ?", (dst_uid,)).fetchone()

                # 1. Crop thumbnail: move src -> dst when dst has none (else drop the orphan src crop)
                src_crop_new: Optional[str] = None
                src_crop_old: Optional[str] = None
                crops_dir = self.crops_dir
                if crops_dir is not None:
                    src_file = crops_dir / f"{src_uid}.jpg"
                    dst_file = crops_dir / f"{dst_uid}.jpg"
                    if src_file.exists():
                        src_crop_old = str(src_file)
                        try:
                            if not dst_file.exists():
                                src_file.replace(dst_file)
                            else:
                                src_file.unlink()
                            src_crop_new = str(dst_file)
                        except Exception as fe:
                            logger.debug(f"Crop move failed during merge: {fe}")

                # 2. Persons rows (dst row must exist before signatures can point at it: FK)
                if src_row is not None:
                    if dst_row is None:
                        cursor.execute("""
                            INSERT INTO persons (person_uid, first_seen, last_seen, last_camera, upper_color, lower_color,
                                                 accessories_json, posture, notes, status)
                            SELECT ?, first_seen, last_seen, last_camera, upper_color, lower_color,
                                   accessories_json, posture, notes, status FROM persons WHERE person_uid = ?
                        """, (dst_uid, src_uid))
                    else:
                        first_seen = min(str(src_row["first_seen"] or ""), str(dst_row["first_seen"] or "")) or dst_row["first_seen"]
                        src_newer = str(src_row["last_seen"] or "") > str(dst_row["last_seen"] or "")
                        last_seen = src_row["last_seen"] if src_newer else dst_row["last_seen"]
                        cursor.execute("UPDATE persons SET first_seen = ?, last_seen = ? WHERE person_uid = ?",
                                       (first_seen, last_seen, dst_uid))
                        if src_newer:
                            # The most recent sighting carries the freshest attribute summary
                            cursor.execute("""
                                UPDATE persons SET last_camera = ?, upper_color = ?, lower_color = ?,
                                                   accessories_json = ?, posture = ?
                                WHERE person_uid = ?
                            """, (src_row["last_camera"], src_row["upper_color"], src_row["lower_color"],
                                  src_row["accessories_json"], src_row["posture"], dst_uid))

                # 3. Signatures -> dst (and repoint their crop paths)
                cursor.execute("UPDATE signatures SET person_uid = ? WHERE person_uid = ?", (dst_uid, src_uid))
                if src_crop_old and src_crop_new:
                    cursor.execute("UPDATE signatures SET crop_path = ? WHERE crop_path = ?", (src_crop_new, src_crop_old))
                if src_row is not None:
                    cursor.execute("DELETE FROM persons WHERE person_uid = ?", (src_uid,))

                # 4. Identity profile rows: keep dst; rename src when dst has none
                has_dst_profile = cursor.execute("SELECT 1 FROM identity_profiles WHERE person_uid = ?",
                                                 (dst_uid,)).fetchone() is not None
                if has_dst_profile:
                    cursor.execute("DELETE FROM identity_profiles WHERE person_uid = ?", (src_uid,))
                else:
                    cursor.execute("UPDATE identity_profiles SET person_uid = ? WHERE person_uid = ?", (dst_uid, src_uid))
                self.conn.commit()

                # 5. In-memory indices
                for idx, uid in list(self.index_to_uid.items()):
                    if uid == src_uid:
                        self.index_to_uid[idx] = dst_uid
                src_mean = self.person_mean_embeddings.pop(src_uid, None)
                if src_mean is not None:
                    dst_mean = self.person_mean_embeddings.get(dst_uid)
                    if dst_mean is None:
                        self.person_mean_embeddings[dst_uid] = src_mean
                    else:
                        merged = 0.5 * (dst_mean + src_mean)
                        n = np.linalg.norm(merged)
                        self.person_mean_embeddings[dst_uid] = (merged / n if n > 0 else merged).astype(np.float32)
                src_sig = self.uid_to_sig.pop(src_uid, None)
                if src_sig is not None:
                    src_sig.person_uid = dst_uid
                    if src_crop_old and src_sig.crop_path == src_crop_old:
                        src_sig.crop_path = src_crop_new
                    if dst_uid not in self.uid_to_sig:
                        self.uid_to_sig[dst_uid] = src_sig
                dst_sig = self.uid_to_sig.get(dst_uid)
                if dst_sig is not None and src_crop_old and dst_sig.crop_path == src_crop_old:
                    dst_sig.crop_path = src_crop_new

                self.log_event("IDENTITY_MERGED", f"{src_uid} merged into {dst_uid}")
                logger.info(f"Identity {src_uid} merged into {dst_uid}")
                return True
            except Exception as e:
                try:
                    self.conn.rollback()
                except Exception:
                    pass
                logger.error(f"Failed to merge identity {src_uid} into {dst_uid}: {e}")
                return False

    # ------------------------------------------------------------------ misc
    def log_event(self, event_type: str, details: str, severity: str = "INFO") -> None:
        """Insert system log entry (thread-safe)."""
        with self._lock:
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
                    "faiss_indexed": self.index.ntotal if self.index is not None else 0
                }
            except Exception:
                return {"total_persons": 0, "total_signatures": 0, "faiss_indexed": 0}

    def close(self) -> None:
        """Close SQLite connection."""
        with self._lock:
            if hasattr(self, 'conn') and self.conn:
                try:
                    self.conn.close()
                except Exception:
                    pass
