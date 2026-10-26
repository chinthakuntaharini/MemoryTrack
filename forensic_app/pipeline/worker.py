"""
Asynchronous Video Pipeline QThread Worker
Runs detection, pose, attributes, ReID and persistent identity assignment off the PyQt6 UI thread.

Per frame:
1. detect + track (full-frame person and accessory detection)
2. one batched pose call over the freshly measured tracks
3. globally exclusive accessory -> person assignment (keypoint aware)
4. per track: pose-guided garment parsing + scored attributes; the ROI accessory re-detection pass is
   throttled and staggered per track (every frame while a track is still being identified) and capped
   per frame
5. one batched ReID call (identifying tracks every frame, identified tracks every other frame)
6. IdentityManager.update -> persistent person UIDs ("Identifying..." until there is enough evidence,
   then a re-acquired or a new UID)
7. tracker-archived tracks -> IdentityManager.end_tracks; identity events -> UI signal + system log
8. signatures (identified tracks only, throttled), best thumbnail crop per UID, HUD drawing, UI signals
"""

import os
import time
import logging
import threading
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

from forensic_app.pipeline.detector import PedestrianDetector
from forensic_app.pipeline.parser import HumanParser
from forensic_app.pipeline.attributes import AttributeExtractor
from forensic_app.pipeline.pose import PoseBiometrics
from forensic_app.pipeline.reid import ReIDFeatureExtractor
from forensic_app.core.identity import (
    IdentityAssignment, IdentityManager, TrackObservation, TENTATIVE_LABEL, estimate_observation_quality,
)
from forensic_app.core.signature import ForensicSignature
from forensic_app.core.database import ForensicDatabase
from forensic_app.utils.drawing import (
    draw_forensic_bbox, draw_skeleton, identity_color_bgr, identity_color_hex,
)

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TENTATIVE_SKELETON_HEX = "#8A94A6"


@dataclass
class WorkerConfig:
    """Cadence / throttling knobs of the pipeline worker (intervals are counted in frames per track)."""
    roi_interval: int = 4                 # ROI accessory pass of an identified track every N frames ...
    roi_interval_tentative: int = 1       # ... and of a track that is still being identified
    max_roi_per_frame: int = 2            # hard cap of ROI passes per frame (~30 ms each with yolo11n on CPU)
    reid_interval: int = 2                # ReID embedding of an identified track every N frames (must not be
                                          # a multiple of the identity manager's gallery cadence of 3)
    signature_interval: int = 15          # DB signature snapshot of an identified track every N frames
    emit_interval: int = 3                # signature_detected per track at most every N frames
    min_box_side: int = 10                # boxes smaller than this (either side) are ignored
    det_conf_ref: float = 0.5             # detector confidence that gives full observation quality
    min_reid_quality: float = 0.10        # observations below this quality skip the ReID pass
    crop_jpeg_quality: int = 90           # thumbnail JPEG quality
    error_log_interval_s: float = 5.0     # repeated per-track errors are logged at most this often
    max_consecutive_frame_errors: int = 50  # whole-frame failures in a row before the stream is aborted
    fps_ema: float = 0.1                  # smoothing of the reported processing FPS


@dataclass
class _TrackState:
    """Worker-side bookkeeping of one tracker track (throttling and last outputs)."""
    track_id: int
    first_frame: int
    last_seen_frame: int = 0
    last_roi_frame: int = -10 ** 9
    last_reid_frame: int = -10 ** 9
    last_sig_frame: int = -10 ** 9
    last_emit_frame: int = -10 ** 9
    saved_uid: Optional[str] = None       # uid of the last saved signature
    emitted_uid: Optional[str] = None
    emitted_status: Optional[str] = None
    crop_uid: Optional[str] = None        # uid whose thumbnail is known to exist on disk
    origin_uid: Optional[str] = None      # uid the origin below refers to
    origin: Optional[str] = None          # "new" (minted by this track) or "reacquired" (matched the gallery)
    assignment: Optional[IdentityAssignment] = None
    embedding: Optional[np.ndarray] = None  # last valid ReID embedding
    biometrics: Dict[str, float] = field(default_factory=dict)


class ForensicPipelineWorker(QThread):
    """Real-time video inference loop running on a dedicated worker thread."""

    # PyQt6 Signals
    frame_ready = pyqtSignal(np.ndarray)
    signature_detected = pyqtSignal(dict)
    identity_event = pyqtSignal(dict)            # IdentityManager events + camera_id + timestamp
    stats_updated = pyqtSignal(int, int, float)  # active_tracks, session_identities, fps
    worker_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self,
                 db: ForensicDatabase,
                 source_type: str = "video",
                 source_path: Any = None,
                 camera_id: str = "CAM_01",
                 device: str = "cpu",
                 config: Optional[WorkerConfig] = None):
        super().__init__()
        self.db = db
        self.source_type = source_type
        self.source_path = source_path
        self.camera_id = camera_id or "CAM_01"
        self.device = device
        self.config = config or WorkerConfig()

        self._is_running = True
        self._is_paused = False
        self._reset_event = threading.Event()

        # Thumbnails live next to the database (db.merge_identities moves them there); in-memory DBs use
        # the project's database/crops folder
        db_crops = getattr(db, "crops_dir", None)
        self.crops_dir = Path(db_crops) if db_crops is not None else _PROJECT_ROOT / "database" / "crops"

        # Pipeline components (created on the worker thread in run())
        self.detector = None
        self.parser = None
        self.attributes = None
        self.pose = None
        self.reid = None
        self.identity: Optional[IdentityManager] = None

        # Backwards compatibility: current track id -> person uid (identified tracks only)
        self.track_to_uid: Dict[int, str] = {}
        self._tracks: Dict[int, _TrackState] = {}

        # Runtime statistics
        self.frame_index = 0
        self.processing_fps = 0.0
        self.stage_ms: Dict[str, float] = {}
        self._db_generation = int(getattr(db, "clear_generation", 0) or 0)
        self._error_log_times: Dict[str, float] = {}
        self._error_counts: Dict[str, int] = {}
        self._roi_available = False

    # ------------------------------------------------------------------ thread entry point
    def run(self) -> None:
        """Main execution loop of the worker thread."""
        cap = None
        try:
            if not self._init_components():
                return

            cap = self._open_capture()
            if cap is None or not cap.isOpened():
                self.error_occurred.emit(f"Could not open video stream: {self.source_path}")
                return

            stream_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if 1.0 < stream_fps < 241.0 and hasattr(self.pose, "set_fps"):
                self.pose.set_fps(stream_fps)

            is_live = self.source_type == "webcam"
            read_failures = 0
            frame_errors = 0
            t_last = time.perf_counter()

            while self._is_running:
                if self._is_paused:
                    self.msleep(30)
                    t_last = time.perf_counter()
                    continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    read_failures += 1
                    if is_live and read_failures < 50:
                        self.msleep(20)  # transient webcam hiccup
                        continue
                    break
                read_failures = 0

                frame = self._normalize_frame(frame)
                if frame is None:
                    continue

                self.frame_index += 1
                self._apply_pending_reset()
                try:
                    annotated, n_active = self._process_frame(frame, self.frame_index)
                    frame_errors = 0
                except Exception as e:
                    frame_errors += 1
                    self._log_error("frame", e)
                    if frame_errors >= self.config.max_consecutive_frame_errors:
                        raise
                    annotated, n_active = frame, 0

                t_now = time.perf_counter()
                dt = t_now - t_last
                t_last = t_now
                if dt > 0:
                    inst = 1.0 / dt
                    a = self.config.fps_ema
                    self.processing_fps = inst if self.processing_fps <= 0 else (1 - a) * self.processing_fps + a * inst

                self.stats_updated.emit(int(n_active), int(self.identity.session_identity_count),
                                        round(float(self.processing_fps), 1))
                self.frame_ready.emit(annotated)

        except Exception as e:
            logger.exception(f"Inference pipeline fatal error: {e}")
            self.error_occurred.emit(f"Inference pipeline error: {str(e)}")
        finally:
            self._shutdown()
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self.worker_finished.emit()

    # ------------------------------------------------------------------ control (any thread)
    def stop(self) -> None:
        """Gracefully signal the worker loop to terminate (profiles are flushed on exit)."""
        self._is_running = False
        self._is_paused = False
        if self.isRunning() and QThread.currentThread() is not self:
            self.wait(1000)

    def pause(self) -> None:
        """Pause frame playback."""
        self._is_paused = True

    def resume(self) -> None:
        """Resume frame playback."""
        self._is_paused = False

    def reset_identities(self) -> None:
        """
        Thread-safe request to forget all identity state (e.g. after the database was cleared).
        Applied at the start of the next frame; applied immediately when the worker is idle.
        """
        self._reset_event.set()
        if not self.isRunning() and self.identity is not None:
            self._apply_pending_reset(force=True)

    # ------------------------------------------------------------------ setup / teardown
    def _init_components(self) -> bool:
        """Create every pipeline component on the worker thread. Emits error_occurred on failure."""
        try:
            self.detector = PedestrianDetector(device=self.device)
            self.parser = HumanParser()
            self.attributes = AttributeExtractor()
            self.pose = PoseBiometrics(device=self.device)
            self.reid = ReIDFeatureExtractor(device=self.device)
            self.identity = IdentityManager(self.db, embedding_model_id=getattr(self.reid, "model_id", "unknown"))
        except Exception as e:
            logger.exception(f"Pipeline initialisation failed: {e}")
            self.error_occurred.emit(f"Pipeline initialisation failed: {e}")
            return False
        self._roi_available = callable(getattr(self.detector, "detect_accessories_in_roi", None))
        self._db_generation = int(getattr(self.db, "clear_generation", 0) or 0)
        if getattr(self.detector, "model", True) is None:
            self.error_occurred.emit("Person detector weights (yolo11n.pt) were not found locally; "
                                     "no people will be detected.")
        logger.info(f"Pipeline ready on {self.camera_id}: ReID '{getattr(self.reid, 'model_id', '?')}', "
                    f"accessory model '{getattr(self.detector, 'accessory_model_name', '?')}'")
        return True

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        if self.source_type == "webcam":
            src: Any = self.source_path if self.source_path not in (None, "") else 0
            try:
                src = int(src)
            except (TypeError, ValueError):
                src = str(src)
            cap = cv2.VideoCapture(src)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low on live cameras
            except Exception:
                pass
            return cap
        return cv2.VideoCapture(str(self.source_path))

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> Optional[np.ndarray]:
        """Return a 3-channel BGR uint8 frame or None for unusable frames."""
        if frame is None or frame.size == 0:
            return None
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim != 3:
            return None
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.shape[2] != 3:
            return None
        return frame

    def _shutdown(self) -> None:
        """End every open track (commits its evidence), forward final events, persist dirty profiles."""
        if self.identity is None:
            return
        try:
            # Holding the DB lock makes the "was the database cleared meanwhile?" check and the final
            # writes atomic with respect to clear_database() (which takes the same lock)
            with self._db_lock():
                if self._db_cleared_since_start() or self._reset_event.is_set():
                    logger.info("Database was cleared: discarding in-memory identity state on exit")
                    self.identity.reset()
                    self._reset_event.clear()
                    self._db_generation = int(getattr(self.db, "clear_generation", 0) or 0)
                else:
                    if self._tracks:
                        self.identity.end_tracks(list(self._tracks.keys()))
                    self.identity.flush()
                    self._forward_events()
        except Exception as e:
            logger.warning(f"Identity flush on shutdown failed: {e}")
        if self.pose is not None and hasattr(self.pose, "forget_track"):
            for tid in list(self._tracks.keys()):
                self.pose.forget_track(tid)
        self._tracks.clear()
        self.track_to_uid.clear()

    def _db_lock(self):
        """The database's re-entrant lock (a no-op context for databases without one)."""
        lock = getattr(self.db, "_lock", None)
        return lock if lock is not None else nullcontext()

    def _db_cleared_since_start(self) -> bool:
        return int(getattr(self.db, "clear_generation", 0) or 0) != self._db_generation

    def _apply_pending_reset(self, force: bool = False) -> None:
        """Apply a reset_identities() request or a database wipe noticed through the clear counter."""
        if not (force or self._reset_event.is_set() or self._db_cleared_since_start()):
            return
        self._reset_event.clear()
        self._db_generation = int(getattr(self.db, "clear_generation", 0) or 0)
        try:
            if self.identity is not None:
                self.identity.reset()
        except Exception as e:
            logger.warning(f"Identity reset failed: {e}")
        if self.pose is not None and hasattr(self.pose, "forget_track"):
            for tid in list(self._tracks.keys()):
                self.pose.forget_track(tid)
        self._tracks.clear()
        self.track_to_uid.clear()
        logger.info("Identity state reset")

    # ------------------------------------------------------------------ per-frame pipeline
    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, int]:
        """Run the full pipeline on one frame. Returns (annotated frame, number of active tracks)."""
        cfg = self.config
        fh, fw = frame.shape[:2]
        t0 = time.perf_counter()

        # 1. Detection + tracking (full-frame accessories included)
        active, archived, frame_accs = self.detector.detect_and_track(frame)
        t_det = time.perf_counter()

        fresh: List[Tuple[Any, Box]] = []      # measured this frame
        coasting: List[Tuple[Any, Box]] = []   # tracker coasts through a missed detection
        for trk in active:
            box = self._clip_box(trk.bbox, fw, fh)
            if box is None:
                continue
            if int(getattr(trk, "lost_frames", 0) or 0) == 0:
                fresh.append((trk, box))
            else:
                coasting.append((trk, box))

        tids = [t.track_id for t, _ in fresh]
        boxes = [b for _, b in fresh]
        for tid in tids:
            st = self._tracks.get(tid)
            if st is None:
                st = _TrackState(track_id=tid, first_frame=frame_idx)
                self._tracks[tid] = st
            st.last_seen_frame = frame_idx

        # 2. Batched pose
        poses = self._run_pose(frame, boxes, tids)
        t_pose = time.perf_counter()

        # 3. Exclusive accessory assignment (each detected item goes to at most one person)
        kp_frame = {tid: p[0] for tid, p in zip(tids, poses) if p[0] is not None}
        assigned = self._assign_accessories({tid: b for tid, b in zip(tids, boxes)}, frame_accs, kp_frame)

        # Observation quality (needed to schedule ReID)
        qualities: Dict[int, float] = {}
        for (trk, box), pose in zip(fresh, poses):
            q = estimate_observation_quality(box, frame.shape, pose[1])
            det_c = float(getattr(trk, "confidence", 1.0) or 0.0)
            qualities[trk.track_id] = float(q * min(1.0, det_c / max(cfg.det_conf_ref, 1e-6)))

        # 4. Attributes (ROI accessory pass throttled / staggered / capped)
        roi_tids = self._schedule_roi(tids, frame_idx)
        all_boxes = {t.track_id: b for t, b in fresh + coasting}
        observations: List[TrackObservation] = []
        crops: Dict[int, np.ndarray] = {}
        for (trk, box), (kf, kc, bio, posture) in zip(fresh, poses):
            tid = trk.track_id
            try:
                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2]
                crops[tid] = crop
                parsed = self.parser.parse_pedestrian(crop, keypoints=kc)
                run_roi = tid in roi_tids
                others = [b for otid, b in all_boxes.items() if otid != tid]
                upper, lower, accs = self.attributes.extract_attributes_scored(
                    frame, box, parsed,
                    accessory_detector=self.detector if (run_roi and self._roi_available) else None,
                    keypoints=kf,
                    keypoints_crop=kc,
                    initial_accessories=assigned.get(tid) or None,
                    exclude_boxes=others,
                )
                if run_roi:
                    self._tracks[tid].last_roi_frame = frame_idx
                self._tracks[tid].biometrics = dict(bio or {})
                observations.append(TrackObservation(
                    track_id=tid,
                    frame_idx=frame_idx,
                    bbox=box,
                    embedding=None,  # filled after the batched ReID pass
                    upper=upper,
                    lower=lower,
                    accessories=accs,
                    accessory_checked=run_roi,
                    quality=qualities.get(tid, 0.0),
                    posture=posture or "Standing",
                    biometrics=dict(bio or {}),
                    crop=crop,
                ))
            except Exception as e:
                self._log_error("track", e, tid)
        t_attr = time.perf_counter()

        # 5. Batched ReID (identifying tracks every frame, identified tracks every reid_interval frames)
        reid_obs = [o for o in observations if self._reid_due(o, frame_idx)]
        zero = np.zeros(int(getattr(self.reid, "embedding_dim", 512) or 512), dtype=np.float32)
        embeddings = None
        if reid_obs:
            try:
                embeddings = self.reid.extract_batch(frame, [o.bbox for o in reid_obs])
            except Exception as e:
                self._log_error("reid", e)
        for i, o in enumerate(reid_obs):
            row = embeddings[i] if embeddings is not None and i < len(embeddings) else None
            if row is not None and float(np.linalg.norm(row)) > 1e-6:
                o.embedding = row
                st = self._tracks[o.track_id]
                st.embedding = row
                st.last_reid_frame = frame_idx
        for o in observations:
            if o.embedding is None:
                o.embedding = zero  # appearance unavailable this frame
        t_reid = time.perf_counter()

        annotated = frame.copy()
        archived_ids = [t.track_id for t in archived]
        assignments: Dict[int, IdentityAssignment] = {}
        # The identity / persistence section holds the DB lock so that a concurrent clear_database()
        # either happens before it (noticed here -> reset) or after it (noticed next frame -> reset),
        # never in the middle of identity writes.
        with self._db_lock():
            if self._db_cleared_since_start():
                self._apply_pending_reset()   # wiped while this frame was being analysed
                t_id = time.perf_counter()
            else:
                # 6. Identity assignment
                try:
                    assignments = self.identity.update(observations, frame_idx)
                except Exception as e:
                    self._log_error("identity", e)

                # 7. Archived tracks release their identities
                if archived_ids:
                    self._end_tracks(archived_ids)
                self._forward_events()
                t_id = time.perf_counter()

                # 8a. Signatures, thumbnails, UI payloads (identified tracks only)
                for trk, box in fresh:
                    tid = trk.track_id
                    a = assignments.get(tid)
                    st = self._tracks.get(tid)
                    if a is None or st is None:
                        continue
                    try:
                        st.assignment = a
                        if a.person_uid:
                            if st.origin_uid != a.person_uid:
                                st.origin_uid = a.person_uid
                                st.origin = "new" if (a.status == "new" and not a.merged_from) else "reacquired"
                            self.track_to_uid[tid] = a.person_uid
                            self._publish_identified(trk, box, a, st, crops.get(tid), frame_idx)
                        else:
                            self.track_to_uid.pop(tid, None)
                    except Exception as e:
                        self._log_error("output", e, tid)

        # 8b. Drawing: coasting tracks with their last identity, then tentative, then identified on top
        for trk, box in coasting:
            st = self._tracks.get(trk.track_id)
            if st is not None and st.assignment is not None:
                self._safe_draw(annotated, trk, box, st.assignment, None)
        pose_by_tid = {tid: p for tid, p in zip(tids, poses)}
        order = sorted(fresh, key=lambda tb: bool(assignments.get(tb[0].track_id) is not None
                                                  and assignments[tb[0].track_id].person_uid))
        for trk, box in order:
            a = assignments.get(trk.track_id)
            if a is not None:
                self._safe_draw(annotated, trk, box, a, pose_by_tid.get(trk.track_id, (None,))[0])
        t_out = time.perf_counter()

        self._record_timing({"detect": t_det - t0, "pose": t_pose - t_det, "attributes": t_attr - t_pose,
                             "reid": t_reid - t_attr, "identity": t_id - t_reid, "output": t_out - t_id,
                             "total": t_out - t0})
        return annotated, len(fresh) + len(coasting)

    # ------------------------------------------------------------------ stages
    def _run_pose(self, frame: np.ndarray, boxes: List[Box], tids: List[int]) -> List[Tuple]:
        """Batched pose; every failure degrades to 'no keypoints' for that frame."""
        default = [(None, None, {}, "Standing") for _ in boxes]
        if not boxes or self.pose is None:
            return default
        try:
            if hasattr(self.pose, "estimate_pose_batch"):
                res = list(self.pose.estimate_pose_batch(frame, boxes, tids))
            else:
                res = [self.pose.estimate_pose(frame, b, t) for b, t in zip(boxes, tids)]
            if len(res) != len(boxes):
                return default
            return res
        except Exception as e:
            self._log_error("pose", e)
            return default

    def _assign_accessories(self, person_boxes: Dict[int, Box], frame_accs: List,
                            kp_frame: Dict[int, np.ndarray]) -> Dict[int, Dict[str, float]]:
        """Frame-level accessory detections -> per-track {name: confidence} (exclusive when supported)."""
        if not person_boxes or not frame_accs:
            return {tid: {} for tid in person_boxes}
        try:
            assign = getattr(self.detector, "assign_accessories", None)
            if callable(assign):
                return assign(person_boxes, frame_accs, kp_frame)
            legacy = getattr(self.detector, "associate_accessories", None)
            if callable(legacy):
                return {tid: {name: 0.5 for name in legacy(b, frame_accs)} for tid, b in person_boxes.items()}
        except Exception as e:
            self._log_error("accessory_assign", e)
        return {tid: {} for tid in person_boxes}

    def _is_identified(self, tid: int) -> bool:
        st = self._tracks.get(tid)
        return st is not None and st.assignment is not None and bool(st.assignment.person_uid)

    def _schedule_roi(self, tids: List[int], frame_idx: int) -> Set[int]:
        """
        Pick the tracks that get the (expensive) ROI accessory pass this frame: identifying tracks every
        `roi_interval_tentative` frames, identified ones every `roi_interval` frames, most overdue first,
        at most `max_roi_per_frame` per frame (the overdue ordering staggers tracks naturally).
        """
        cfg = self.config
        due = []
        for tid in tids:
            st = self._tracks[tid]
            tentative = not self._is_identified(tid)
            interval = cfg.roi_interval_tentative if tentative else cfg.roi_interval
            since = frame_idx - st.last_roi_frame
            if since >= max(1, interval):
                due.append((0 if tentative else 1, -since, tid))
        due.sort()
        return {tid for _, _, tid in due[:max(0, cfg.max_roi_per_frame)]}

    def _reid_due(self, o: TrackObservation, frame_idx: int) -> bool:
        if o.quality < self.config.min_reid_quality:
            return False
        if not self._is_identified(o.track_id):
            return True
        st = self._tracks[o.track_id]
        return frame_idx - st.last_reid_frame >= max(1, self.config.reid_interval)

    def _end_tracks(self, track_ids: List[int]) -> None:
        """Tracker archived these tracks: release their identities and forget worker state."""
        try:
            self.identity.end_tracks(track_ids)
        except Exception as e:
            self._log_error("end_tracks", e)
        for tid in track_ids:
            st = self._tracks.pop(tid, None)
            uid = self.track_to_uid.pop(tid, None)
            if self.pose is not None and hasattr(self.pose, "forget_track"):
                self.pose.forget_track(tid)
            if uid or (st is not None and st.assignment is not None and st.assignment.person_uid):
                uid = uid or st.assignment.person_uid
                self.db.log_event("TRACK_ARCHIVED", f"Track #{tid} archived to pool as {uid} ({self.camera_id})")

    def _forward_events(self) -> None:
        """Drain identity events -> identity_event signal (+ camera, timestamp) and the system log."""
        try:
            events = self.identity.get_events()
        except Exception as e:
            self._log_error("events", e)
            return
        for ev in events:
            payload = dict(ev)
            uid = payload.get("person_uid")
            payload["camera_id"] = self.camera_id
            payload["timestamp"] = time.time()
            payload["identity_color"] = identity_color_hex(uid)
            kind = str(payload.get("type") or "")
            tid = payload.get("track_id")
            if kind == "merged":
                # db.merge_identities already logged IDENTITY_MERGED; just relabel the live mapping
                src = payload.get("merged_from")
                for k, v in list(self.track_to_uid.items()):
                    if v == src:
                        self.track_to_uid[k] = uid
            elif kind in ("new", "reacquired"):
                conf = float(payload.get("confidence") or 0.0)
                text = (f"New identity {uid} registered" if kind == "new"
                        else f"{uid} re-identified (score {conf:.2f})")
                self.db.log_event(f"IDENTITY_{kind.upper()}", f"{text} on {self.camera_id}, track #{tid}")
            self.identity_event.emit(payload)

    # ------------------------------------------------------------------ outputs
    def _publish_identified(self, trk: Any, box: Box, a: IdentityAssignment, st: _TrackState,
                            crop: Optional[np.ndarray], frame_idx: int) -> None:
        """Thumbnail, throttled DB signature and throttled signature_detected for an identified track."""
        cfg = self.config
        crop_path = self._maybe_save_crop(st, a, crop)
        sig = None
        if st.saved_uid != a.person_uid or frame_idx - st.last_sig_frame >= cfg.signature_interval:
            sig = self._build_signature(trk, box, a, st, crop_path)
            if self.db.save_signature(sig):
                st.saved_uid = a.person_uid
                st.last_sig_frame = frame_idx
        if (st.emitted_uid != a.person_uid or st.emitted_status != a.status
                or frame_idx - st.last_emit_frame >= cfg.emit_interval):
            if sig is None:
                sig = self._build_signature(trk, box, a, st, crop_path)
            self.signature_detected.emit(self._signature_payload(sig, a, self._match_score(st, a)))
            st.emitted_uid, st.emitted_status, st.last_emit_frame = a.person_uid, a.status, frame_idx

    def _crop_path_for(self, uid: str) -> Path:
        return self.crops_dir / f"{uid}.jpg"

    def _maybe_save_crop(self, st: _TrackState, a: IdentityAssignment, crop: Optional[np.ndarray]) -> Optional[str]:
        """Write the identity thumbnail when this frame is its best crop or no thumbnail exists yet."""
        uid = a.person_uid
        path = self._crop_path_for(uid)
        exists = st.crop_uid == uid or path.exists()
        if crop is not None and crop.size > 0 and (a.is_best_crop or not exists):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, int(self.config.crop_jpeg_quality)])
                if ok:
                    tmp = path.with_name(path.stem + ".tmp.jpg")
                    buf.tofile(str(tmp))
                    try:
                        os.replace(str(tmp), str(path))  # atomic: the UI never reads a half-written file
                    except OSError:
                        buf.tofile(str(path))
                        try:
                            tmp.unlink()
                        except OSError:
                            pass
                    exists = True
            except Exception as e:
                self._log_error("crop", e, a.track_id)
        if exists:
            st.crop_uid = uid
            return str(path)
        return None

    def _build_signature(self, trk: Any, box: Box, a: IdentityAssignment, st: _TrackState,
                         crop_path: Optional[str]) -> ForensicSignature:
        """Signature from the identity manager's track consensus (colours, accessories, posture)."""
        emb = st.embedding
        emb_list = emb.astype(float).tolist() if emb is not None else [0.0] * 512
        return ForensicSignature(
            person_uid=a.person_uid,
            camera_id=self.camera_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            upper_garment=dict(a.upper),
            lower_garment=dict(a.lower),
            accessories=list(a.accessories),
            biometrics=dict(st.biometrics) if st.biometrics else {"torso_leg_ratio": 0.0, "shoulder_hip_ratio": 0.0,
                                                                  "cadence_hz": 0.0},
            posture=a.posture or "Standing",
            embedding=emb_list,
            track_id=int(trk.track_id),
            bbox=[int(v) for v in box],
            crop_path=crop_path,
            confidence=float(getattr(trk, "confidence", 1.0) or 0.0),
        )

    @staticmethod
    def _match_score(st: Optional[_TrackState], a: IdentityAssignment) -> Optional[float]:
        """Gallery match score of a re-acquired identity; None for identities minted by this track
        (their confidence measures distinctiveness, not a match)."""
        if not a.person_uid or st is None or st.origin_uid != a.person_uid or st.origin != "reacquired":
            return None
        return float(np.clip(a.confidence, 0.0, 1.0))

    def _signature_payload(self, sig: ForensicSignature, a: IdentityAssignment,
                           match_score: Optional[float]) -> Dict[str, Any]:
        payload = sig.to_dict()
        payload["status"] = a.status
        payload["match_score"] = match_score
        payload["display_label"] = a.display_label
        payload["identity_color"] = identity_color_hex(a.person_uid)
        if a.merged_from:
            payload["merged_from"] = a.merged_from
        return payload

    def _safe_draw(self, frame: np.ndarray, trk: Any, box: Box, a: IdentityAssignment,
                   keypoints: Optional[np.ndarray]) -> None:
        try:
            self._draw_track(frame, trk, box, a, keypoints)
        except Exception as e:
            self._log_error("draw", e, trk.track_id)

    def _draw_track(self, frame: np.ndarray, trk: Any, box: Box, a: IdentityAssignment,
                    keypoints: Optional[np.ndarray]) -> None:
        """Skeleton + HUD box in the identity colour (muted dashed box while identifying)."""
        uid = a.person_uid
        tentative = not uid
        if keypoints is not None:
            draw_skeleton(frame, keypoints, color=_TENTATIVE_SKELETON_HEX if tentative else identity_color_hex(uid))
        status = "tentative" if tentative else a.status
        match_score = None if tentative else self._match_score(self._tracks.get(trk.track_id), a)
        draw_forensic_bbox(
            frame,
            bbox=box,
            person_uid=uid or "",
            track_id=trk.track_id,
            upper_color=a.upper.get("color", "Unknown"),
            upper_hex=a.upper.get("hex", "#808080"),
            lower_color=a.lower.get("color", "Unknown"),
            lower_hex=a.lower.get("hex", "#404040"),
            accessories=list(a.accessories),
            posture=a.posture,
            confidence=float(getattr(trk, "confidence", 1.0) or 0.0),
            identity_color=None if tentative else identity_color_bgr(uid),
            status=status,
            match_score=match_score,
            display_label=a.display_label or (TENTATIVE_LABEL if tentative else uid),
        )

    # ------------------------------------------------------------------ helpers
    def _clip_box(self, bbox: Any, fw: int, fh: int) -> Optional[Box]:
        try:
            x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
        except Exception:
            return None
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 - x1 < self.config.min_box_side or y2 - y1 < self.config.min_box_side:
            return None
        return (x1, y1, x2, y2)

    def _log_error(self, stage: str, err: Exception, track_id: Optional[int] = None) -> None:
        """Rate-limited error logging (the loop never stops for per-track failures)."""
        key = f"{stage}:{type(err).__name__}"
        self._error_counts[key] = self._error_counts.get(key, 0) + 1
        now = time.monotonic()
        last = self._error_log_times.get(key)
        where = f" (track #{track_id})" if track_id is not None else ""
        if last is None or now - last >= self.config.error_log_interval_s:
            self._error_log_times[key] = now
            logger.warning(f"Pipeline {stage} error{where}: {err!r} [{self._error_counts[key]} so far]",
                           exc_info=last is None)
        else:
            logger.debug(f"Pipeline {stage} error{where}: {err!r}")

    @property
    def error_counts(self) -> Dict[str, int]:
        """Per-stage error counters (stage:ExceptionType -> count) since the worker started."""
        return dict(self._error_counts)

    def _record_timing(self, stages: Dict[str, float]) -> None:
        for name, sec in stages.items():
            ms = 1000.0 * sec
            prev = self.stage_ms.get(name)
            self.stage_ms[name] = ms if prev is None else 0.9 * prev + 0.1 * ms
