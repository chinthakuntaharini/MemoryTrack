"""
Asynchronous Video Pipeline QThread Worker
Decouples heavy deep learning inference from PyQt6 UI thread to eliminate UI freezes.
"""

import cv2
import time
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import defaultdict, deque, Counter
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

from forensic_app.pipeline.detector import PedestrianDetector
from forensic_app.pipeline.parser import HumanParser
from forensic_app.pipeline.attributes import AttributeExtractor
from forensic_app.pipeline.pose import PoseBiometrics
from forensic_app.pipeline.reid import ReIDFeatureExtractor
from forensic_app.core.signature import ForensicSignature
from forensic_app.core.database import ForensicDatabase
from forensic_app.utils.drawing import draw_forensic_bbox, draw_skeleton

logger = logging.getLogger(__name__)


class ForensicPipelineWorker(QThread):
    """Real-time video inference loop running on a dedicated worker thread."""

    # PyQt6 Signals
    frame_ready = pyqtSignal(np.ndarray)
    signature_detected = pyqtSignal(dict)
    stats_updated = pyqtSignal(int, int, float)  # active_tracks, total_profiles, fps
    worker_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self,
                 db: ForensicDatabase,
                 source_type: str = "video",
                 source_path: Any = None,
                 camera_id: str = "CAM_01",
                 device: str = "cpu"):
        super().__init__()
        self.db = db
        self.source_type = source_type
        self.source_path = source_path
        self.camera_id = camera_id
        self.device = device

        self._is_running = True
        self._is_paused = False

        # Crops storage directory
        self.crops_dir = Path(__file__).resolve().parent.parent.parent / "database" / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        # Track to persistent UID mapping
        self.track_to_uid: Dict[int, str] = {}
        self.next_person_seq = 1

        # Temporal consensus smoothing history per person_uid
        self.person_history: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "upper_colors": deque(maxlen=40),
            "lower_colors": deque(maxlen=40),
            "accessories": deque(maxlen=40),
            "postures": deque(maxlen=30)
        })

    def run(self) -> None:
        """Main execution loop of the worker thread."""
        try:
            # Initialize pipeline components on the worker thread
            detector = PedestrianDetector(device=self.device)
            parser = HumanParser()
            attributes = AttributeExtractor()
            pose_extractor = PoseBiometrics(device=self.device)
            reid_extractor = ReIDFeatureExtractor(device=self.device)

            # Open video source
            if self.source_type == "webcam":
                cap = cv2.VideoCapture(int(self.source_path or 0))
            else:
                cap = cv2.VideoCapture(str(self.source_path))

            if not cap.isOpened():
                self.error_occurred.emit(f"Could not open video stream: {self.source_path}")
                self.worker_finished.emit()
                return

            frame_count = 0
            start_time = time.time()
            fps = 0.0

            while self._is_running:
                if self._is_paused:
                    self.msleep(50)
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                t_frame_start = time.time()

                # Validate frame shape
                if frame is None or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
                    continue

                # 1. Detection and Tracking with full-frame accessory detection
                active_tracks, archived_tracks, frame_accessories = detector.detect_and_track(frame)

                # 2. Process Archived Tracks (Save final state to DB)
                for trk in archived_tracks:
                    uid = self.track_to_uid.get(trk.track_id)
                    if uid:
                        logger_msg = f"Track #{trk.track_id} archived to pool as {uid}"
                        self.db.log_event("TRACK_ARCHIVED", logger_msg)

                # 3. Process Each Active Tracklet
                annotated_frame = frame.copy()

                for trk in active_tracks:
                    try:
                        bx1, by1, bx2, by2 = trk.bbox
                        h_f, w_f = frame.shape[:2]
                        bx1, by1 = max(0, bx1), max(0, by1)
                        bx2, by2 = min(w_f, bx2), min(h_f, by2)

                        if (bx2 - bx1) < 10 or (by2 - by1) < 10:
                            continue

                        person_crop = frame[by1:by2, bx1:bx2]
                        if person_crop.size == 0:
                            continue

                        # 3a. Frame-level accessory association
                        associated_accs = detector.associate_accessories((bx1, by1, bx2, by2), frame_accessories)

                        # 3b. Pose Estimation & Skeletal Biometrics (both full-frame and crop-relative)
                        keypoints_f, keypoints_c, biometrics, posture = pose_extractor.estimate_pose(
                            frame, (bx1, by1, bx2, by2), trk.track_id
                        )

                        # 3c. Human Parsing (Separates upper/lower garments, checks bare legs / shorts)
                        parsed_garments = parser.parse_pedestrian(
                            person_crop,
                            keypoints=keypoints_c
                        )

                        # 3d. CIELAB Dominant Color & Multi-Scale Accessory Detection
                        upper_dict, lower_dict, detected_accs = attributes.extract_attributes(
                            frame, (bx1, by1, bx2, by2), parsed_garments,
                            yolo_model=detector.model,
                            keypoints=keypoints_f,
                            keypoints_crop=keypoints_c,
                            initial_accessories=associated_accs
                        )

                        # 3e. 512-D Re-ID Feature Embedding
                        embedding = reid_extractor.extract(frame, (bx1, by1, bx2, by2))

                        # 3f. Global Person Identity Association (Re-ID Matching)
                        person_uid = self.track_to_uid.get(trk.track_id)
                        if person_uid is None:
                            # Search FAISS & Database for existing person profile
                            matched_uid, match_conf = self.db.match_person(
                                query_embedding=embedding,
                                upper_color=upper_dict.get("color"),
                                lower_color=lower_dict.get("color"),
                                min_similarity=0.58
                            )

                            # Ensure active tracklets in this same frame don't steal the same UID
                            active_uids = {
                                self.track_to_uid[ot.track_id]
                                for ot in active_tracks
                                if ot.track_id in self.track_to_uid and ot.track_id != trk.track_id
                            }

                            if matched_uid is not None and matched_uid not in active_uids:
                                person_uid = matched_uid
                                logger.info(f"Re-ID Match: Track #{trk.track_id} associated with {person_uid} (Sim: {match_conf:.2f})")
                            else:
                                person_uid = f"UID-{self.camera_id}-{self.next_person_seq:04d}"
                                self.next_person_seq += 1
                                logger.info(f"New Person registered: {person_uid} for Track #{trk.track_id}")

                            self.track_to_uid[trk.track_id] = person_uid

                        # 3g. Temporal Multi-Shot Consensus Smoothing
                        hist = self.person_history[person_uid]
                        if upper_dict.get("color") != "Unknown":
                            hist["upper_colors"].append((upper_dict.get("color"), upper_dict.get("hex", "#808080")))
                        if lower_dict.get("color") not in ["Unknown", "Not Detected"]:
                            hist["lower_colors"].append((lower_dict.get("color"), lower_dict.get("hex", "#404040")))
                        hist["accessories"].append(detected_accs)
                        hist["postures"].append(posture)

                        if hist["upper_colors"]:
                            u_counts = Counter([c[0] for c in hist["upper_colors"]])
                            top_u = u_counts.most_common(1)[0][0]
                            for name, hx in reversed(hist["upper_colors"]):
                                if name == top_u:
                                    upper_dict["color"] = top_u
                                    upper_dict["hex"] = hx
                                    break

                        if hist["lower_colors"]:
                            l_counts = Counter([c[0] for c in hist["lower_colors"]])
                            top_l = l_counts.most_common(1)[0][0]
                            for name, hx in reversed(hist["lower_colors"]):
                                if name == top_l:
                                    lower_dict["color"] = top_l
                                    lower_dict["hex"] = hx
                                    break

                        if hist["accessories"]:
                            flat_accs = [a for sub in hist["accessories"] for a in sub]
                            if flat_accs:
                                acc_counts = Counter(flat_accs)
                                min_votes = max(1, int(len(hist["accessories"]) * 0.25))
                                consensus_accs = [acc for acc, count in acc_counts.items() if count >= min_votes]
                                if consensus_accs:
                                    detected_accs = consensus_accs

                        if hist["postures"]:
                            p_counts = Counter(hist["postures"])
                            posture = p_counts.most_common(1)[0][0]

                        # Save crop image every 15 frames for gallery
                        crop_filename = None
                        if frame_count % 15 == 0 or trk.hits == 1:
                            crop_filename = str(self.crops_dir / f"{person_uid}.jpg")
                            cv2.imwrite(crop_filename, person_crop)

                        # 3f. Build Forensic Signature
                        sig = ForensicSignature(
                            person_uid=person_uid,
                            camera_id=self.camera_id,
                            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            upper_garment=upper_dict,
                            lower_garment=lower_dict,
                            accessories=detected_accs,
                            biometrics=biometrics,
                            posture=posture,
                            embedding=embedding.tolist(),
                            track_id=trk.track_id,
                            bbox=list(trk.bbox),
                            crop_path=crop_filename,
                            confidence=trk.confidence
                        )

                        # Persist to database & FAISS index
                        if frame_count % 5 == 0:
                            self.db.save_signature(sig)

                        # Emit signature update signal
                        self.signature_detected.emit(sig.to_dict())

                        # 3g. Visual Rendering (Skeleton + Forensic HUD Bounding Badge)
                        if keypoints_f is not None:
                            draw_skeleton(annotated_frame, keypoints_f)

                        draw_forensic_bbox(
                            annotated_frame,
                            bbox=(bx1, by1, bx2, by2),
                            person_uid=person_uid,
                            track_id=trk.track_id,
                            upper_color=upper_dict.get("color", "Unknown"),
                            upper_hex=upper_dict.get("hex", "#808080"),
                            lower_color=lower_dict.get("color", "Not Detected"),
                            lower_hex=lower_dict.get("hex", "#404040"),
                            accessories=detected_accs,
                            posture=posture,
                            confidence=trk.confidence
                        )

                    except Exception as trk_err:
                        logger.debug(f"Per-track processing exception on track #{trk.track_id}: {trk_err}")

                # Compute FPS
                dt = time.time() - t_frame_start
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

                # Update stats
                total_profiles = len(self.track_to_uid)
                self.stats_updated.emit(len(active_tracks), total_profiles, round(fps, 1))

                # Emit visual frame for canvas display
                self.frame_ready.emit(annotated_frame)

            cap.release()

        except Exception as e:
            logger.error(f"Inference pipeline fatal error: {e}")
            self.error_occurred.emit(f"Inference pipeline error: {str(e)}")
        finally:
            self.worker_finished.emit()

    def stop(self) -> None:
        """Gracefully signal the worker loop to terminate."""
        self._is_running = False
        self.wait(1000)

    def pause(self) -> None:
        """Pause frame playback."""
        self._is_paused = True

    def resume(self) -> None:
        """Resume frame playback."""
        self._is_paused = False
