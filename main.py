"""
MemoryTrack - Main CLI Pipeline Runner
Orchestrates the complete missing person tracking pipeline.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import MemoryTrack components
from utils.config_loader import ConfigLoader
from utils.db_manager import DatabaseManager
from utils.video_loader import VideoLoader, SingleVideoLoader, CameraFrame
from utils.visualization import Visualizer

from core.detector import PersonDetector, MultiObjectTracker
from core.pose_extractor import PoseExtractor
from core.reid_extractor import ReIDExtractor
from core.color_extractor import ColorExtractor
from core.feature_fusion import FeatureFusion
from core.memory_bank import AdaptiveMemoryBank
from core.accessory_extractor import AccessoryExtractor, AccessoryFeatures
from core.occlusion_detector import OcclusionDetector
from core.xai import ExplanationGenerator


class MemoryTrackPipeline:
    """Main pipeline orchestrator for MemoryTrack system."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MemoryTrack pipeline.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = ConfigLoader(config_path)
        self.config.validate()

        # Initialize database
        db_config = self.config.get_database_config()
        self.db = DatabaseManager(db_config.get('path'))

        # Initialize components
        self._init_detector()
        self._init_feature_extractors()
        self._init_fusion()
        self._init_memory_bank()
        self._init_visualizer()
        self._init_auxiliary()

        logger.info("MemoryTrack pipeline initialized")

    def _init_detector(self):
        """Initialize detection and tracking components."""
        det_config = self.config.get_detection_config()
        track_config = self.config.get_tracking_config()

        self.detector = PersonDetector(
            model_path=det_config.get('model_path', 'yolo11n.pt'),
            confidence_threshold=det_config.get('confidence_threshold', 0.5),
            nms_threshold=det_config.get('nms_threshold', 0.45),
            device=det_config.get('device', 'cpu')
        )

        self.tracker = MultiObjectTracker(
            max_age=track_config.get('max_age', 30),
            min_hits=track_config.get('min_hits', 3),
            iou_threshold=track_config.get('iou_threshold', 0.3)
        )

        logger.info("Detector and tracker initialized")

    def _init_feature_extractors(self):
        """Initialize feature extraction components."""
        feat_config = self.config.get_feature_extraction_config()

        # Pose extractor
        pose_config = feat_config.get('pose', {})
        self.pose_extractor = PoseExtractor(
            model_complexity=pose_config.get('model_complexity', 1),
            min_detection_confidence=pose_config.get('min_detection_confidence', 0.5)
        )

        # ReID extractor
        reid_config = feat_config.get('reid', {})
        self.reid_extractor = ReIDExtractor(
            model_name=reid_config.get('model_name', 'osnet_x0_25'),
            weights_path=reid_config.get('weights_path'),
            embedding_dim=reid_config.get('embedding_dim', 512),
            device=reid_config.get('device', 'cpu')
        )

        # Color extractor
        color_config = feat_config.get('color', {})
        self.color_extractor = ColorExtractor(
            hist_bins=color_config.get('hist_bins', 16),
            regions=color_config.get('regions', ['upper_body', 'lower_body'])
        )

        # Accessory extractor
        acc_config = feat_config.get('accessories', {})
        self.accessory_extractor = AccessoryExtractor(
            confidence_threshold=acc_config.get('confidence_threshold', 0.4),
            accessory_classes=acc_config.get('classes', None)
        )

        logger.info("Feature extractors initialized")

    def _init_fusion(self):
        """Initialize feature fusion module."""
        weights = self.config.get_fusion_weights()
        self.feature_fusion = FeatureFusion(weights=weights)
        logger.info("Feature fusion initialized")

    def _init_memory_bank(self):
        """Initialize adaptive memory bank."""
        mem_config = self.config.get_memory_bank()

        self.memory_bank = AdaptiveMemoryBank(
            embedding_dim=mem_config.get('embedding_dim', 720),
            decay_rate=mem_config.get('decay_rate', 0.0001),
            max_snapshots_per_person=mem_config.get('max_snapshots_per_person', 10),
            update_threshold=mem_config.get('update_threshold', 0.8)
        )

        logger.info("Memory bank initialized")

    def _init_visualizer(self):
        """Initialize visualizer."""
        self.visualizer = Visualizer(
            show_features=True,
            show_confidence=True,
            show_trajectory=True
        )
        logger.info("Visualizer initialized")

    def _init_auxiliary(self):
        """Initialize auxiliary components (occlusion, XAI)."""
        self.occlusion_detector = OcclusionDetector()
        self.explanation_generator = ExplanationGenerator()
        logger.info("Auxiliary components initialized")

    def process_frame(self, frame: np.ndarray, camera_id: str = "cam_0",
                     frame_number: int = 0) -> dict:
        """
        Process a single frame through the complete pipeline.

        Args:
            frame: Input frame
            camera_id: Camera ID
            frame_number: Frame number

        Returns:
            Dictionary with processing results
        """
        results = {
            'camera_id': camera_id,
            'frame_number': frame_number,
            'detections': [],
            'tracks': {},
            'features': {},
            'matches': []
        }

        try:
            # Step 1: Detection (persons + accessories)
            detections = self.detector.detect(frame)
            results['detections'] = detections

            if not detections:
                logger.debug("No detections in frame")
                return results

            # Step 2: Tracking
            tracks = self.tracker.update(detections)
            results['tracks'] = tracks

            if not tracks:
                logger.debug("No active tracks")
                return results

            # Detect accessories for the whole frame
            accessory_detections = self.detector.detect_with_accessories(
                frame,
                accessory_classes=self.accessory_extractor.accessory_classes
            )

            # Collect person bboxes for occlusion detection
            bboxes = [
                (track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2)
                for track in tracks.values() if track.state != 'lost'
            ]

            # Step 3: Feature extraction for each track
            for track_id, track in tracks.items():
                if track.state == 'lost':
                    continue

                bbox = (track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2)

                # Occlusion detection
                other_bboxes = [
                    b for b in bboxes if b != bbox
                ]
                occlusion = self.occlusion_detector.detect(bbox, other_bboxes)

                # Extract pose
                pose_features = self.pose_extractor.extract(frame, bbox)

                # Extract ReID features
                reid_features = self.reid_extractor.extract(frame, bbox)

                # Extract color features
                color_features = None
                if pose_features:
                    color_features = self.color_extractor.extract(
                        frame, bbox,
                        pose_features.keypoints,
                        pose_features.keypoint_confidences
                    )

                # Extract accessory features
                accessory_features = self.accessory_extractor.extract(
                    frame, bbox,
                    detections=accessory_detections
                )

                # Extract motion features from track
                motion_features = self._extract_motion_features(track)

                # Store features
                track_features = {
                    'reid': reid_features,
                    'pose': self.pose_extractor.extract_feature_vector(pose_features) if pose_features else None,
                    'color': self.color_extractor.extract_feature_vector(color_features) if color_features else None,
                    'accessory': self.accessory_extractor.extract_feature_vector(accessory_features),
                    'motion': motion_features
                }

                results['features'][track_id] = track_features

                # Step 4: Feature fusion with occlusion-aware weights
                valid_features = {k: v for k, v in track_features.items()
                                  if v is not None and len(v) > 0}

                if valid_features:
                    # Build confidences per modality
                    confidences = self._build_modality_confidences(
                        pose_features, color_features, accessory_features, track
                    )

                    fusion_result = self.feature_fusion.fuse(
                        valid_features,
                        confidences=confidences,
                        occlusion_flags=occlusion.occlusion_flags
                    )

                    # Step 5: Search in memory bank
                    matches = self.memory_bank.search(
                        fusion_result.fused_vector,
                        top_k=3
                    )

                    if matches:
                        # Generate explanations for matches
                        for match in matches:
                            stored_modalities = match.snapshot_used.modality_features
                            if stored_modalities:
                                explanation = self.explanation_generator.generate(
                                    valid_features,
                                    stored_modalities,
                                    match.confidence
                                )
                                match.explanation = explanation.summary
                                match.modality_similarities = explanation.modality_contributions

                        results['matches'].extend(matches)

                        # Add to memory bank if high confidence track
                        if fusion_result.confidence > 0.7:
                            self.memory_bank.add_profile(
                                person_id=track_id,
                                features=fusion_result.fused_vector,
                                camera_id=camera_id,
                                confidence=fusion_result.confidence,
                                modality_features=valid_features,
                                metadata={
                                    'modality_similarities':
                                        matches[0].modality_similarities if matches else {},
                                    'explanation':
                                        matches[0].explanation if matches else ''
                                }
                            )

                    # Step 6: Visualization
                    frame = self.visualizer.draw_bbox(frame, track.bbox, track_id)

                    if pose_features:
                        frame = self.visualizer.draw_pose(frame, pose_features, bbox)

                    frame = self.visualizer.draw_trajectory(frame, track)

                    if occlusion.is_occluded:
                        # Draw occlusion warning
                        cv2.putText(
                            frame,
                            f"Occluded: {','.join(occlusion.affected_modalities)}",
                            (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 255),
                            1
                        )

            results['processed_frame'] = frame

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            results['error'] = str(e)

        return results

    def _build_modality_confidences(self, pose_features, color_features,
                                   accessory_features, track) -> Dict[str, float]:
        """
        Build confidence scores for each modality.

        Args:
            pose_features: PoseFeatures object or None
            color_features: ColorFeatures object or None
            accessory_features: AccessoryFeatures object
            track: Track object

        Returns:
            Dictionary of modality -> confidence
        """
        confidences = {
            'reid': track.bbox.confidence,
            'pose': pose_features.confidence if pose_features else 0.0,
            'color': color_features.confidence if color_features else 0.0,
            'accessory': accessory_features.confidence,
            'motion': 0.8  # Motion is generally reliable when track is active
        }
        return confidences

    def _extract_motion_features(self, track) -> np.ndarray:
        """
        Extract motion features from track.

        Args:
            track: Track object

        Returns:
            Motion feature vector (16-dimensional)
        """
        features = np.zeros(16, dtype=np.float32)

        if len(track.history) >= 2:
            # Velocity
            features[0] = track.velocity[0]
            features[1] = track.velocity[1]

            # Speed
            speed = np.sqrt(track.velocity[0]**2 + track.velocity[1]**2)
            features[2] = speed

            # Direction
            if speed > 0:
                features[3] = np.arctan2(track.velocity[1], track.velocity[0])

            # Recent positions (last 5 points)
            recent_history = track.history[-5:]
            for i, (x, y) in enumerate(recent_history):
                if i < 6:
                    features[4 + i*2] = x
                    features[5 + i*2] = y

        return features

    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     show_display: bool = True) -> dict:
        """
        Process a video file through the complete pipeline.

        Args:
            video_path: Path to video file
            output_path: Optional path to save output video
            show_display: Whether to display video during processing

        Returns:
            Summary statistics dictionary
        """
        logger.info(f"Processing video: {video_path}")

        # Initialize video loader
        video_loader = SingleVideoLoader(video_path, camera_id="main")

        # Initialize video writer if output path specified
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        summary = {
            'video_path': video_path,
            'frames_processed': 0,
            'total_matches': 0,
            'avg_fps': 0.0
        }

        frame_count = 0
        start_time = time.time()

        try:
            for camera_frame in video_loader:
                frame = camera_frame.frame

                # Process frame
                results = self.process_frame(
                    frame,
                    camera_id=camera_frame.camera_id,
                    frame_number=camera_frame.frame_number
                )

                # Get processed frame with visualizations
                processed_frame = results.get('processed_frame', frame)

                # Add camera info
                fps = frame_count / (time.time() - start_time + 1e-6)
                processed_frame = self.visualizer.draw_camera_info(
                    processed_frame,
                    camera_frame.camera_id,
                    camera_frame.frame_number,
                    fps
                )

                # Initialize video writer
                if output_path and video_writer is None:
                    h, w = processed_frame.shape[:2]
                    video_writer = cv2.VideoWriter(
                        output_path, fourcc, 30.0, (w, h)
                    )

                # Write frame
                if video_writer:
                    video_writer.write(processed_frame)

                # Display frame
                if show_display:
                    cv2.imshow('MemoryTrack', processed_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User requested quit")
                        break

                frame_count += 1
                summary['total_matches'] += len(results.get('matches', []))

                # Log progress
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count} frames")

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")

        finally:
            # Cleanup
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()

            elapsed_time = time.time() - start_time
            avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            logger.info(f"Video processing complete:")
            logger.info(f"  Frames processed: {frame_count}")
            logger.info(f"  Elapsed time: {elapsed_time:.2f}s")
            logger.info(f"  Average FPS: {avg_fps:.2f}")

            # Print memory bank statistics
            stats = self.memory_bank.get_statistics()
            logger.info(f"Memory bank statistics:")
            logger.info(f"  Total profiles: {stats['total_profiles']}")
            logger.info(f"  Total snapshots: {stats['total_snapshots']}")

            summary['frames_processed'] = frame_count
            summary['avg_fps'] = avg_fps

        return summary

    def process_batch(self, video_paths: List[str], output_dir: Optional[str] = None,
                     show_display: bool = False) -> Dict[str, dict]:
        """
        Process multiple videos in batch mode.

        Args:
            video_paths: List of video file paths
            output_dir: Optional directory to save output videos
            show_display: Whether to display video during processing

        Returns:
            Dictionary of video_path -> summary statistics
        """
        logger.info(f"Batch processing {len(video_paths)} videos")

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        all_results = {}
        total_start = time.time()

        for video_path in video_paths:
            output_file = None
            if output_dir:
                video_name = Path(video_path).stem
                output_file = str(Path(output_dir) / f"{video_name}_processed.mp4")

            summary = self.process_video(
                video_path,
                output_path=output_file,
                show_display=show_display
            )

            all_results[video_path] = summary

        total_time = time.time() - total_start
        total_frames = sum(r['frames_processed'] for r in all_results.values())
        total_matches = sum(r['total_matches'] for r in all_results.values())

        logger.info("=" * 50)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info(f"  Videos processed: {len(video_paths)}")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  Total matches: {total_matches}")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info("=" * 50)

        return all_results

    def process_webcam(self, camera_index: int = 0) -> None:
        """
        Process webcam stream in real-time.

        Args:
            camera_index: Webcam device index
        """
        logger.info(f"Processing webcam stream from camera {camera_index}")

        video_loader = SingleVideoLoader(camera_index, camera_id="webcam")

        frame_count = 0
        start_time = time.time()

        try:
            for camera_frame in video_loader:
                frame = camera_frame.frame

                # Process frame
                results = self.process_frame(
                    frame,
                    camera_id=camera_frame.camera_id,
                    frame_number=camera_frame.frame_number
                )

                # Get processed frame
                processed_frame = results.get('processed_frame', frame)

                # Add camera info
                fps = frame_count / (time.time() - start_time + 1e-6)
                processed_frame = self.visualizer.draw_camera_info(
                    processed_frame,
                    camera_frame.camera_id,
                    camera_frame.frame_number,
                    fps
                )

                # Display frame
                cv2.imshow('MemoryTrack - Webcam', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested quit")
                    break

                frame_count += 1

                # Log progress
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count} frames at {fps:.1f} FPS")

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")

        finally:
            cv2.destroyAllWindows()
            logger.info("Webcam processing complete")

    def cleanup(self):
        """Cleanup resources."""
        self.pose_extractor.close()
        self.reid_extractor.close()
        self.db.close()
        logger.info("Pipeline cleanup complete")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='MemoryTrack - Adaptive Memory-Based Missing Person Tracking System'
    )

    # Input sources
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Path to video file')
    input_group.add_argument('--webcam', type=int, default=0, help='Webcam device index')
    input_group.add_argument('--batch', type=str, nargs='+',
                             help='List of video file paths for batch processing')

    # Output options
    parser.add_argument('--output', type=str, help='Path to save output video')
    parser.add_argument('--output-dir', type=str,
                        help='Directory to save batch output videos')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display during processing')

    # Configuration
    parser.add_argument('--config', type=str, help='Path to configuration file')

    # Other options
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Initialize pipeline
    try:
        pipeline = MemoryTrackPipeline(config_path=args.config)

        # Process based on input type
        if args.batch:
            pipeline.process_batch(
                video_paths=args.batch,
                output_dir=args.output_dir,
                show_display=not args.no_display
            )
        elif args.video:
            pipeline.process_video(
                video_path=args.video,
                output_path=args.output,
                show_display=not args.no_display
            )
        elif args.webcam is not None:
            pipeline.process_webcam(camera_index=args.webcam)

        # Cleanup
        pipeline.cleanup()

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
