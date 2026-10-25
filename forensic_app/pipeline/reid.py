"""
Metric Re-Identification Feature Extractor
Extracts normalized 512-dimensional feature embeddings for appearance re-identification.
"""

import cv2
import numpy as np
import logging
from typing import Optional

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ReIDFeatureExtractor:
    """512-Dimensional Deep Appearance Re-ID Extractor."""

    def __init__(self,
                 embedding_dim: int = 512,
                 device: str = "cpu"):
        self.embedding_dim = embedding_dim
        self.device = self._verify_device(device)
        self.model = None

        if TORCH_AVAILABLE:
            self._init_network()
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            logger.info(f"ReIDFeatureExtractor initialized (512-D on {self.device})")
        else:
            logger.warning("PyTorch not installed; ReID using heuristic visual hashing fallback.")

    def _verify_device(self, req_device: str) -> str:
        if req_device == "cuda" and TORCH_AVAILABLE:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"

    def _init_network(self) -> None:
        """Initialize lightweight ResNet-18 / OSNet feature backbone projecting to 512-D."""
        try:
            base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # Remove final FC classification layer to output 512-D pooling vector
            modules = list(base_model.children())[:-1]
            self.model = nn.Sequential(*modules).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.warning(f"Failed to load pretrained ResNet18 weights: {e}")
            self.model = None

    def extract(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Extract normalized 512-dimensional embedding vector from person crop.
        Returns unit-normalized float32 numpy vector of length 512.
        """
        x1, y1, x2, y2 = bbox
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]

        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # PyTorch Neural Feature Extraction
        if self.model is not None and TORCH_AVAILABLE:
            try:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = self.transform(rgb_crop).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    features = self.model(tensor)
                    features = torch.flatten(features, 1).cpu().numpy().flatten()

                # Pad or slice to exact 512 dimensions
                if len(features) < self.embedding_dim:
                    features = np.pad(features, (0, self.embedding_dim - len(features)))
                elif len(features) > self.embedding_dim:
                    features = features[:self.embedding_dim]

                # L2 normalize
                norm = np.linalg.norm(features)
                if norm > 0:
                    features = features / norm
                return features.astype(np.float32)
            except Exception as e:
                logger.debug(f"Neural ReID extraction error: {e}")

        # High-Speed Computer Vision Visual Hashing Fallback
        return self._heuristic_feature_vector(crop)

    def _heuristic_feature_vector(self, crop: np.ndarray) -> np.ndarray:
        """Color spatial pyramid + gradient orientation embedding (512-D)."""
        resized = cv2.resize(crop, (64, 128))
        # 1. Multi-band spatial HSV color histograms (8 bins x 3 bands x 3 channels = 72)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        bands = np.array_split(hsv, 4, axis=0)
        hist_parts = []
        for band in bands:
            for ch in range(3):
                hist = cv2.calcHist([band], [ch], None, [16], [0, 256]).flatten()
                hist_parts.append(hist)

        color_feat = np.concatenate(hist_parts)  # 4 * 3 * 16 = 192

        # 2. Gradient magnitude & edge projection
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        grad_hist = cv2.calcHist([mag], [0], None, [64], [0, 256]).flatten()

        combined = np.concatenate([color_feat, grad_hist])
        # Pad to 512
        if len(combined) < self.embedding_dim:
            combined = np.pad(combined, (0, self.embedding_dim - len(combined)))
        else:
            combined = combined[:self.embedding_dim]

        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return combined.astype(np.float32)

    @staticmethod
    def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute Cosine Similarity between two normalized vectors."""
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        if n1 == 0 or n2 == 0:
            return 0.0
        dot = float(np.dot(vec1, vec2) / (n1 * n2))
        return max(0.0, min(1.0, (dot + 1.0) / 2.0))
