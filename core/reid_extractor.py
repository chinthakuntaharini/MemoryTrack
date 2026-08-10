"""
ReID extractor module for MemoryTrack system.
Implements TorchReID-based appearance feature extraction for person re-identification.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import logging

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
except ImportError:
    raise ImportError("PyTorch not found. Install with: pip install torch torchvision")

logger = logging.getLogger(__name__)


class ReIDExtractor:
    """TorchReID-based appearance embedding extractor."""
    
    # Standard ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(self, model_name: str = "osnet_x0_25",
                 weights_path: Optional[str] = None,
                 embedding_dim: int = 512,
                 device: str = "cuda"):
        """
        Initialize ReID feature extractor.
        
        Args:
            model_name: Name of the ReID model ('osnet_x0_25', 'resnet50', etc.)
            weights_path: Path to pretrained weights file
            embedding_dim: Dimension of output embedding
            device: Device to run inference ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = self._check_device(device)
        
        self.model = self._load_model(weights_path)
        self.transform = self._get_transform()
        
        logger.info(f"ReIDExtractor initialized with {model_name} on {self.device}")
    
    def _check_device(self, device: str) -> str:
        """
        Check if specified device is available, fallback to CPU if not.
        
        Args:
            device: Requested device
            
        Returns:
            Available device
        """
        if device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                return "cpu"
        return device
    
    def _load_model(self, weights_path: Optional[str]) -> nn.Module:
        """
        Load ReID model.
        
        Args:
            weights_path: Path to pretrained weights
            
        Returns:
            Loaded model
        """
        try:
            # Try to use TorchReID if available
            try:
                from torchreid import models
                model = models.build_model(
                    name=self.model_name,
                    num_classes=1,  # We don't need classification
                    pretrained=weights_path is None
                )
                
                if weights_path and Path(weights_path).exists():
                    state_dict = torch.load(weights_path, map_location='cpu')
                    model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Loaded weights from {weights_path}")
                
            except ImportError:
                # Fallback to simple ResNet if TorchReID not available
                logger.warning("TorchReID not available, using fallback ResNet model")
                model = self._build_fallback_model()
            
            model = model.to(self.device)
            model.eval()
            
            logger.info(f"ReID model loaded: {self.model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load ReID model: {e}")
            # Return fallback model
            return self._build_fallback_model()
    
    def _build_fallback_model(self) -> nn.Module:
        """
        Build a fallback ResNet-50 model for ReID.
        
        Returns:
            ResNet-50 model modified for feature extraction
        """
        from torchvision.models import resnet50
        
        model = resnet50(pretrained=True)
        
        # Remove the final classification layer
        model = nn.Sequential(*list(model.children())[:-1])
        
        # Add adaptive pooling to get fixed size output
        model = nn.Sequential(
            model,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, self.embedding_dim)
        )
        
        model = model.to(self.device)
        model.eval()
        
        logger.info("Using fallback ResNet-50 model")
        return model
    
    def _get_transform(self) -> transforms.Compose:
        """
        Get image preprocessing transform.
        
        Returns:
            Composed transform
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # Standard ReID image size
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        tensor = self.transform(image_rgb)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def extract(self, frame: np.ndarray,
                bbox: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
        """
        Extract appearance embedding from frame.
        
        Args:
            frame: Input frame (BGR format)
            bbox: Optional bounding box (x1, y1, x2, y2) to crop to person
            
        Returns:
            Feature vector (embedding_dim-dimensional)
        """
        try:
            # Crop to bounding box if provided
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                frame = frame[int(y1):int(y2), int(x1):int(x2)]
            
            # Ensure frame is not empty
            if frame.size == 0:
                logger.warning("Empty frame provided for ReID extraction")
                return np.zeros(self.embedding_dim, dtype=np.float32)
            
            # Resize if too small
            h, w = frame.shape[:2]
            if h < 32 or w < 16:
                frame = cv2.resize(frame, (128, 256))
            
            # Preprocess
            input_tensor = self.preprocess(frame)
            
            # Extract features
            with torch.no_grad():
                embedding = self.model(input_tensor)
            
            # Convert to numpy and normalize
            embedding = embedding.cpu().numpy().flatten()
            embedding = self._normalize(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"ReID extraction failed: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        L2 normalize vector.
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def extract_batch(self, frames: list,
                     bboxes: Optional[list] = None) -> np.ndarray:
        """
        Extract embeddings from multiple frames in batch.
        
        Args:
            frames: List of input frames
            bboxes: Optional list of bounding boxes
            
        Returns:
            Array of shape (N, embedding_dim)
        """
        embeddings = []
        
        for i, frame in enumerate(frames):
            bbox = bboxes[i] if bboxes else None
            embedding = self.extract(frame, bbox)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def compute_similarity(self, embedding1: np.ndarray,
                          embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Normalize embeddings
        emb1_norm = self._normalize(embedding1)
        emb2_norm = self._normalize(embedding2)
        
        # Compute cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        return float(similarity)
    
    def close(self) -> None:
        """Release model resources."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("ReIDExtractor closed")


class SimpleReIDExtractor:
    """
    Simplified ReID extractor using precomputed features.
    Fallback for when deep learning models are not available.
    """
    
    def __init__(self, embedding_dim: int = 512):
        """
        Initialize simple ReID extractor.
        
        Args:
            embedding_dim: Dimension of output embedding
        """
        self.embedding_dim = embedding_dim
        logger.info("SimpleReIDExtractor initialized (fallback mode)")
    
    def extract(self, frame: np.ndarray,
                bbox: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
        """
        Extract simple color-based features as fallback.
        
        Args:
            frame: Input frame
            bbox: Optional bounding box
            
        Returns:
            Feature vector
        """
        # Crop to bounding box if provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            frame = frame[int(y1):int(y2), int(x1):int(x2)]
        
        # Resize to standard size
        frame = cv2.resize(frame, (128, 256))
        
        # Convert to HSV and compute histogram
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Compute histogram for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [64], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [64], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [64], [0, 256])
        
        # Normalize histograms
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Concatenate and pad/truncate to embedding_dim
        features = np.concatenate([hist_h, hist_s, hist_v])
        
        if len(features) < self.embedding_dim:
            features = np.pad(features, (0, self.embedding_dim - len(features)))
        elif len(features) > self.embedding_dim:
            features = features[:self.embedding_dim]
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
