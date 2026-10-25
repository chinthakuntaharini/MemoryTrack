"""
Forensic Person Signature Entity Schema
Implements the standardized JSON-serializable forensic biometric trait model.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class ForensicSignature:
    """Standardized composite forensic identity representation."""
    person_uid: str
    camera_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    upper_garment: Dict[str, str] = field(default_factory=lambda: {"color": "Unknown", "hex": "#808080"})
    lower_garment: Dict[str, str] = field(default_factory=lambda: {"color": "Not Detected", "hex": "#404040"})
    accessories: List[str] = field(default_factory=list)
    biometrics: Dict[str, float] = field(default_factory=lambda: {"torso_leg_ratio": 0.0, "shoulder_hip_ratio": 0.0, "cadence_hz": 0.0})
    posture: str = "Standing"
    embedding: List[float] = field(default_factory=lambda: [0.0] * 512)
    track_id: int = 0
    bbox: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    crop_path: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert signature to a standard Python dictionary."""
        d = asdict(self)
        # Ensure embedding is list of floats
        if isinstance(d.get("embedding"), np.ndarray):
            d["embedding"] = d["embedding"].tolist()
        return d

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize signature to standard JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForensicSignature":
        """Reconstruct signature object from a dictionary."""
        return cls(
            person_uid=data.get("person_uid", "UID-UNKNOWN"),
            camera_id=data.get("camera_id", "CAM_01"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
            upper_garment=data.get("upper_garment", {"color": "Unknown", "hex": "#808080"}),
            lower_garment=data.get("lower_garment", {"color": "Not Detected", "hex": "#404040"}),
            accessories=data.get("accessories", []),
            biometrics=data.get("biometrics", {"torso_leg_ratio": 0.0, "shoulder_hip_ratio": 0.0, "cadence_hz": 0.0}),
            posture=data.get("posture", "Standing"),
            embedding=data.get("embedding", [0.0] * 512),
            track_id=data.get("track_id", 0),
            bbox=data.get("bbox", [0, 0, 0, 0]),
            crop_path=data.get("crop_path"),
            confidence=data.get("confidence", 1.0)
        )

    def get_embedding_numpy(self) -> np.ndarray:
        """Return 512-D embedding as a float32 unit-normalized numpy array."""
        arr = np.array(self.embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr
