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
    # Garment dicts: {"color", "hex"} plus optional "lab" [L, a, b], "confidence" and (lower) "type"
    upper_garment: Dict[str, Any] = field(default_factory=lambda: {"color": "Unknown", "hex": "#808080"})
    lower_garment: Dict[str, Any] = field(default_factory=lambda: {"color": "Not Detected", "hex": "#404040"})
    accessories: List[str] = field(default_factory=list)
    biometrics: Dict[str, float] = field(default_factory=lambda: {"torso_leg_ratio": 0.0, "shoulder_hip_ratio": 0.0, "cadence_hz": 0.0})
    posture: str = "Standing"
    embedding: List[float] = field(default_factory=lambda: [0.0] * 512)
    track_id: int = 0
    bbox: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    crop_path: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert signature to a standard Python dictionary (JSON-serialisable)."""
        d = asdict(self)
        # Ensure embedding / Lab values are plain lists of floats
        if isinstance(d.get("embedding"), np.ndarray):
            d["embedding"] = d["embedding"].astype(float).tolist()
        for key in ("upper_garment", "lower_garment"):
            garment = d.get(key)
            if isinstance(garment, dict) and isinstance(garment.get("lab"), np.ndarray):
                garment["lab"] = [float(v) for v in garment["lab"]]
        if isinstance(d.get("bbox"), (tuple, np.ndarray)):
            d["bbox"] = [int(v) for v in d["bbox"]]
        return d

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize signature to standard JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForensicSignature":
        """Reconstruct signature object from a dictionary."""
        return cls(
            person_uid=data.get("person_uid") or "UID-UNKNOWN",
            camera_id=data.get("camera_id") or "CAM_01",
            timestamp=data.get("timestamp") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            upper_garment=dict(data.get("upper_garment") or {"color": "Unknown", "hex": "#808080"}),
            lower_garment=dict(data.get("lower_garment") or {"color": "Not Detected", "hex": "#404040"}),
            accessories=list(data.get("accessories") or []),
            biometrics=dict(data.get("biometrics") or {"torso_leg_ratio": 0.0, "shoulder_hip_ratio": 0.0, "cadence_hz": 0.0}),
            posture=data.get("posture") or "Standing",
            embedding=list(data.get("embedding") if data.get("embedding") is not None else [0.0] * 512),
            track_id=int(data.get("track_id") or 0),
            bbox=list(data.get("bbox") or [0, 0, 0, 0]),
            crop_path=data.get("crop_path"),
            confidence=float(data.get("confidence", 1.0) if data.get("confidence") is not None else 1.0)
        )

    def get_embedding_numpy(self) -> np.ndarray:
        """Return the embedding as a float32 unit-normalized numpy array (zeros when invalid)."""
        arr = np.nan_to_num(np.asarray(self.embedding, dtype=np.float32).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr
