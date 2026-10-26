"""
Forensic Attribute Catalog
Canonical accessory vocabulary shared by the detection pipeline, identity matcher, and query UI.
"""

from typing import Dict, FrozenSet, List

# Canonical accessory name -> human readable label (also defines UI ordering)
ACCESSORY_LABELS: Dict[str, str] = {
    "backpack": "Backpack",
    "handbag": "Handbag / Shoulder Bag",
    "suitcase": "Suitcase / Luggage",
    "umbrella": "Umbrella",
    "cap": "Cap / Hat",
    "helmet": "Helmet",
    "glasses": "Glasses / Sunglasses",
    "face mask": "Face Mask",
    "tie": "Tie",
    "bottle": "Bottle",
    "cell phone": "Cell Phone",
    "laptop": "Laptop",
}

# COCO class id -> canonical accessory name (detectable by stock YOLO COCO weights)
COCO_ACCESSORY_CLASSES: Dict[int, str] = {
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    39: "bottle",
    67: "cell phone",
    73: "laptop",
}

# Worn / carried items that stay with a person across sightings (strong identity cues)
IDENTITY_STABLE_ACCESSORIES: FrozenSet[str] = frozenset({
    "backpack", "handbag", "suitcase", "cap", "helmet", "glasses", "face mask", "tie",
})

# Hand-held items that come and go between sightings (weak identity cues)
TRANSIENT_ACCESSORIES: FrozenSet[str] = frozenset({
    "umbrella", "bottle", "cell phone", "laptop",
})

# Open-vocabulary detector prompts -> canonical name (only used when an optional YOLO-World model is present)
OPEN_VOCAB_PROMPTS: Dict[str, str] = {
    "backpack": "backpack",
    "handbag": "handbag",
    "shoulder bag": "handbag",
    "tote bag": "handbag",
    "suitcase": "suitcase",
    "umbrella": "umbrella",
    "baseball cap": "cap",
    "hat": "cap",
    "helmet": "helmet",
    "sunglasses": "glasses",
    "eyeglasses": "glasses",
    "face mask": "face mask",
    "necktie": "tie",
    "bottle": "bottle",
    "cell phone": "cell phone",
    "laptop": "laptop",
}

# Lower garment presence classes emitted by the human parser
LOWER_GARMENT_TYPES = ("Pants/Jeans", "Bare Legs / Shorts", "None / Occluded")

# Per-class minimum detector confidence for an accessory to be reported in a frame.
# Bags are often detected with modest confidence by small models (keep recall), while small hand-held
# objects (bottle, phone) are the main source of false positives with yolo11n (need stronger evidence).
ACCESSORY_CONF_THRESHOLDS: Dict[str, float] = {
    "backpack": 0.30,
    "handbag": 0.28,
    "suitcase": 0.35,
    "umbrella": 0.35,
    "tie": 0.40,
    "bottle": 0.45,
    "cell phone": 0.45,
    "laptop": 0.45,
    "cap": 0.50,
    "helmet": 0.50,
    "glasses": 0.55,
    "face mask": 0.55,
}
DEFAULT_ACCESSORY_CONF_THRESHOLD = 0.40

# Synonyms accepted from legacy records, queries, and detector class names
_ACCESSORY_ALIASES: Dict[str, str] = {
    "hat": "cap",
    "headwear": "cap",
    "headwear / cap": "cap",
    "baseball cap": "cap",
    "bag": "handbag",
    "shoulder bag": "handbag",
    "purse": "handbag",
    "luggage": "suitcase",
    "sunglasses": "glasses",
    "eyeglasses": "glasses",
    "mask": "face mask",
    "phone": "cell phone",
    "mobile phone": "cell phone",
    "necktie": "tie",
}


def normalize_accessory(name: str) -> str:
    """Map any accessory spelling (legacy, query, or detector label) to its canonical name."""
    key = (name or "").strip().lower()
    return _ACCESSORY_ALIASES.get(key, key)


def accessory_label(name: str) -> str:
    """Return the display label for an accessory name."""
    canonical = normalize_accessory(name)
    return ACCESSORY_LABELS.get(canonical, canonical.title())


def accessory_threshold(name: str) -> float:
    """Minimum per-frame detector confidence for an accessory class (canonical or alias name)."""
    return ACCESSORY_CONF_THRESHOLDS.get(normalize_accessory(name), DEFAULT_ACCESSORY_CONF_THRESHOLD)


def sort_accessories(names) -> List[str]:
    """Deduplicate canonical accessory names and order them like ACCESSORY_LABELS (unknown names last)."""
    order = {n: i for i, n in enumerate(ACCESSORY_LABELS)}
    uniq = {normalize_accessory(n) for n in names if n}
    return sorted(uniq, key=lambda n: (order.get(n, len(order)), n))
