import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Starting import test...")
from core import (
    AdaptiveMemoryBank,
    AccessoryExtractor,
    OcclusionDetector,
    ExplanationGenerator,
    Explanation
)
print("Core imports OK")

import numpy as np

# Test memory bank
bank = AdaptiveMemoryBank(embedding_dim=720)
feat = np.random.rand(720).astype(np.float32)
bank.add_profile(person_id=1, features=feat, camera_id="cam_1", confidence=0.9,
                 modality_features={'reid': np.random.rand(512).astype(np.float32)})
print("Memory bank add_profile OK")
print("Profiles:", len(bank.profiles))

# Test accessory extractor
acc = AccessoryExtractor()
print("AccessoryExtractor OK, vector_dim:", acc.vector_dim)

# Test occlusion detector
occ = OcclusionDetector()
print("OcclusionDetector OK")

# Test XAI
xai = ExplanationGenerator()
print("ExplanationGenerator OK")

# Test search with modality features in metadata
query = np.random.rand(720).astype(np.float32)
results = bank.search(query, top_k=1)
print("Search results:", len(results))
if results:
    print("Match person_id:", results[0].person_id)

print("ALL IMPORTS AND BASIC FUNCTIONS WORK")

