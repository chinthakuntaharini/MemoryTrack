"""
Generates a realistic synthetic test surveillance video for pipeline testing.
Features 2 moving pedestrians:
- Person 1: Wearing Navy Blue shirt and Beige pants, carrying a dark backpack.
- Person 2: Wearing Red shirt and bare legs / shorts (no pants).
"""

import cv2
import numpy as np
from pathlib import Path

output_path = Path(__file__).resolve().parent.parent / "tests" / "test_surveillance_stream.mp4"
output_path.parent.mkdir(parents=True, exist_ok=True)

width, height = 640, 480
fps = 20
num_frames = 60

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

for f in range(num_frames):
    # Surveillance background (grey concrete pavement with lighting)
    frame = np.full((height, width, 3), 140, dtype=np.uint8)
    cv2.line(frame, (0, 300), (640, 300), (100, 100, 100), 2)  # Wall/ground line

    # --- Person 1 (Walking left to right) ---
    p1_x = 60 + f * 5
    p1_y = 120

    # Head
    cv2.circle(frame, (p1_x + 25, p1_y + 20), 16, (140, 160, 210), -1)  # Skin tone
    # Upper Body: Navy Blue shirt (BGR: 62, 23, 15)
    cv2.rectangle(frame, (p1_x + 10, p1_y + 36), (p1_x + 40, p1_y + 110), (62, 23, 15), -1)
    # Lower Body: Beige pants (BGR: 187, 212, 224)
    cv2.rectangle(frame, (p1_x + 12, p1_y + 110), (p1_x + 24, p1_y + 210), (187, 212, 224), -1)
    cv2.rectangle(frame, (p1_x + 26, p1_y + 110), (p1_x + 38, p1_y + 210), (187, 212, 224), -1)
    # Backpack on back (BGR: 20, 20, 20)
    cv2.rectangle(frame, (p1_x - 6, p1_y + 45), (p1_x + 10, p1_y + 95), (20, 20, 20), -1)

    # --- Person 2 (Walking right to left) ---
    p2_x = 540 - f * 4
    p2_y = 130

    # Head with Dark Cap (BGR: 15, 15, 15)
    cv2.circle(frame, (p2_x + 25, p2_y + 20), 15, (140, 160, 210), -1)  # Face skin
    cv2.rectangle(frame, (p2_x + 10, p2_y + 4), (p2_x + 40, p2_y + 18), (15, 15, 15), -1)  # Cap
    # Upper Body: Red shirt (BGR: 38, 38, 220)
    cv2.rectangle(frame, (p2_x + 10, p2_y + 35), (p2_x + 40, p2_y + 105), (38, 38, 220), -1)
    # Lower Body: Bare Legs / Shorts (Shorts down to y+120, then bare skin)
    cv2.rectangle(frame, (p2_x + 12, p2_y + 105), (p2_x + 38, p2_y + 122), (200, 200, 200), -1)  # White gym shorts
    # Bare Legs (Skin tone BGR: 140, 160, 210)
    cv2.rectangle(frame, (p2_x + 13, p2_y + 122), (p2_x + 23, p2_y + 205), (140, 160, 210), -1)
    cv2.rectangle(frame, (p2_x + 27, p2_y + 122), (p2_x + 37, p2_y + 205), (140, 160, 210), -1)

    out.write(frame)

out.release()
print(f"Generated synthetic video at {output_path}")
