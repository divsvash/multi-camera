import cv2
import os

VIDEO_PATH = "test_frames/demo_video1.mp4"

FRAMES = [
    6,
    57,
    108,
    168,
    357,
    450,
    738,
    762,
    783,
    903,
    969,
    1065,
    1161,
    1167,
    1170,
    1266,
    1272,
    1299,
    1320,
    1344,
    1350,
    1533,
    1596,
    1764,
    1812,
    1848,
]

os.makedirs("gt_inspection_frames", exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

for frame_number in FRAMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ok, frame = cap.read()

    if not ok:
        print(f"FAILED: frame {frame_number}")
        continue

    path = f"gt_inspection_frames/frame_{frame_number:04d}.jpg"
    cv2.imwrite(path, frame)

    print(f"saved {path}")

cap.release()