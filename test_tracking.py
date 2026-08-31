# ============================================================
# Within-camera tracking — turns per-frame detections into
# tracklets (one ID per vehicle across its time in frame).
# ============================================================

from ultralytics import YOLO
import cv2

model = YOLO("plate_detector_v1.pt")

VIDEO_PATH = "test_frames/demo_video1.mp4"  # <-- point this at an actual video file

# model.track() instead of model() — this is what enables tracking.
# persist=True keeps track IDs consistent across frames in this stream.
results_generator = model.track(
    source=VIDEO_PATH,
    persist=True,
    tracker="bytetrack.yaml",   # ships with ultralytics, no extra install needed
    stream=True                  # process frame-by-frame instead of loading whole video into memory
)

# --- Just watch what tracking gives you, frame by frame ---
for frame_idx, result in enumerate(results_generator):
    if result.boxes.id is None:
        # no tracked objects this frame
        continue

    track_ids = result.boxes.id.int().tolist()
    classes = result.boxes.cls.int().tolist()
    confidences = result.boxes.conf.tolist()

    for track_id, cls_id, conf in zip(track_ids, classes, confidences):
        class_name = model.names[cls_id]
        print(f"Frame {frame_idx}: Track ID {track_id} | {class_name} | conf {conf:.2f}")

    # Stop early for a quick sanity check — remove this once you trust it works
    if frame_idx > 100:
        break

print("\nDone. Look at the Track IDs above — the SAME vehicle across frames")
print("should keep the SAME ID number. That's the whole point of tracking.")