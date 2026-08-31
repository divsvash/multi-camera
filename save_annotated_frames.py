# ============================================================
# Saves one annotated frame per tracklet, with ONLY that
# tracklet's box drawn (not other cars in the same frame),
# so there's zero ambiguity about which plate belongs to
# which tracklet when you fill in ground truth.
# ============================================================

from ultralytics import YOLO
import cv2
import json
import os

VIDEO_PATH = "test_frames/demo_video1.mp4"
EVENTS_PATH = "observation_events.json"
OUTPUT_DIR = "ground_truth_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(EVENTS_PATH, "r") as f:
    events = json.load(f)

# For each tracklet, we need to know the mid frame AND run tracking
# (not plain detection) so we get consistent track IDs matching
# what full_pipeline.py originally assigned.
frames_needed = {}
for event in events:
    tracklet_id = event["tracklet_id"]
    track_num = int(tracklet_id.split("_trk_")[1])
    first = event["frame_first_seen"]
    last = event["frame_last_seen"]
    mid_frame = (first + last) // 2
    frames_needed.setdefault(mid_frame, []).append((track_num, tracklet_id))

detector = YOLO("plate_detector_v1.pt")

results_generator = detector.track(
    source=VIDEO_PATH,
    persist=True,
    tracker="bytetrack.yaml",
    stream=True
)

cap = cv2.VideoCapture(VIDEO_PATH)

frame_idx = 0
saved_count = 0
max_frame_needed = max(frames_needed.keys())

print(f"Looking for frames covering {len(events)} tracklets...")

for result in results_generator:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx in frames_needed:
        wanted_tracks = frames_needed[frame_idx]  # list of (track_num, tracklet_id)

        if result.boxes.id is not None:
            track_ids_here = result.boxes.id.int().tolist()
            boxes_xyxy = result.boxes.xyxy.tolist()

            for track_num, tracklet_id in wanted_tracks:
                # start from a CLEAN copy of the frame each time -
                # so only ONE box gets drawn, not all detections
                clean_frame = frame.copy()

                found = False
                for tid, box in zip(track_ids_here, boxes_xyxy):
                    if tid == track_num:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(clean_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        found = True

                label = f"TRACKLET: {tracklet_id}  (frame {frame_idx})"
                if not found:
                    label += "  [WARNING: box not found in this exact frame]"

                cv2.putText(
                    clean_frame, label,
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3
                )

                out_path = os.path.join(OUTPUT_DIR, f"{tracklet_id}_frame{frame_idx}.jpg")
                cv2.imwrite(out_path, clean_frame)
                saved_count += 1
                print(f"Saved: {out_path}  {'' if found else '(WARNING: track not visible this frame)'}")

    frame_idx += 1
    if frame_idx > max_frame_needed:
        break

print(f"\nDone. Saved {saved_count} annotated frames to {OUTPUT_DIR}/")
print("Each image has ONLY that tracklet's box drawn in red - so whatever")
print("plate is inside the red box is the one you should read for that tracklet.")
print("If you see a 'WARNING: box not found' image, that frame didn't have")
print("this track visible - open the next/prior saved frame for that ID instead,")
print("or just scrub the video yourself to a frame where that car IS visible.")