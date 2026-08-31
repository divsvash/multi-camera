# ============================================================
# Full node pipeline (single camera, replay mode):
# Track -> OCR each frame of each track -> extract vehicle
# embedding/attributes periodically -> aggregate into ONE
# observation event per completed tracklet, matching SCHEMA.md.
#
# Efficiency notes (matters once this runs on real edge hardware):
#  - OCR runs every 5th frame per track, not every frame
#  - Attributes (embedding/type/colour) run every 10th frame -
#    they're expensive (2 extra model forward passes) and don't
#    change fast, so no need to run them as often as OCR
#  - Position-based character voting instead of whole-string
#    majority vote for OCR aggregation
#  - Tracks get a grace period before finalizing, to survive
#    brief occlusion without fragmenting into multiple events
# ============================================================

from ultralytics import YOLO
from ocr_ensemble import read_plate_ensemble
import cv2
import json
import re
from collections import defaultdict, Counter

from preprocess_plate import preprocess_plate_crop
from plate_corrector import correct_plate
from vehicle_attributes import get_vehicle_attributes

import os
import requests

# --- Config (env vars, matching SCHEMA.md - docker-ready, no code changes needed
# when the container person wires this up later) ---
CAMERA_ID = os.environ.get("CAMERA_ID", "CAM_TEST_01")
CAMERA_LAT = float(os.environ.get("CAMERA_LAT", 28.6139))
CAMERA_LON = float(os.environ.get("CAMERA_LON", 77.2090))
VIDEO_PATH = os.environ.get("VIDEO_SOURCE", "test_frames/demo_video1.mp4")
BACKEND_URL = os.environ.get("BACKEND_URL")  # e.g. http://backend:8000/api/events - None = local-only mode

MAX_FRAMES_MISSING = 60     # grace period before a track is finalized
FRAME_LIMIT = None          # full video - no cap, this is a real run
BOX_PADDING_RATIO = 0.15     # expand plate box before cropping, avoids half-cut plates
OCR_EVERY_N_FRAMES = 5       # MIN gap between OCR calls per track (perf guard) -
                              # quality gating below decides WHICH frame in that
                              # window actually gets OCR'd, instead of always the Nth
ATTRS_EVERY_N_FRAMES = 10    # how often to run embedding/type/colour per active track
MIN_CONFIDENCE_FOR_OCR = 0.25  # skip OCR below this - low-conf crops rarely yield
                                 # usable reads and just add noise to the vote
MIN_PLATE_CROP_WIDTH_PX = 80     # below this, there's not enough resolution for OCR
                                  # to have a real shot - measured post-padding
MIN_SHARPNESS = 60.0              # Laplacian variance floor. Motion-blurred/out-of-focus
                                   # crops score far below sharp ones (~10x+ in testing).
                                   # TUNE THIS against your own footage before trusting it -
                                   # print sharpness_score() values on a sample of your
                                   # crops first and pick a threshold that actually
                                   # separates your readable ones from your blurry ones.
MIN_PLATE_ASPECT_RATIO = 1.8       # width/height. Indian plates run roughly 2:1-6:1.
MAX_PLATE_ASPECT_RATIO = 6.5       # Boxes outside this range are very unlikely to be
                                    # an actual plate (false detection, occluded plate,
                                    # or box locked onto something else nearby) - reject
                                    # before they ever reach OCR instead of feeding OCR
                                    # garbage and hoping voting saves it.

detector = YOLO("plate_detector_v1.pt")

active_tracks = defaultdict(lambda: {
    "raw_reads": [],
    "det_confidences": [],
    "attribute_samples": [],
    "first_seen": None,
    "last_seen": None,
    "observed_frames": [],
    "last_ocr_frame": None,
    "observations": []
})

finished_events = []


def sharpness_score(crop):
    """Laplacian variance - a cheap, fast proxy for focus/motion-blur.
    Sharp, in-focus crops score much higher than blurred ones (verified
    ~10-13x separation on synthetic sharp-vs-blurred test crops)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def vote_by_position(raw_reads_with_conf):
    """
    Confidence-weighted voting: the most common character at each position,
    where each read's 'vote' is weighted by its OCR confidence rather than
    counting every read equally. A high-confidence read should outweigh
    several low-confidence noisy ones.
    """
    if not raw_reads_with_conf:
        return None

    cleaned = []
    for text, conf in raw_reads_with_conf:
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        if clean_text:
            cleaned.append((clean_text, conf))
    if not cleaned:
        return None

    lengths = [len(t) for t, c in cleaned]
    target_len = max(set(lengths), key=lengths.count)
    same_len = [(t, c) for t, c in cleaned if len(t) == target_len]
    if not same_len:
        same_len = cleaned
        target_len = max(len(t) for t, c in same_len)

    voted = ""
    for i in range(target_len):
        char_weights = defaultdict(float)
        for text, conf in same_len:
            if i < len(text):
                char_weights[text[i]] += conf  # weight by confidence, not just count
        if char_weights:
            voted += max(char_weights, key=char_weights.get)
    return voted


def aggregate_attributes(attribute_samples):
    """Majority-vote type/colour across samples, keep the most recent embedding."""
    if not attribute_samples:
        return {"type": "unknown", "colour": "unknown", "embedding": None}

    types = [a["type"] for a in attribute_samples if a["type"] != "unknown"]
    colours = [a["colour"] for a in attribute_samples if a["colour"] != "unknown"]

    best_type = Counter(types).most_common(1)[0][0] if types else "unknown"
    best_colour = Counter(colours).most_common(1)[0][0] if colours else "unknown"
    latest_embedding = attribute_samples[-1]["embedding"]  # most recent = most representative of exit view

    return {"type": best_type, "colour": best_colour, "embedding": latest_embedding}


def finalize_track(track_id, track_data):
    """Turn a completed track's accumulated data into one observation event (SCHEMA.md shape)."""
    raw_reads_with_conf = [r for r in track_data["raw_reads"] if r[0]]
    raw_reads = [r[0] for r in raw_reads_with_conf]  # text only, for the JSON output

    if not raw_reads_with_conf:
        best_plate, plate_format, plate_valid = None, None, False
    else:
        voted_text = vote_by_position(raw_reads_with_conf)
        result = correct_plate(voted_text) if voted_text else {"corrected": None, "format": "unknown", "valid": False}
        best_plate, plate_format, plate_valid = result["corrected"], result["format"], result["valid"]

    avg_det_conf = sum(track_data["det_confidences"]) / len(track_data["det_confidences"])
    vehicle_attrs = aggregate_attributes(track_data["attribute_samples"])

    event = {
        "event_id": f"evt_{CAMERA_ID}_{track_id}",
        "camera_id": CAMERA_ID,
        "camera_location": {"lat": CAMERA_LAT, "lon": CAMERA_LON},
        "tracklet_id": f"{CAMERA_ID}_trk_{track_id}",
        "frame_first_seen": track_data["first_seen"],
        "frame_last_seen": track_data["last_seen"],
        "observed_frames": track_data["observed_frames"],
        "observations": track_data["observations"],
        "plate": {
            "text": best_plate,
            "format": plate_format,
            "valid": plate_valid,
            "num_reads": len(raw_reads),
            "raw_ocr_reads": raw_reads,
        },
        "vehicle": {
            "type": vehicle_attrs["type"],
            "colour": vehicle_attrs["colour"],
            "embedding": vehicle_attrs["embedding"],
        },
        "detection_confidence_avg": round(avg_det_conf, 2),
    }

    finished_events.append(event)
    print(f"\n>>> FINALIZED tracklet {track_id}: plate={best_plate} "
          f"(valid={plate_valid}), vehicle={vehicle_attrs['type']}/{vehicle_attrs['colour']}, "
          f"{len(raw_reads)} OCR reads, frames {track_data['first_seen']}-{track_data['last_seen']}\n")

    # Send to backend immediately, per SCHEMA.md's transport contract.
    # If BACKEND_URL isn't set (local dev/testing), just skip sending -
    # events still accumulate in finished_events and get saved to disk at the end.
    if BACKEND_URL:
        try:
            response = requests.post(BACKEND_URL, json=event, timeout=2)
            if response.status_code == 200:
                print(f"    -> sent to backend OK ({event['event_id']})")
            else:
                print(f"    -> backend responded {response.status_code} for {event['event_id']}")
        except requests.exceptions.RequestException as err:
            # Don't crash the node if the backend is down - log and keep going.
            # Real deployments would queue/retry; not required for the demo.
            print(f"    -> WARNING: could not reach backend ({err})")


results_generator = detector.track(
    source=VIDEO_PATH,
    persist=True,
    tracker="bytetrack_india.yaml",
    stream=True
)

frame_idx = 0
seen_this_frame = set()
pending_removal = {}

for result in results_generator:
    frame = result.orig_img

    if frame is None:
        continue

    seen_this_frame.clear()

    if result.boxes.id is not None:
        track_ids = result.boxes.id.int().tolist()
        boxes_xyxy = result.boxes.xyxy.tolist()
        det_confidences = result.boxes.conf.tolist()

        for track_id, box, det_conf in zip(track_ids, boxes_xyxy, det_confidences):
            seen_this_frame.add(track_id)
            pending_removal.pop(track_id, None)

            x1, y1, x2, y2 = map(int, box)
            box_w, box_h = x2 - x1, y2 - y1
            pad_x, pad_y = int(box_w * BOX_PADDING_RATIO), int(box_h * BOX_PADDING_RATIO)
            frame_h, frame_w = frame.shape[:2]

            x1p, y1p = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2p, y2p = min(frame_w, x2 + pad_x), min(frame_h, y2 + pad_y)
            plate_crop = frame[y1p:y2p, x1p:x2p]
            if plate_crop.size == 0:
                continue

            # Shape sanity check: reject boxes that don't look like a plate at all
            # BEFORE they're used as evidence. A near-square or sliver-thin box is
            # very unlikely to be an actual plate - feeding it to OCR just produces
            # garbage that voting can't recover from (this is what was happening
            # to tracks with unrelated-looking OCR output like 'RoKagtaaoo').
            crop_h_raw, crop_w_raw = plate_crop.shape[:2]
            aspect_ratio = crop_w_raw / crop_h_raw if crop_h_raw > 0 else 0
            if not (MIN_PLATE_ASPECT_RATIO <= aspect_ratio <= MAX_PLATE_ASPECT_RATIO):
                print(f"    [reject] track {track_id} frame {frame_idx}: "
                      f"bad aspect ratio {aspect_ratio:.2f} (box {crop_w_raw}x{crop_h_raw})")
                continue

            track_data = active_tracks[track_id]
            if track_data["first_seen"] is None:
                track_data["first_seen"] = frame_idx
            track_data["last_seen"] = frame_idx
            track_data["observed_frames"].append(frame_idx)
            track_data["observations"].append({
    "frame": frame_idx,
    "bbox": [x1, y1, x2, y2],
})
            track_data["det_confidences"].append(det_conf)

            # Quality-gated OCR trigger: don't run OCR on a schedule regardless of
            # crop quality - only spend OCR budget on frames that are actually
            # sharp and large enough to plausibly read, at most once every
            # OCR_EVERY_N_FRAMES per track (perf guard, edge devices are slow).
            crop_w = plate_crop.shape[1]
            enough_gap = (
                track_data["last_ocr_frame"] is None
                or frame_idx - track_data["last_ocr_frame"] >= OCR_EVERY_N_FRAMES
            )
            if (
                enough_gap
                and det_conf >= MIN_CONFIDENCE_FOR_OCR
                and crop_w >= MIN_PLATE_CROP_WIDTH_PX
                and sharpness_score(plate_crop) >= MIN_SHARPNESS
            ):
                crop_for_ocr = preprocess_plate_crop(plate_crop)
                ensemble_results = read_plate_ensemble(crop_for_ocr)
                for raw_text, avg_conf in ensemble_results:
                    track_data["raw_reads"].append((raw_text, avg_conf))
                track_data["last_ocr_frame"] = frame_idx

            if frame_idx % ATTRS_EVERY_N_FRAMES == 0:
                # wider region around the plate as a rough vehicle-body proxy
                vy1 = max(0, y1 - box_h * 3)
                vy2 = min(frame_h, y2 + box_h * 1)
                vx1 = max(0, x1 - box_w * 2)
                vx2 = min(frame_w, x2 + box_w * 2)
                vehicle_region = frame[vy1:vy2, vx1:vx2]
                if vehicle_region.size > 0:
                    # DEBUG: save what the colour/type extractor actually sees
                    import os
                    os.makedirs("debug_vehicle_crops", exist_ok=True)
                    cv2.imwrite(f"debug_vehicle_crops/track{track_id}_frame{frame_idx}.jpg", vehicle_region)

                    attrs = get_vehicle_attributes(vehicle_region)
                    track_data["attribute_samples"].append(attrs)

    for track_id in list(active_tracks.keys()):
        if track_id not in seen_this_frame and track_id not in pending_removal:
            pending_removal[track_id] = frame_idx

    for track_id in list(pending_removal.keys()):
        frames_since_seen = frame_idx - active_tracks[track_id]["last_seen"]
        if frames_since_seen > MAX_FRAMES_MISSING:
            finalize_track(track_id, active_tracks[track_id])
            del active_tracks[track_id]
            del pending_removal[track_id]

    frame_idx += 1
    if FRAME_LIMIT is not None and frame_idx > FRAME_LIMIT:
        break

for track_id, track_data in active_tracks.items():
    finalize_track(track_id, track_data)

print(f"\n=== DONE: {len(finished_events)} total observation events ===")
with open("observation_events.json", "w") as f:
    json.dump(finished_events, f, indent=2)
print("Saved to observation_events.json")