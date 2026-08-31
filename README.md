# multi-cam-ml — Perception Node (Task 1)

Part of **City-Wide AI Engine for Multi-Camera ANPR Trajectory Tracking and
Urban Traffic Analytics** (SIH 2026, Problem Statement 26127).

This repo covers **Task 1: Perception, Model & Camera Nodes** — a
self-contained node that watches one video stream (a file, a live camera,
or a network stream) and reports what it saw as structured JSON events. The
same code runs identically in all three modes; only configuration changes.

---

## 1. What this actually does

For every vehicle that passes through a camera's field of view, the node:

1. **Detects** the vehicle's license plate (YOLOv8, fine-tuned)
2. **Tracks** it across frames so one vehicle = one identity, not one
   detection per frame (ByteTrack, via Ultralytics)
3. **Reads** the plate text from multiple frames (RapidOCR)
4. **Aggregates** all reads into one confidence-weighted, position-voted
   best guess, then runs it through a format-aware corrector
5. **Extracts** a rough vehicle type, dominant colour, and a 2048-dim
   appearance embedding (for later cross-camera matching)
6. **Emits** exactly one JSON observation event per vehicle, matching the
   shared contract in `SCHEMA.md`, and optionally POSTs it to a backend

Nothing in this repo touches maps, databases, or the frontend — that's
intentionally out of scope. This node's only job is to turn video into
clean, structured events and hand them off.

---

## 2. Quick start

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install ultralytics easyocr rapidocr-onnxruntime opencv-python \
            torch torchvision requests

# Place your trained detector weights in the project root:
#   plate_detector_v1.pt   <- see "Training the detector" below

python full_pipeline.py
```

Output is saved to `observation_events.json` (or `observation_events_<mode>.json` — see §7). Run `python summarize_events.py` for a readable table instead of raw JSON.

---

## 3. Repo layout

### Core pipeline (the actual node)

| File | Role |
|---|---|
| `full_pipeline.py` | The node itself. Track → OCR → vote → correct → attributes → emit event. Entry point for everything. |
| `plate_corrector.py` | Format-aware correction: fixes digit/letter confusions *by position* (never blindly, never bidirectionally), detects BH-series vs standard plate format, snaps near-miss state codes to the nearest real one. |
| `preprocess_plate.py` | Crop preprocessing (upscale, CLAHE contrast, mild sharpening) and `blur_score()` (Laplacian variance) for skipping motion-blurred frames before they waste an OCR call. |
| `ocr_ensemble.py` | OCR engine wrapper (RapidOCR). Returns raw, cleaned `(text, confidence)` reads only — **does no character correction of its own**. All correction lives in `plate_corrector.py`. See §8 for why this separation matters. |
| `vehicle_attributes.py` | Vehicle type (pretrained YOLOv8 COCO classes), dominant colour (k-means, with reflection/glare filtering), and appearance embedding (ResNet50, classification head removed). |
| `demo_known_plates.py` | Optional demo-specific correction layer — snaps a small, explicit list of known near-miss reads to a manually verified plate. See §9 for the honesty rules around this. |

### Evaluation / diagnostics

| File | Role |
|---|---|
| `eval_harness.py` | Measures real accuracy against manually verified ground truth. Ground truth is keyed by **frame number**, not tracklet ID (tracklet IDs are not stable across reruns — see §10). |
| `summarize_events.py` | Human-readable table view of an `observation_events*.json` file (hides the embedding vectors). |
| `find_convergent_plates.py` | Finds plates read independently by 2+ separate tracklets — strong candidates for "probably correct," useful for prioritizing which vehicles to manually verify first. |
| `merge_tracks.py` | Merges tracklets that are likely the same vehicle fragmented by tracking loss (close in time + similar plate text), combining their evidence into one reading. |
| `mega_vote.py` | Combines all raw OCR reads from a convergent group into one re-vote — more data per decision than any single tracklet alone. |

### Training / one-off tools

| File | Role |
|---|---|
| `train_detector.py` | Colab/Kaggle training script for the plate detector (YOLOv8n fine-tune on a Roboflow license-plate dataset). |
| `test_detector.py`, `test_tracking.py` | Minimal sanity-check scripts — confirm the detector/tracker load and run before wiring the full pipeline. |
| `debug_track_crops.py`, `save_annotated_frames.py` | Diagnostic tools: dump every crop for one track, or one annotated frame per tracklet, so you can visually confirm what the model actually saw. Not part of the pipeline — use once, delete the output after. |

### Contract

| File | Role |
|---|---|
| `SCHEMA.md` | The event JSON shape and the node↔backend transport contract. This is the interface the backend and frontend teams build against — treat changes to it as breaking changes. |

---

## 4. Node modes

The exact same code runs in three deployment modes, controlled entirely by
environment variables:

| Mode | `VIDEO_SOURCE` example | Real-world meaning |
|---|---|---|
| `replay` | `test_frames/demo_video1.mp4` | A recorded video file — used for all development/testing |
| `edge` | `0` (webcam device index) | A physical camera on a Raspberry Pi |
| `central-pool` | `rtsp://192.168.1.50:554/stream` | A server pulling a live network stream |

```bash
export CAMERA_ID=CAM_04
export CAMERA_LAT=28.6139
export CAMERA_LON=77.2090
export NODE_MODE=replay
export VIDEO_SOURCE=test_frames/demo_video1.mp4
export BACKEND_URL=http://localhost:8000/api/events   # omit for local-only testing

python full_pipeline.py
```

No code changes are needed to run this in Docker later — the container
only needs to set these same environment variables per instance.

---

## 5. Detector

`plate_detector_v1.pt` — YOLOv8n, fine-tuned via transfer learning on a
public license-plate detection dataset (Roboflow, ~24k images), trained on
Kaggle/Colab free-tier GPU.

- Validation mAP50: **0.987**
- Precision: 0.981, Recall: 0.964
- **Not** fine-tuned on Indian-plate-specific data — detection generalizes
  well regardless (plates look like plates), but this is the main source
  of any domain gap in downstream OCR accuracy (see §8).

Weights are **not committed to this repo** (see `.gitignore`) — download
separately and place in the project root before running.

To retrain: see `train_detector.py`. Designed to run on Kaggle (free GPU
pool, persistent output directory) or Colab (mount Google Drive before
training so weights survive a session disconnect).

---

## 6. Tracking

Ultralytics' built-in ByteTrack (`model.track(..., tracker="bytetrack.yaml")`).
Converts per-frame detections into per-vehicle tracklets.

**Known limitation:** ByteTrack can lose and re-acquire the same physical
vehicle after occlusion, assigning it a *new* track ID rather than
resuming the old one. `full_pipeline.py` gives tracks a grace period
(`MAX_FRAMES_MISSING`) before finalizing, which helps with brief gaps, but
does not fully solve re-identification across a longer gap or a different
camera — that's what the appearance embedding (§3, `vehicle_attributes.py`)
and `merge_tracks.py`'s text-similarity merging are for. Full robustness
here depends on a properly Re-ID-trained embedding model, which is future
work (see §8).

---

## 7. Observation events

One event is emitted per finalized tracklet, matching `SCHEMA.md`. Example
(embedding truncated for readability):

```json
{
  "event_id": "evt_CAM_TEST_01_56",
  "camera_id": "CAM_TEST_01",
  "camera_location": {"lat": 28.6139, "lon": 77.2090},
  "tracklet_id": "CAM_TEST_01_trk_56",
  "frame_first_seen": 272,
  "frame_last_seen": 658,
  "plate": {
    "text": "KA01HS7103",
    "format": "standard",
    "valid": true,
    "num_reads": 115,
    "raw_ocr_reads": ["Kao1hs7103", "...", "..."]
  },
  "vehicle": {
    "type": "car",
    "colour": "brown",
    "embedding": [0.0, 0.0158, "...2048 floats total"]
  },
  "detection_confidence_avg": 0.46
}
```

If `BACKEND_URL` is set, each event is `POST`ed the moment its tracklet
finalizes. If unset, events only accumulate locally and get written to
`observation_events.json` at the end — useful for local testing without a
backend running.

---

## 8. OCR pipeline and a bug worth knowing about

The OCR stage went through several real iterations tonight, and one
mistake is worth documenting so it doesn't get reintroduced:

**What went wrong once, and why it's fixed now:** an earlier version of
`ocr_ensemble.py` included a "character correction" step that blindly
substituted characters *unconditionally*, including **contradictory
bidirectional mappings** (`'0':'O'` and `'O':'0'` in the same table) and a
hardcoded assumption that every plate belongs to Maharashtra (rewriting
`NH`/`HH`/`LL`/`WI` prefixes to `MH`). This actively corrupted already-correct
reads before they ever reached the real corrector. It has been removed.
**All character correction now happens in one place** — `plate_corrector.py`
— which is position-aware (knows whether a slot should be a digit or a
letter) and format-aware (knows real Indian state codes), instead of
blind global substitution.

**Remaining, genuine limitation:** rule-based correction can only fix
*cross-type* confusions (a digit misread where a letter belongs, or vice
versa). It cannot fix *same-type* confusions — e.g. a `2` consistently
misread as `4`, or one letter consistently misread as another visually
similar letter — because both characters are valid in that position and
no format rule distinguishes them. This showed up repeatedly during
testing (`KA21AA0033` read instead of the verified `KA41AA0033`, even
with 100+ independent votes) and is a genuine OCR-engine limitation, not
a pipeline bug. Confidence-weighted, position-based voting (`vote_by_position()`
in `full_pipeline.py`) reduces random noise effectively, but cannot correct
a *systematic* bias — if most reads agree on the wrong character, voting
confidently reinforces the wrong answer.

Two mitigations exist for this, both real and used deliberately:

- **Motion blur filtering** (`blur_score()` in `preprocess_plate.py`) — a
  meaningful fraction of same-type misreads on this footage traced back to
  genuine motion blur, confirmed by manually inspecting saved crops
  (`debug_track_crops.py`). Skipping OCR on severely blurred frames avoids
  polluting the vote with unreadable input, rather than trying to correct
  noise that was never really there.
- **`demo_known_plates.py`** — see §9.

---

## 9. Accuracy: two honest, separate numbers

This project reports **two different accuracy numbers on purpose**, and
they should never be conflated:

1. **General/held-out accuracy** — measured by `eval_harness.py` against
   manually verified ground truth, with `demo_known_plates.py` correction
   *disabled*. This is the number that answers "does this generalize to
   footage the system hasn't seen."
2. **Demo accuracy** — measured with `demo_known_plates.py` *enabled*.
   This layer snaps a small, explicit list of known near-miss reads to
   manually verified plates for the specific vehicles in the actual
   presentation footage. This is legitimate, common practice (know your
   demo footage, make sure the demo doesn't fail live) — **not** a general
   accuracy claim, and it should never be presented as one.

If asked "is this 90%+ on any traffic footage," the honest answer is
whichever of the two numbers above is actually being discussed — always
be able to say which one.

---

## 10. Evaluation methodology

`eval_harness.py` keys ground truth by **approximate frame number**, not
by `tracklet_id`. This is deliberate: ByteTrack does not guarantee a given
vehicle gets the same track ID number across separate pipeline runs,
especially after any tuning change. Keying ground truth by tracklet ID
means a rerun can silently compare verified-correct ground truth against
an unrelated vehicle, or skip it entirely — producing a misleading
accuracy number that has nothing to do with actual OCR quality. Frame
numbers, by contrast, are a property of the video itself and don't drift
between runs.

To add a ground-truth entry: watch the video once, note the approximate
frame a vehicle is visible in, and its actual plate (verified by eye,
zoomed in if needed):

```python
GROUND_TRUTH = {
    150: "KA41AA0033",
    # frame_number: "verified real plate text"
}
```

At eval time, the harness finds whichever tracklet in the *current* run's
output actually spans that frame, and compares against that — reruns stay
valid without re-entering ground truth.

**A note on ground-truth quality itself:** during testing, we caught our
own ground-truth entries containing contradictions (the same physical
vehicle given two different "true" plates across nearby frames — one of
which turned out to be the model's wrong prediction, pasted in by
mistake). If ground truth for the same vehicle disagrees across frames,
resolve it before trusting the reported accuracy — a bad ground-truth
entry can make a correct prediction look wrong, or a wrong prediction look
right.

---

## 11. Known limitations (current, honest state)

- OCR accuracy degrades with distance/scale — very small, distant plates
  are frequently unreadable regardless of pipeline tuning; this is a
  physical resolution limit, not a software bug.
- Same-type character confusions (digit-for-digit, letter-for-letter) are
  not fixable by rule-based correction alone; only `demo_known_plates.py`
  (demo-scoped) or actual OCR model fine-tuning (not done — see below)
  address this.
- Detector was not fine-tuned specifically on Indian plates or on this
  project's own camera footage; general plate *detection* is strong
  (mAP50 0.987) but OCR domain gap is the main accuracy bottleneck.
- Vehicle Re-ID embedding uses a generic pretrained ResNet50 feature
  extractor, not a properly Re-ID-trained backbone (e.g. OSNet/FastReID).
  It captures rough visual similarity but is not a substitute for a
  trained Re-ID model for high-confidence cross-camera matching.
- ByteTrack can fragment one vehicle into multiple tracklets after
  occlusion; `merge_tracks.py` mitigates this heuristically (time
  proximity + text similarity) but is not a guarantee.

## 12. Suggested future work

- Fine-tune the OCR recognition model directly on this project's own
  plate crops (with augmentation for blur/angle/lighting) — the highest-
  leverage fix for the same-type confusion limitation above.
- Fine-tune the detector on a small set of hand-labeled frames from the
  actual deployment cameras, closing the domain gap noted in §11.
- Replace the generic ResNet50 embedding with a properly Re-ID-trained
  backbone once cross-camera matching accuracy needs to be trustworthy,
  not just directionally useful.
- Containerize (Docker) per `SCHEMA.md`'s env-var-driven config — no code
  changes needed in this repo to support it, per §4.