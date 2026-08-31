import json
import re
from collections import defaultdict

# ============================================================
# EVALUATION CONFIG
# ============================================================

GT_FPS = 20.0
PIPELINE_FPS = 60.0

# Minimum IoU between the manually identified GT vehicle box
# and a pipeline detection for us to call it the same vehicle.
IOU_THRESHOLD = 0.30

MIN_READS_TO_COUNT = 1


# ============================================================
# GROUND TRUTH
#
# IMPORTANT:
# These frame numbers are from the 20 FPS GT timeline.
#
# pipeline_frame = GT frame * 60 / 20
#
# bbox is OPTIONAL.
#
# If bbox is supplied, we can evaluate:
#   1. whether the target vehicle was detected
#   2. whether the correct track was associated
#   3. OCR accuracy
#
# If bbox is missing, we cannot measure detection recall for that
# GT entry, so it is excluded from the detection metric.
# ============================================================

GROUND_TRUTH = {

    150: {
    "plate": "KA41AA0033",
    "bbox": [433, 2034, 665, 2142],
},

    604: {
        "plate": None,
        "bbox": None,
    },

    616: {
        "plate": "KA05NC8111",
        "bbox": None,
    },

    588: {
        "plate": "KA07N5205",
        "bbox": None,
    },

    532: {
        "plate": "AP28CC4284",
        "bbox": None,
    },

    511: {
        "plate": "AP28CC4284",
        "bbox": None,
    },

    433: {
        "plate": "AP28CC4284",
        "bbox": None,
    },

    448: {
        "plate": "KA51HB7942",
        "bbox": None,
    },

    450: {
        "plate": None,
        "bbox": None,
    },

    323: {
        "plate": "KA01AE7247",
        "bbox": None,
    },

    440: {
        "plate": "KA01AE7247",
        "bbox": None,
    },

    355: {
        "plate": "KA05MU0712",
        "bbox": [2340, 1793, 2472, 1861],
    },

    422: {
        "plate": None,
        "bbox": None,
    },

    424: {
        "plate": "AP28CC4284",
        "bbox": [2541, 1615, 2663, 1682],
    },

    389: {
        "plate": "KA05MU0712",
        "bbox": None,
    },

    390: {
        "plate": "AP28CC4284",
        "bbox": None,
    },

    387: {
        "plate": None,
        "bbox": None,
    },

    301: {
        "plate": "KA05MU0712",
        "bbox": None,
    },

    261: {
        "plate": None,
        "bbox": None,
    },

    254: {
        "plate": "IO8205636",
        "bbox": None,
    },

    246: {
        "plate": None,
        "bbox": None,
    },

    56: {
        "plate": "KA01MS7103",
        "bbox": None,
    },

    119: {
        "plate": "KA01MS7103",
        "bbox": None,
    },

    2: {
        "plate": "KA21AA0033",
        "bbox": [1215, 1997, 1482, 2108],
    },

    36: {
        "plate": "KY14N0033",
        "bbox": None,
    },

    19: {
        "plate": "KA21AA0033",
        "bbox": [1401, 1887, 1613, 1983],
    },
}


# ============================================================
# HELPERS
# ============================================================

def clean(text):
    if text is None:
        return None

    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def char_level_accuracy(predicted, truth):
    if not predicted or not truth:
        return 0.0

    max_len = max(len(predicted), len(truth))

    matches = sum(
        1
        for a, b in zip(predicted, truth)
        if a == b
    )

    return matches / max_len


def bbox_iou(box_a, box_b):
    """
    Calculate IoU between:
        [x1, y1, x2, y2]
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    intersection = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def pipeline_frame_from_gt(gt_frame):
    return round(
        gt_frame * PIPELINE_FPS / GT_FPS
    )


# ============================================================
# FIND DETECTIONS AT A SPECIFIC PIPELINE FRAME
# ============================================================

def get_observations_at_frame(events, pipeline_frame):
    """
    Find every tracklet that has an observation at the exact
    pipeline frame.

    We use the actual observation bbox rather than merely checking
    frame_first_seen/frame_last_seen.

    This is critical because a tracklet can span a frame while the
    vehicle was actually absent from that exact frame.
    """

    matches = []

    for event in events:

        for observation in event.get("observations", []):

            if observation.get("frame") == pipeline_frame:

                matches.append({
                    "event": event,
                    "bbox": observation.get("bbox"),
                })

                break

    return matches


# ============================================================
# MATCH GT VEHICLE TO CURRENT PIPELINE TRACK
# ============================================================

def find_matching_track(events, gt_frame, gt_bbox):

    pipeline_frame = pipeline_frame_from_gt(gt_frame)

    observations = get_observations_at_frame(
        events,
        pipeline_frame
    )

    if not observations:
        return {
            "status": "not_detected",
            "pipeline_frame": pipeline_frame,
            "event": None,
            "iou": 0.0,
        }

    # --------------------------------------------------------
    # If we have a manually annotated GT bbox:
    # use IoU to identify the actual vehicle.
    # --------------------------------------------------------

    if gt_bbox is not None:

        candidates = []

        for item in observations:

            if item["bbox"] is None:
                continue

            iou = bbox_iou(
                gt_bbox,
                item["bbox"]
            )

            candidates.append(
                (iou, item["event"], item["bbox"])
            )

        if not candidates:

            return {
                "status": "not_detected",
                "pipeline_frame": pipeline_frame,
                "event": None,
                "iou": 0.0,
            }

        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        best_iou, best_event, best_bbox = candidates[0]

        if best_iou < IOU_THRESHOLD:

            return {
                "status": "not_detected",
                "pipeline_frame": pipeline_frame,
                "event": None,
                "iou": best_iou,
            }

        return {
            "status": "matched",
            "pipeline_frame": pipeline_frame,
            "event": best_event,
            "iou": best_iou,
        }

    # --------------------------------------------------------
    # NO GT BBOX
    #
    # We cannot know which vehicle is the target.
    # Therefore do NOT arbitrarily select one.
    # --------------------------------------------------------

    return {
        "status": "needs_bbox",
        "pipeline_frame": pipeline_frame,
        "event": None,
        "iou": 0.0,
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate(events_path="observation_events.json"):

    with open(events_path, "r") as f:
        events = json.load(f)

    detection_results = []
    ocr_results = []

    not_detected = []
    needs_bbox = []

    for gt_frame, gt_data in GROUND_TRUTH.items():

        truth = clean(gt_data["plate"])
        gt_bbox = gt_data["bbox"]

        pipeline_frame = pipeline_frame_from_gt(
            gt_frame
        )

        match = find_matching_track(
            events,
            gt_frame,
            gt_bbox
        )

        # ====================================================
        # NO GT BBOX
        # ====================================================

        if match["status"] == "needs_bbox":

            needs_bbox.append({
                "gt_frame": gt_frame,
                "pipeline_frame": pipeline_frame,
                "truth": truth,
            })

            continue

        # ====================================================
        # TARGET VEHICLE NOT DETECTED
        # ====================================================

        if match["status"] == "not_detected":

            detection_results.append({
                "gt_frame": gt_frame,
                "pipeline_frame": pipeline_frame,
                "detected": False,
                "iou": match["iou"],
                "truth": truth,
            })

            not_detected.append({
                "gt_frame": gt_frame,
                "pipeline_frame": pipeline_frame,
                "truth": truth,
            })

            continue

        # ====================================================
        # TARGET VEHICLE DETECTED
        # ====================================================

        event = match["event"]

        detection_results.append({
            "gt_frame": gt_frame,
            "pipeline_frame": pipeline_frame,
            "detected": True,
            "iou": match["iou"],
            "truth": truth,
            "tracklet_id": event["tracklet_id"],
        })

        # If GT truth is intentionally None, don't evaluate OCR.
        if truth is None:
            continue

        predicted = clean(
            event["plate"]["text"]
        )

        num_reads = event["plate"].get(
            "num_reads",
            0
        )

        if num_reads < MIN_READS_TO_COUNT:

            ocr_results.append({
                "gt_frame": gt_frame,
                "pipeline_frame": pipeline_frame,
                "tracklet_id": event["tracklet_id"],
                "truth": truth,
                "predicted": predicted,
                "exact_match": False,
                "char_accuracy": 0.0,
                "num_reads": num_reads,
                "ocr_status": "insufficient_reads",
            })

            continue

        exact_match = (
            predicted == truth
        )

        char_acc = char_level_accuracy(
            predicted,
            truth
        )

        ocr_results.append({
            "gt_frame": gt_frame,
            "pipeline_frame": pipeline_frame,
            "tracklet_id": event["tracklet_id"],
            "truth": truth,
            "predicted": predicted,
            "exact_match": exact_match,
            "char_accuracy": char_acc,
            "num_reads": num_reads,
            "ocr_status": "evaluated",
        })

    # ========================================================
    # PRINT DETECTION RESULTS
    # ========================================================

    print()
    print("=" * 100)
    print("DETECTION / ASSOCIATION RESULTS")
    print("=" * 100)

    for r in detection_results:

        status = (
            "DETECTED"
            if r["detected"]
            else "MISSED"
        )

        print(
            f"GT~{r['gt_frame']:4d} "
            f"→ pipeline~{r['pipeline_frame']:4d} "
            f"{status:10s} "
            f"IoU={r['iou']:.2f} "
            f"truth={r['truth'] or ''}"
        )

    # ========================================================
    # PRINT OCR RESULTS
    # ========================================================

    print()
    print("=" * 100)
    print("OCR RESULTS")
    print("=" * 100)

    for r in ocr_results:

        if r["exact_match"]:
            status = "CORRECT"
        else:
            status = "WRONG"

        print(
            f"GT~{r['gt_frame']:4d} "
            f"→ pipeline~{r['pipeline_frame']:4d} "
            f"{r['tracklet_id']:28s} "
            f"predicted={r['predicted'] or 'None':15s} "
            f"truth={r['truth']:15s} "
            f"[{status}] "
            f"char_acc={r['char_accuracy']:.0%} "
            f"reads={r['num_reads']}"
        )

    # ========================================================
    # GT ENTRIES THAT NEED MANUAL BBOXES
    # ========================================================

    if needs_bbox:

        print()
        print("=" * 100)
        print("GT ENTRIES NEEDING MANUAL BBOXES")
        print("=" * 100)

        for r in needs_bbox:

            print(
                f"GT~{r['gt_frame']:4d} "
                f"→ pipeline~{r['pipeline_frame']:4d} "
                f"truth={r['truth'] or 'None'}"
            )

        print()
        print(
            "These entries are NOT counted as detector failures."
        )
        print(
            "Add their manually annotated vehicle bbox to "
            "GROUND_TRUTH before using them for detection recall."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # --------------------------------------------------------
    # Detection recall
    # --------------------------------------------------------

    if detection_results:

        detected = sum(
            1
            for r in detection_results
            if r["detected"]
        )

        total_detection_gt = len(
            detection_results
        )

        detection_recall = (
            detected / total_detection_gt
        )

        print(
            f"Detection recall:          "
            f"{detection_recall:.1%} "
            f"({detected}/{total_detection_gt})"
        )

    else:

        print(
            "Detection recall:          N/A "
            "(no GT bounding boxes)"
        )

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    if ocr_results:

        exact_matches = sum(
            1
            for r in ocr_results
            if r["exact_match"]
        )

        exact_accuracy = (
            exact_matches / len(ocr_results)
        )

        avg_char_accuracy = (
            sum(
                r["char_accuracy"]
                for r in ocr_results
            )
            / len(ocr_results)
        )

        print(
            f"OCR exact accuracy:        "
            f"{exact_accuracy:.1%} "
            f"({exact_matches}/{len(ocr_results)})"
        )

        print(
            f"OCR character accuracy:    "
            f"{avg_char_accuracy:.1%}"
        )

    else:

        print(
            "OCR accuracy:              N/A"
        )

    # --------------------------------------------------------
    # Manual work remaining
    # --------------------------------------------------------

    print(
        f"GT entries needing bbox:    "
        f"{len(needs_bbox)}"
    )

    print(
        f"GT entries with bbox:       "
        f"{len(detection_results)}"
    )

    print("=" * 100)


if __name__ == "__main__":

    import sys

    events_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "observation_events.json"
    )

    evaluate(events_file)
