# ============================================================
# Post-processing merge: combines tracks that are almost
# certainly the same physical vehicle but got split into
# different Track IDs by ByteTrack (e.g. due to occlusion).
#
# Heuristic: two tracks are merged if they are close in time
# (one ends shortly before the other starts) AND their voted
# plate text is similar (small edit distance).
# ============================================================

import json
import re
from collections import Counter

def edit_distance(a, b):
    """Simple Levenshtein distance."""
    if a is None or b is None:
        return 999
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]

def get_best_plate_from_group(group):
    """
    Select the best plate from a group of merged events.
    Uses a combination of:
    1. Most common plate text (frequency voting)
    2. Highest confidence
    3. Best format validity
    """
    if not group:
        return None
    
    # Get all plate texts, weighted by how much evidence backed each one
    plate_candidates = []
    for event in group:
        text = event["plate"]["text"]
        if text:
            # NOTE: full_pipeline.py saves raw_ocr_reads as plain text strings
            # only (confidence is used during voting but not written to the
            # JSON output), so there's no per-read confidence available here.
            # Rank on what's actually in the data instead: how many reads
            # backed this result, and whether it passed format validation.
            plate_candidates.append({
                "text": text,
                "num_reads": event["plate"]["num_reads"],
                "valid": event["plate"]["valid"],
                "format": event["plate"]["format"]
            })
    
    if not plate_candidates:
        return None
    
    # Count frequency of each plate text
    text_counts = Counter([p["text"] for p in plate_candidates])
    most_common_text = text_counts.most_common(1)[0][0]
    most_common_count = text_counts.most_common(1)[0][1]
    
    # Filter candidates that match the most common text
    matching_candidates = [p for p in plate_candidates if p["text"] == most_common_text]
    
    # Among matching candidates, prefer a validated format, then more reads
    best_candidate = max(matching_candidates, key=lambda x: (x["valid"], x["num_reads"]))
    
    return best_candidate

def merge_events(events, max_frame_gap=80, max_edit_distance=3):
    """
    Merge events likely to be the same vehicle.
    Returns a new list with merged events combined into one.
    """
    events_sorted = sorted(events, key=lambda e: e["frame_first_seen"])
    merged = []
    used = set()

    for i, event_a in enumerate(events_sorted):
        if i in used:
            continue

        group = [event_a]
        used.add(i)

        for j in range(i + 1, len(events_sorted)):
            if j in used:
                continue
            event_b = events_sorted[j]

            # Check if events overlap or are close in time
            frame_gap = event_b["frame_first_seen"] - group[-1]["frame_last_seen"]
            if frame_gap < 0 or frame_gap > max_frame_gap:
                continue

            text_a = group[-1]["plate"]["text"]
            text_b = event_b["plate"]["text"]
            if text_a is None or text_b is None:
                continue

            # Check if plates are similar (allowing for OCR errors)
            dist = edit_distance(text_a, text_b)
            if dist <= max_edit_distance:
                group.append(event_b)
                used.add(j)

        if len(group) == 1:
            merged.append(group[0])
        else:
            # Combine the group into one event
            all_reads = []
            for e in group:
                all_reads.extend(e["plate"]["raw_ocr_reads"])
            
            # Get the best plate from the group
            best_plate = get_best_plate_from_group(group)
            
            if not best_plate:
                # Fallback: use the event with most reads
                best_event = max(group, key=lambda e: e["plate"]["num_reads"])
                combined = {
                    "event_id": group[0]["event_id"] + "_merged",
                    "camera_id": group[0]["camera_id"],
                    "camera_location": group[0]["camera_location"],
                    "tracklet_id": "+".join(e["tracklet_id"] for e in group),
                    "frame_first_seen": group[0]["frame_first_seen"],
                    "frame_last_seen": group[-1]["frame_last_seen"],
                    "plate": {
                        "text": best_event["plate"]["text"],
                        "format": best_event["plate"]["format"],
                        "valid": best_event["plate"]["valid"],
                        "num_reads": len(all_reads),
                        "raw_ocr_reads": all_reads,
                        "merged_from": [e["tracklet_id"] for e in group],
                    },
                    "detection_confidence_avg": sum(e["detection_confidence_avg"] for e in group) / len(group),
                }
            else:
                combined = {
                    "event_id": group[0]["event_id"] + "_merged",
                    "camera_id": group[0]["camera_id"],
                    "camera_location": group[0]["camera_location"],
                    "tracklet_id": "+".join(e["tracklet_id"] for e in group),
                    "frame_first_seen": group[0]["frame_first_seen"],
                    "frame_last_seen": group[-1]["frame_last_seen"],
                    "plate": {
                        "text": best_plate["text"],
                        "format": best_plate["format"],
                        "valid": best_plate["valid"],
                        "num_reads": len(all_reads),
                        "raw_ocr_reads": all_reads,
                        "merged_from": [e["tracklet_id"] for e in group],
                        "confidence": best_plate["confidence"],
                    },
                    "detection_confidence_avg": sum(e["detection_confidence_avg"] for e in group) / len(group),
                }
            
            merged.append(combined)
            print(f"MERGED: {[e['tracklet_id'] for e in group]} -> one vehicle "
                  f"(texts: {[e['plate']['text'] for e in group]})")
            print(f"  -> CHOSE: {combined['plate']['text']} (confidence: {combined['plate'].get('confidence', 0.5):.3f})")

    return merged

if __name__ == "__main__":
    with open("observation_events.json", "r") as f:
        events = json.load(f)

    print(f"Before merge: {len(events)} events\n")
    merged = merge_events(events)
    print(f"\nAfter merge: {len(merged)} events")

    with open("observation_events_merged.json", "w") as f:
        json.dump(merged, f, indent=2)
    print("Saved to observation_events_merged.json")