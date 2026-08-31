# ============================================================
# Shows exactly what happened to each track, character by
# character: every raw OCR read, what correct_plate() does to
# EACH read individually, and what the final voted/corrected
# result was - all against ground truth.
#
# Run this after full_pipeline.py + merge_tracks.py, same as
# eval_diagnostics.py.
# ============================================================

import json
import re
from collections import defaultdict

from eval_harness import GROUND_TRUTH, clean
from plate_corrector import correct_plate

# Copied from full_pipeline.py rather than imported - full_pipeline.py's main
# loop runs at module level (not guarded by if __name__ == "__main__"), so
# importing from it would re-run your entire video pipeline just to grab
# this one function. Keep this in sync with full_pipeline.py by hand.
def vote_by_position(raw_reads_with_conf):
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
                char_weights[text[i]] += conf
        if char_weights:
            voted += max(char_weights, key=char_weights.get)
    return voted


def inspect(events_path="observation_events_merged.json"):
    with open(events_path, "r") as f:
        events = json.load(f)

    for event in events:
        tracklet_id = event["tracklet_id"]
        truth = GROUND_TRUTH.get(tracklet_id)
        if truth is None:
            continue
        truth_clean = clean(truth)

        raw_reads = event["plate"].get("raw_ocr_reads", [])
        print("=" * 78)
        print(f"TRACK: {tracklet_id}   GROUND TRUTH: {truth_clean}   "
              f"(final predicted: {clean(event['plate']['text'])})")
        print("=" * 78)

        if not raw_reads:
            print("  (no raw OCR reads recorded for this track)")
            continue

        print(f"  {'#':<3} {'raw OCR read':<20} {'cleaned':<14} "
              f"{'per-read corrected':<20} {'valid?':<7}")
        print("  " + "-" * 74)
        for i, raw in enumerate(raw_reads):
            cleaned = clean(raw)
            result = correct_plate(cleaned) if cleaned else {"corrected": None, "valid": False}
            match_marker = " <-- exact" if result["corrected"] == truth_clean else ""
            print(f"  {i:<3} {raw!r:<20} {cleaned or '':<14} "
                  f"{str(result['corrected']):<20} {str(result['valid']):<7}{match_marker}")

        # Reconstruct exactly what the pipeline's own vote_by_position saw -
        # note: this needs (text, confidence) tuples, which raw_ocr_reads
        # alone doesn't give us (confidence isn't saved per-read in the
        # output JSON currently - only the aggregate). Using conf=1.0 as a
        # stand-in just to show what position-voting on the RAW strings looks
        # like; real confidences would change weighting but not the core
        # alignment issue if one exists.
        fake_conf_reads = [(r, 1.0) for r in raw_reads]
        voted_on_raw = vote_by_position(fake_conf_reads)
        print(f"\n  vote_by_position() on RAW (uncorrected) reads -> {voted_on_raw!r}")
        print(f"  correct_plate() applied to that vote            -> "
              f"{correct_plate(clean(voted_on_raw))['corrected'] if voted_on_raw else None!r}")
        print(f"  ACTUAL final result stored in event              -> {clean(event['plate']['text'])!r}")
        print()


if __name__ == "__main__":
    inspect()