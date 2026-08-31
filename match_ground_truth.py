# ============================================================
# Shared ground-truth lookup, used by every eval script.
#
# Exists because this logic was previously copy-pasted into
# eval_harness.py, eval_diagnostics.py, and char_by_char_eval.py
# independently, using `if key in track_id` (substring containment).
# That's dangerous with numeric track IDs: "CAM_TEST_01_trk_2" is a
# substring of "CAM_TEST_01_trk_261", so a track explicitly marked
# None (no readable plate) was silently scored against the wrong
# vehicle's plate. Verified this actually happens, not hypothetical.
#
# This version splits merged tracklet IDs ("trkA+trkB+trkC") on '+'
# and requires an EXACT match on each component - handles the real
# case (merged tracks) that motivated substring matching in the
# first place, without the collision risk.
# ============================================================


def match_ground_truth(track_id, ground_truth):
    """Returns (matched_key, truth_value) or (None, None) if no exact
    component of track_id is a key in ground_truth."""
    for component in track_id.split('+'):
        if component in ground_truth:
            return component, ground_truth[component]
    return None, None