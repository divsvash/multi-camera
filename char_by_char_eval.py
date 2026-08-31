# char_by_char_eval.py
# ============================================================
# Character-by-character evaluation - much more fair!
# ============================================================

import json
import re
from collections import defaultdict, Counter
from match_ground_truth import match_ground_truth

# CORRECTED GROUND TRUTH
GROUND_TRUTH = {
    "CAM_TEST_01_trk_2": "KA21AA0033",
    "CAM_TEST_01_trk_19": "KA21AA0033",
    "CAM_TEST_01_trk_323": "KA01AE7247",
    "CAM_TEST_01_trk_440": "KA01AE7247",
    "CAM_TEST_01_trk_420": "KA01AE7247",
    "CAM_TEST_01_trk_387": "KA01AE7247",
    "CAM_TEST_01_trk_301": "KA05MU0712",
    "CAM_TEST_01_trk_355": "KA05MU0712",
    "CAM_TEST_01_trk_389": "KA05MU0712",
    "CAM_TEST_01_trk_390": "AP28CC4284",
    "CAM_TEST_01_trk_424": "AP28CC4284",
    "CAM_TEST_01_trk_433": "AP28CC4284",
    "CAM_TEST_01_trk_511": "AP28CC4284",
    "CAM_TEST_01_trk_532": "AP28CC4284",
    "CAM_TEST_01_trk_448": "KA51HB7942",
    "CAM_TEST_01_trk_588": "KA07N5205",
    "CAM_TEST_01_trk_616": "KA05NC8111",
    "CAM_TEST_01_trk_56": "KA01MS7103",
    "CAM_TEST_01_trk_119": "KA01MS7103",
    "CAM_TEST_01_trk_261": None,
    "CAM_TEST_01_trk_422": None,
    "CAM_TEST_01_trk_450": None,
    "CAM_TEST_01_trk_604": None,
}

def clean_text(text):
    if not text:
        return ""
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def char_accuracy(pred, truth):
    """Calculate character accuracy with position alignment"""
    if not pred or not truth:
        return 0.0, []
    
    pred = clean_text(pred)
    truth = clean_text(truth)
    
    if not pred or not truth:
        return 0.0, []
    
    # Pad to same length
    max_len = max(len(pred), len(truth))
    pred = pred.ljust(max_len)
    truth = truth.ljust(max_len)
    
    correct = 0
    details = []
    for i, (p, t) in enumerate(zip(pred, truth)):
        is_correct = (p == t)
        if is_correct:
            correct += 1
        details.append({
            "position": i,
            "pred": p,
            "truth": t,
            "correct": is_correct
        })
    
    return correct / max_len, details

def analyze_char_confusions(events_data):
    """Find which characters are most commonly confused"""
    confusions = defaultdict(lambda: defaultdict(int))
    position_confusions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for event in events_data:
        track_id = event.get("tracklet_id", "")
        predicted = event.get("plate", {}).get("text")
        num_reads = event.get("plate", {}).get("num_reads", 0)
        
        _, truth = match_ground_truth(track_id, GROUND_TRUTH)
        
        if not truth or num_reads == 0:
            continue
        
        _, details = char_accuracy(predicted, truth)
        for detail in details:
            if not detail["correct"]:
                p = detail["pred"]
                t = detail["truth"]
                pos = detail["position"]
                confusions[t][p] += 1
                position_confusions[pos][t][p] += 1
    
    return confusions, position_confusions

def main():
    # Load events
    try:
        with open("observation_events_merged.json", 'r') as f:
            events = json.load(f)
    except:
        with open("observation_events.json", 'r') as f:
            events = json.load(f)
    
    print("=" * 70)
    print("CHARACTER-BY-CHARACTER EVALUATION")
    print("=" * 70)
    
    results = []
    total_char_accuracy = 0
    
    for event in events:
        track_id = event.get("tracklet_id", "")
        predicted = event.get("plate", {}).get("text")
        num_reads = event.get("plate", {}).get("num_reads", 0)
        raw_reads = event.get("plate", {}).get("raw_ocr_reads", [])
        
        _, truth = match_ground_truth(track_id, GROUND_TRUTH)
        
        if not truth or num_reads == 0:
            continue
        
        char_acc, details = char_accuracy(predicted, truth)
        char_acc_pct = char_acc * 100
        
        # Check if any individual raw read got it right
        raw_correct = False
        for read in raw_reads:
            if isinstance(read, (list, tuple)):
                read_text = read[0] if len(read) > 0 else ""
            else:
                read_text = read
            if clean_text(read_text) == clean_text(truth):
                raw_correct = True
                break
        
        results.append({
            "track_id": track_id,
            "predicted": predicted,
            "truth": truth,
            "char_acc": char_acc_pct,
            "details": details,
            "num_reads": num_reads,
            "raw_had_correct": raw_correct
        })
        total_char_accuracy += char_acc_pct
    
    # Print per-track results
    print("\nPER-TRACK RESULTS (character-level)")
    print("-" * 70)
    
    for r in results:
        # Color code based on accuracy
        if r["char_acc"] >= 90:
            status = "✅ EXCELLENT"
        elif r["char_acc"] >= 70:
            status = "👍 GOOD"
        elif r["char_acc"] >= 50:
            status = "⚠️ OK"
        else:
            status = "❌ POOR"
        
        print(f"{r['track_id']:30} {r['predicted']:15} vs {r['truth']:15} -> {r['char_acc']:.1f}% {status}")
        
        # Show which characters are wrong
        wrong_chars = [d for d in r["details"] if not d["correct"]]
        if wrong_chars:
            pos_str = ", ".join([f"pos{d['position']}: '{d['pred']}'→'{d['truth']}'" for d in wrong_chars])
            print(f"  ❌ Errors: {pos_str}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(results)
    avg_char_acc = total_char_accuracy / total if total > 0 else 0
    
    # Count how many have >90% character accuracy
    near_perfect = sum(1 for r in results if r["char_acc"] >= 90)
    good = sum(1 for r in results if 70 <= r["char_acc"] < 90)
    ok = sum(1 for r in results if 50 <= r["char_acc"] < 70)
    poor = sum(1 for r in results if r["char_acc"] < 50)
    
    print(f"Tracks evaluated:  {total}")
    print(f"Average character accuracy: {avg_char_acc:.1f}%")
    print(f"\nCharacter Accuracy Breakdown:")
    print(f"  ✅ 90%+:   {near_perfect} tracks ({near_perfect/total*100:.0f}%) - Excellent")
    print(f"  👍 70-89%: {good} tracks ({good/total*100:.0f}%) - Good")
    print(f"  ⚠️ 50-69%: {ok} tracks ({ok/total*100:.0f}%) - OK")
    print(f"  ❌ <50%:   {poor} tracks ({poor/total*100:.0f}%) - Needs Work")
    
    # Find character confusions
    print("\n" + "=" * 70)
    print("MOST COMMON CHARACTER CONFUSIONS")
    print("=" * 70)
    
    confusions, pos_confusions = analyze_char_confusions(events)
    
    for truth_char, pred_chars in sorted(confusions.items(), key=lambda x: sum(x[1].values()), reverse=True)[:10]:
        total_confusions = sum(pred_chars.values())
        print(f"\n'{truth_char}' was misread as:")
        for pred_char, count in sorted(pred_chars.items(), key=lambda x: -x[1])[:5]:
            print(f"  '{pred_char}': {count} times ({count/total_confusions*100:.0f}%)")
    
    # Position-specific analysis
    print("\n" + "=" * 70)
    print("POSITION-SPECIFIC CONFUSIONS")
    print("=" * 70)
    
    for pos in sorted(pos_confusions.keys()):
        chars_at_pos = pos_confusions[pos]
        total_at_pos = sum(sum(pred_counts.values()) for pred_counts in chars_at_pos.values())
        if total_at_pos > 0:
            print(f"\nPosition {pos}: {total_at_pos} errors")
            for truth_char, pred_chars in chars_at_pos.items():
                for pred_char, count in pred_chars.items():
                    print(f"  '{truth_char}' → '{pred_char}': {count} times")
    
    # Check if raw reads ever got it right
    raw_correct_count = sum(1 for r in results if r["raw_had_correct"])
    print("\n" + "=" * 70)
    print("RAW OCR ANALYSIS")
    print("=" * 70)
    print(f"Raw reads that were EXACTLY correct: {raw_correct_count}/{total} ({raw_correct_count/total*100:.1f}%)")
    print("This means your voting is actually HURTING accuracy if it's lower than raw reads!")

if __name__ == "__main__":
    main()