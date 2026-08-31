# ============================================================
# Diagnostic Tool - Analyze OCR performance in detail
# ============================================================

import json
import re
from collections import defaultdict, Counter
from match_ground_truth import match_ground_truth

def clean_plate_text(text):
    if not text:
        return ""
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def character_accuracy(pred, truth):
    if not pred or not truth:
        return 0.0
    pred = clean_plate_text(pred)
    truth = clean_plate_text(truth)
    if not pred or not truth:
        return 0.0
    max_len = max(len(pred), len(truth))
    pred = pred.ljust(max_len)
    truth = truth.ljust(max_len)
    correct = sum(1 for p, t in zip(pred, truth) if p == t)
    return correct / max_len

def exact_match(pred, truth):
    if not pred or not truth:
        return False
    return clean_plate_text(pred) == clean_plate_text(truth)

def analyze_diagnostics():
    # Load events
    try:
        with open("observation_events_merged.json", 'r') as f:
            events = json.load(f)
    except:
        with open("observation_events.json", 'r') as f:
            events = json.load(f)
    
    # Ground truth from eval_harness
    from eval_harness import GROUND_TRUTH
    
    print("=" * 70)
    print("LAYER 1: RAW OCR MODEL QUALITY (before voting, single reads)")
    print("=" * 70)
    
    total_reads = 0
    total_char_accuracy = 0
    total_exact_accuracy = 0
    
    for event in events:
        track_id = event.get("tracklet_id", "")
        raw_reads = event.get("plate", {}).get("raw_ocr_reads", [])
        predicted = event.get("plate", {}).get("text")
        
        # Get ground truth
        _, truth = match_ground_truth(track_id, GROUND_TRUTH)
        
        if not truth or not raw_reads:
            continue
        
        # Analyze each raw read
        for read in raw_reads:
            if isinstance(read, (list, tuple)):
                read_text = read[0] if len(read) > 0 else ""
            else:
                read_text = read
            
            char_acc = character_accuracy(read_text, truth)
            exact = exact_match(read_text, truth)
            
            total_reads += 1
            total_char_accuracy += char_acc
            if exact:
                total_exact_accuracy += 1
    
    avg_char_acc = (total_char_accuracy / total_reads * 100) if total_reads > 0 else 0
    exact_match_rate = (total_exact_accuracy / total_reads * 100) if total_reads > 0 else 0
    
    print(f"Individual reads scored:      {total_reads}")
    print(f"Avg per-read char accuracy:    {avg_char_acc:.1f}%")
    print(f"Avg per-read exact match:      {exact_match_rate:.1f}%")
    
    # Estimate theoretical ceiling
    estimated_plate_length = 10
    ceiling = (avg_char_acc / 100) ** estimated_plate_length * 100
    print(f"Theoretical exact-match ceiling: ~{ceiling:.1f}%")
    
    print("\n" + "=" * 70)
    print("LAYER 2: FINAL VOTED RESULT")
    print("=" * 70)
    
    # Evaluate voted results
    total_tracks = 0
    total_voted_char_acc = 0
    total_voted_exact = 0
    
    for event in events:
        track_id = event.get("tracklet_id", "")
        predicted = event.get("plate", {}).get("text")
        num_reads = event.get("plate", {}).get("num_reads", 0)
        
        _, truth = match_ground_truth(track_id, GROUND_TRUTH)
        
        if not truth or num_reads == 0:
            continue
        
        char_acc = character_accuracy(predicted, truth) * 100
        exact = exact_match(predicted, truth)
        
        total_tracks += 1
        total_voted_char_acc += char_acc
        if exact:
            total_voted_exact += 1
    
    avg_voted_char_acc = (total_voted_char_acc / total_tracks) if total_tracks > 0 else 0
    voted_exact_rate = (total_voted_exact / total_tracks * 100) if total_tracks > 0 else 0
    
    print(f"Tracks scored:                 {total_tracks}")
    print(f"Avg final char accuracy:        {avg_voted_char_acc:.1f}%")
    print(f"Final exact-match accuracy:     {voted_exact_rate:.1f}%")
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    improvement = avg_voted_char_acc - avg_char_acc
    print(f"Voting/correction improved char accuracy by +{improvement:.1f}%")
    
    if avg_char_acc < 70:
        print("\n>>> Your RAW per-character OCR accuracy is below ~70%.")
        print(">>> Fix: better/fine-tuned recognizer, or better plate crops BEFORE OCR.")
    else:
        print("\n>>> Your OCR is working well! Focus on voting and correction.")
    
    print("\n" + "=" * 70)
    print("READS PER TRACK vs FINAL ACCURACY")
    print("=" * 70)
    
    # Group by read count
    groups = {
        "1-2 reads": [],
        "3-5 reads": [],
        "6-10 reads": [],
        "11+ reads": []
    }
    
    for event in events:
        track_id = event.get("tracklet_id", "")
        predicted = event.get("plate", {}).get("text")
        num_reads = event.get("plate", {}).get("num_reads", 0)
        
        _, truth = match_ground_truth(track_id, GROUND_TRUTH)
        
        if not truth or num_reads == 0:
            continue
        
        char_acc = character_accuracy(predicted, truth) * 100
        
        if num_reads <= 2:
            groups["1-2 reads"].append(char_acc)
        elif num_reads <= 5:
            groups["3-5 reads"].append(char_acc)
        elif num_reads <= 10:
            groups["6-10 reads"].append(char_acc)
        else:
            groups["11+ reads"].append(char_acc)
    
    for group_name, accuracies in groups.items():
        avg = sum(accuracies) / len(accuracies) if accuracies else 0
        print(f"{group_name:12} : {len(accuracies):3} tracks, avg char acc = {avg:.1f}%")

if __name__ == "__main__":
    analyze_diagnostics()