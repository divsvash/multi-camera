# ============================================================
# Finds plates that were read independently by 2+ different
# tracklets - these are your highest-confidence candidates for
# ground truth labeling, since independent agreement is strong
# evidence the reading is actually correct.
# ============================================================

import json
from collections import defaultdict


def edit_distance(a, b):
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


with open("observation_events.json", "r") as f:
    events = json.load(f)

# only consider events with a real plate reading and at least some evidence
valid_events = [e for e in events if e["plate"]["text"] and e["plate"]["num_reads"] >= 1]

# group by near-identical plate text (edit distance <= 2 = likely same plate)
groups = []
used = set()

for i, e1 in enumerate(valid_events):
    if i in used:
        continue
    group = [e1]
    used.add(i)
    for j, e2 in enumerate(valid_events):
        if j in used or j == i:
            continue
        if edit_distance(e1["plate"]["text"], e2["plate"]["text"]) <= 2:
            group.append(e2)
            used.add(j)
    if len(group) >= 2:
        groups.append(group)

# sort by group size (most convergent first) and total reads (most evidence first)
groups.sort(key=lambda g: (len(g), sum(e["plate"]["num_reads"] for e in g)), reverse=True)

print(f"Found {len(groups)} plates with independent convergence (2+ separate tracklets)\n")
print("=" * 90)

for group in groups:
    plates_seen = [e["plate"]["text"] for e in group]
    total_reads = sum(e["plate"]["num_reads"] for e in group)
    tracklets = [e["tracklet_id"] for e in group]

    print(f"\n{len(group)} tracklets, {total_reads} total reads:")
    for e in group:
        print(f"  {e['tracklet_id']:30s} -> {e['plate']['text']:15s} "
              f"({e['plate']['num_reads']} reads, frames {e['frame_first_seen']}-{e['frame_last_seen']})")

print("\n" + "=" * 90)
print("\nPRIORITY GROUND-TRUTH CHECKLIST (verify these against the video first):")
print("-" * 90)
for group in groups:
    best = max(group, key=lambda e: e["plate"]["num_reads"])
    mid_frame = (best["frame_first_seen"] + best["frame_last_seen"]) // 2
    print(f"  Plate guess: {best['plate']['text']:15s} | check around frame {mid_frame:5d} | "
          f"tracklets: {', '.join(e['tracklet_id'].split('_trk_')[1] for e in group)}")