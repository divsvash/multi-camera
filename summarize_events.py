# ============================================================
# Quick human-readable summary of observation_events.json -
# hides embeddings, shows what actually matters for review.
# ============================================================

import json

with open("observation_events.json", "r") as f:
    events = json.load(f)

print(f"{len(events)} total events\n")
print(f"{'Tracklet':30s} {'Plate':15s} {'Valid':6s} {'Reads':6s} {'Type':8s} {'Colour':8s} {'DetConf':8s}")
print("-" * 90)

for e in events:
    print(f"{e['tracklet_id']:30s} "
          f"{str(e['plate']['text']):15s} "
          f"{str(e['plate']['valid']):6s} "
          f"{e['plate']['num_reads']:<6d} "
          f"{e['vehicle']['type']:8s} "
          f"{e['vehicle']['colour']:8s} "
          f"{e['detection_confidence_avg']:<8.2f}")