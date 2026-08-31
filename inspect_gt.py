import json
import cv2
import os

GT_FPS = 20.0
PIPELINE_FPS = 60.0

VIDEO_PATH = "test_frames/demo_video1.mp4"

GROUND_TRUTH = {
    150: "KA41AA0033",
    604: None,
    616: "KA05NC8111",
    588: "KA07N5205",
    532: "AP28CC4284",
    511: "AP28CC4284",
    433: "AP28CC4284",
    448: "KA51HB7942",
    450: None,
    323: "KA01AE7247",
    440: "KA01AE7247",
    355: "KA05MU0712",
    422: None,
    424: "AP28CC4284",
    389: "KA05MU0712",
    390: "AP28CC4284",
    387: None,
    301: "KA05MU0712",
    261: None,
    254: "IO8205636",
    246: None,
    56: "KA01MS7103",
    119: "KA01MS7103",
    2: "KA21AA0033",
    36: "KY14N0033",
    19: "KA21AA0033",
}


def get_pipeline_frame(gt_frame):
    return round(
        gt_frame * PIPELINE_FPS / GT_FPS
    )


with open("observation_events.json", "r") as f:
    events = json.load(f)


os.makedirs("gt_inspection", exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


for gt_frame, truth in GROUND_TRUTH.items():

    pipeline_frame = get_pipeline_frame(gt_frame)

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        pipeline_frame
    )

    ok, frame = cap.read()

    if not ok:
        print(
            f"Could not read pipeline frame {pipeline_frame}"
        )
        continue

    # --------------------------------------------------------
    # Draw every pipeline observation that exists at this frame
    # --------------------------------------------------------

    observations = []

    for event in events:

        for obs in event.get("observations", []):

            if obs.get("frame") == pipeline_frame:

                observations.append(
                    (
                        event,
                        obs.get("bbox")
                    )
                )

                break

    for event, bbox in observations:

        if bbox is None:
            continue

        x1, y1, x2, y2 = map(
            int,
            bbox
        )

        plate = (
            event
            .get("plate", {})
            .get("text")
        )

        tracklet = event.get(
            "tracklet_id",
            "unknown"
        )

        label = (
            f"{tracklet} | "
            f"{plate or 'NO_PLATE'}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    header = (
        f"GT frame: {gt_frame} | "
        f"Pipeline frame: {pipeline_frame} | "
        f"GT plate: {truth or 'NONE'}"
    )

    cv2.putText(
        frame,
        header,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        3
    )

    output_path = (
        f"gt_inspection/"
        f"gt_{gt_frame:04d}_"
        f"pipeline_{pipeline_frame:04d}.jpg"
    )

    cv2.imwrite(
        output_path,
        frame
    )

    print(
        f"GT {gt_frame:4d} "
        f"-> pipeline {pipeline_frame:4d} "
        f"-> {output_path}"
    )


cap.release()

print()
print(
    "Done. Open the images in gt_inspection/"
)
