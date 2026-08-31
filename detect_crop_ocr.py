# ============================================================
# Detect -> Crop -> OCR pipeline (run locally in VS Code)
# ============================================================

from ultralytics import YOLO
import cv2
import easyocr
import os
import re
from preprocess_plate import preprocess_plate_crop

# --- 1. Load models ---
detector = YOLO("plate_detector_v1.pt")
ocr_reader = easyocr.Reader(['en'], gpu=False)  # CPU fine for this

from plate_corrector import correct_plate

# --- 2. Run detection on a frame ---
image_path = "test_frames/frame5.jpg"
results = detector(image_path)[0]

frame = cv2.imread(image_path)

os.makedirs("plate_crops", exist_ok=True)

def clean_plate_text(text):
    """Strip anything that isn't a letter/digit, uppercase it."""
    return re.sub(r'[^A-Z0-9]', '', text.upper())

# --- 3. For each detected box, crop it and run OCR ---
crop_count = 0
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    class_name = detector.names[cls_id]
    det_confidence = float(box.conf[0])

    # crop the region out of the original frame
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        continue  # skip empty/invalid crops

    crop_count += 1
    crop_filename = f"plate_crops/crop_{crop_count}_{class_name}.jpg"
    cv2.imwrite(crop_filename, crop)

    print(f"\n--- Detection {crop_count} ---")
    print(f"Class: {class_name}   Detection confidence: {det_confidence:.2f}")
    print(f"Box: [{x1}, {y1}, {x2}, {y2}]")
    print(f"Saved crop: {crop_filename}")

    # Only run OCR on plate crops, not vehicle crops
    if class_name.lower() in ("plate", "license_plate", "licence_plate", "license-plate"):
        if det_confidence < 0.6:
            crop_for_ocr = preprocess_plate_crop(crop)
        else:
            crop_for_ocr = crop

        ocr_results = ocr_reader.readtext(crop_for_ocr)

        if ocr_results:
            raw_text = "".join([r[1] for r in ocr_results])
            confidences = [r[2] for r in ocr_results]
            avg_confidence = sum(confidences) / len(confidences)
            cleaned = clean_plate_text(raw_text)

            print(f"OCR raw text: {raw_text}")
            print(f"OCR cleaned text: {cleaned}")
            print(f"OCR confidence: {avg_confidence:.2f}")

            result = correct_plate(cleaned)
            print(f"Corrected: {result['corrected']}   Format: {result['format']}   Valid: {result['valid']}")
        else:
            print("OCR: no text detected")

print(f"\nTotal detections processed: {crop_count}")
print("Check the plate_crops/ folder to visually inspect each crop.")