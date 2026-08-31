# ============================================================
# OCR wrapper - RapidOCR (ONNX) for Indian license plates.
# Fast, CPU-friendly, no heavy dependency drama (unlike PaddleOCR).
#
# IMPORTANT: this file does NOT do character-level correction.
# That job belongs entirely to plate_corrector.py, which is
# context-aware (knows which position should be a digit vs a
# letter, knows real state codes, knows plate format structure).
#
# An earlier version of this file had a blind, unconditional
# character-substitution table applied to every read - it mapped
# '0'->'O' AND 'O'->'0' in the same dict (destroying already-correct
# characters ~50% of the time), and a hardcoded 'every plate is from
# Maharashtra' rule that rewrote NH/HH/LL/WI prefixes to MH. That
# actively corrupted good OCR reads before they ever reached the
# real corrector. Removed entirely - raw OCR output now flows
# straight to plate_corrector.py, which does this job correctly.
# ============================================================

import cv2
import re
from rapidocr_onnxruntime import RapidOCR

_rapid_ocr = RapidOCR()


def enhance_for_ocr(crop):
    """Preprocess plate crop for better OCR results."""
    if crop is None or crop.size == 0:
        return None

    processed = crop.copy()

    if processed.shape[1] < 200:
        scale = 200 / processed.shape[1]
        processed = cv2.resize(processed, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)

    if len(processed.shape) == 3:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return binary


def clean_text(text):
    """Strip to alphanumeric uppercase only - no character substitution here."""
    if not text:
        return ""
    return re.sub(r'[^A-Za-z0-9]', '', text).upper()


def read_plate_rapid(crop, preprocess=True):
    """
    Read plate using RapidOCR. Returns list of (text, confidence) tuples,
    cleaned to alphanumeric-uppercase only - no character correction
    applied here. That's plate_corrector.py's job.
    """
    if crop is None or crop.size == 0:
        return []

    processed_crop = enhance_for_ocr(crop) if preprocess else crop
    if processed_crop is None:
        processed_crop = crop

    results = []
    try:
        result = _rapid_ocr(processed_crop)

        if result and len(result) > 0:
            ocr_result = result[0] if isinstance(result, tuple) else result

            if ocr_result and isinstance(ocr_result, list):
                for item in ocr_result:
                    if len(item) < 2:
                        continue

                    # RapidOCR result items are typically [bbox, text, confidence]
                    text = item[1]
                    confidence = item[2] if len(item) > 2 else 0.5

                    if not text or not str(text).strip():
                        continue

                    cleaned = clean_text(str(text))
                    if cleaned and len(cleaned) >= 3:
                        try:
                            conf = float(confidence)
                        except (TypeError, ValueError):
                            conf = 0.5
                        results.append((cleaned, conf))

    except Exception as e:
        print(f"    (RapidOCR error: {e})")

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def read_plate_ensemble(crop, preprocess=True):
    """
    Entry point used by full_pipeline.py.
    Returns list of (text, confidence) tuples with NO character-level
    correction applied - raw cleaned OCR output only. All correction
    happens downstream in plate_corrector.py, which has actual format
    and position awareness.
    """
    return read_plate_rapid(crop, preprocess=preprocess)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            print(f"Could not read image: {sys.argv[1]}")
        else:
            print(f"Testing RapidOCR on: {sys.argv[1]}")
            results = read_plate_ensemble(img)
            for i, (text, conf) in enumerate(results):
                print(f"  {i+1}. '{text}' (confidence: {conf:.3f})")
    else:
        print("Usage: python ocr_ensemble.py <image_path>")