import cv2
import numpy as np

def preprocess_plate_crop(crop, upscale_factor=3):
    """
    Preprocess a plate crop to improve OCR accuracy:
    - Upscale (small crops lose detail for OCR)
    - Convert to grayscale
    - CLAHE contrast enhancement (helps with uneven lighting/glare)
    - Light sharpening (kept mild - aggressive sharpening on upscaled
      images can hallucinate false edges that OCR misreads as extra
      characters, which is worse than doing nothing)

    Returns the processed image (still 3-channel, since EasyOCR expects that).
    """
    h, w = crop.shape[:2]
    upscaled = cv2.resize(
        crop,
        (w * upscale_factor, h * upscale_factor),
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    # Mild sharpening only - center weight reduced from 5 to 3 to avoid
    # amplifying upscale artifacts into false character-like edges
    kernel = np.array([[0, -0.5, 0],
                        [-0.5, 3, -0.5],
                        [0, -0.5, 0]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)

    processed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    return processed


def blur_score(crop):
    """
    Measures sharpness using variance of the Laplacian - a standard,
    cheap blur-detection technique. Higher score = sharper image.
    Motion-blurred crops score low; use this to skip or downweight
    frames that are too blurred for OCR to have a real chance.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()