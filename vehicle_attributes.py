# ============================================================
# Vehicle embedding + attribute extraction (type, colour)
#
# Embedding: uses a pretrained ResNet50 (ImageNet weights) with
# the final classification layer removed - NOT a properly
# fine-tuned Re-ID model, but a legitimate, zero-training way
# to get a similarity-capturing feature vector today. Good
# enough to catch "this looks like the same car" in most cases;
# upgrading to a real Re-ID-trained backbone (e.g. OSNet/FastReID)
# is a documented future improvement, not required for the demo.
#
# Attributes:
#   - type: uses a general pretrained YOLOv8 (COCO classes already
#     include car/truck/bus/motorcycle - zero extra training needed)
#   - colour: dominant colour via k-means clustering on pixels,
#     mapped to the nearest common colour name
# ============================================================

import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from ultralytics import YOLO
import cv2
import numpy as np

# --- Embedding model setup ---
_weights = ResNet50_Weights.DEFAULT
_embedding_model = resnet50(weights=_weights)
_embedding_model = torch.nn.Sequential(*list(_embedding_model.children())[:-1])  # drop final FC layer
_embedding_model.eval()

_preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- General object detector for vehicle TYPE (COCO classes) ---
_type_detector = YOLO("yolov8n.pt")  # pretrained COCO weights, auto-downloads if not present
VEHICLE_COCO_CLASSES = {"car", "truck", "bus", "motorcycle"}

# --- Common colour reference points for nearest-colour matching ---
COLOUR_REFERENCE = {
    "white":  (255, 255, 255),
    "black":  (0, 0, 0),
    "silver": (192, 192, 192),
    "grey":   (128, 128, 128),
    "red":    (180, 30, 30),
    "blue":   (30, 60, 150),
    "yellow": (220, 200, 40),
    "green":  (30, 120, 60),
    "brown":  (100, 60, 30),
}


def get_embedding(vehicle_crop):
    """Returns a 2048-dim feature vector for a vehicle crop image (BGR numpy array).
    Rounded to 4 decimal places - negligible precision loss for cosine similarity,
    but keeps JSON files readable and much smaller."""
    rgb = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
    tensor = _preprocess(rgb).unsqueeze(0)
    with torch.no_grad():
        features = _embedding_model(tensor)
    raw = features.squeeze().numpy().tolist()
    return [round(v, 4) for v in raw]


def cosine_similarity(vec_a, vec_b):
    """Compare two embeddings - close to 1.0 means likely the same vehicle."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_vehicle_type(vehicle_crop):
    """Returns the most likely vehicle type using a general pretrained detector."""
    results = _type_detector(vehicle_crop, verbose=False)[0]
    if len(results.boxes) == 0:
        return "unknown", 0.0

    best_conf = 0.0
    best_class = "unknown"
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = _type_detector.names[cls_id]
        conf = float(box.conf[0])
        if class_name in VEHICLE_COCO_CLASSES and conf > best_conf:
            best_conf = conf
            best_class = class_name

    return best_class, best_conf


def get_dominant_colour(vehicle_crop, k=3):
    """
    Finds the dominant colour in the crop using k-means, then maps it
    to the nearest common colour name. Filters out very bright (reflections/
    glare) and very saturated (tail lights, indicators) pixels first, since
    those can otherwise dominate the cluster and skew the result away from
    the actual body paint colour.
    """
    small = cv2.resize(vehicle_crop, (50, 50))

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]

    # keep only pixels that are NOT extremely bright (reflections/glare)
    # and NOT extremely saturated (tail lights, indicators, bright signage)
    mask = (brightness < 240) & (saturation < 200)

    pixels = small.reshape(-1, 3).astype(np.float32)
    mask_flat = mask.reshape(-1)
    filtered_pixels = pixels[mask_flat]

    # fallback to all pixels if filtering removed too much (e.g. a genuinely bright car)
    if len(filtered_pixels) < 20:
        filtered_pixels = pixels

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(filtered_pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    counts = np.bincount(labels.flatten())
    dominant_bgr = centers[np.argmax(counts)]
    dominant_rgb = (int(dominant_bgr[2]), int(dominant_bgr[1]), int(dominant_bgr[0]))

    best_name = "unknown"
    best_dist = float("inf")
    for name, ref_rgb in COLOUR_REFERENCE.items():
        dist = sum((a - b) ** 2 for a, b in zip(dominant_rgb, ref_rgb))
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name, dominant_rgb


def get_vehicle_attributes(vehicle_crop):
    """Convenience wrapper: returns everything at once for a vehicle crop."""
    embedding = get_embedding(vehicle_crop)
    vtype, type_conf = get_vehicle_type(vehicle_crop)
    colour, rgb = get_dominant_colour(vehicle_crop)

    return {
        "embedding": embedding,
        "type": vtype,
        "type_confidence": round(type_conf, 2),
        "colour": colour,
        "colour_rgb": rgb,
    }


if __name__ == "__main__":
    # Quick standalone test
    import sys
    test_image_path = sys.argv[1] if len(sys.argv) > 1 else "test_frames/frame1.jpg"
    crop = cv2.imread(test_image_path)

    if crop is None:
        print(f"Could not read image: {test_image_path}")
    else:
        attrs = get_vehicle_attributes(crop)
        print(f"Type: {attrs['type']} (confidence {attrs['type_confidence']})")
        print(f"Colour: {attrs['colour']} (RGB {attrs['colour_rgb']})")
        print(f"Embedding: {len(attrs['embedding'])}-dim vector, first 5 values: {attrs['embedding'][:5]}")