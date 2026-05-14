import face_recognition_models
import cv2
import pickle
import os
import numpy as np
import dlib
from PIL import Image

# ── Direct dlib setup (bypass broken face_recognition wrapper) ─────────
face_detector = dlib.get_frontal_face_detector()
face_encoder = dlib.face_recognition_model_v1(
    face_recognition_models.face_recognition_model_location()
)


def get_encoding_from_image(rgb_image):
    """Get first face encoding from an image using dlib directly."""
    detections = face_detector(rgb_image, 1)
    if len(detections) == 0:
        return None
    det = detections[0]
    top = max(det.top(), 0)
    bottom = min(det.bottom(), rgb_image.shape[0])
    left = max(det.left(), 0)
    right = min(det.right(), rgb_image.shape[1])
    face_chip = rgb_image[top:bottom, left:right]
    if face_chip.size == 0 or face_chip.shape[0] < 10 or face_chip.shape[1] < 10:
        return None
    face_chip = cv2.resize(face_chip, (150, 150))
    return np.array(face_encoder.compute_face_descriptor(face_chip, 0))


# ── Paths ──────────────────────────────────────────────────────────────
PHOTOS_DIR = r"E:\VS CODE\AI-Projects\Advance Faceial Recognition\FaceLoginSystem\photos"
ENCODINGS_FILE = r"E:\VS CODE\AI-Projects\Advance Faceial Recognition\FaceLoginSystem\encodings.pkl"

known_encodings = []
known_names = []

print("📸 Scanning photos folder...")
print(f"   Folder: {PHOTOS_DIR}\n")

for filename in os.listdir(PHOTOS_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    name = os.path.splitext(filename)[0]
    image_path = os.path.join(PHOTOS_DIR, filename)
    print(f"  Processing: {filename}")

    # Load and convert to RGB numpy array
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)

    # Resize large images for faster processing
    h, w = image.shape[:2]
    if h > 800 or w > 800:
        scale = 800 / max(h, w)
        image = np.array(Image.fromarray(image).resize(
            (int(w*scale), int(h*scale))))

    encoding = get_encoding_from_image(image)

    if encoding is None:
        print(f"  ⚠️  No face found in {filename}, skipping.")
        continue

    known_encodings.append(encoding)
    known_names.append(name)
    print(f"  ✅ Registered: {name}")

# Save encodings
with open(ENCODINGS_FILE, 'wb') as f:
    pickle.dump((known_encodings, known_names), f)

print(f"\n✅ Done! {len(known_names)} face(s) registered: {known_names}")
