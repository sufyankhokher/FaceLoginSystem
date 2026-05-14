import face_recognition
import face_recognition_models
import cv2
import pickle
import numpy as np
import dlib
import threading
import time

# ── Direct dlib setup ──────────────────────────────────────────────────
face_detector = dlib.get_frontal_face_detector()
face_encoder = dlib.face_recognition_model_v1(
    face_recognition_models.face_recognition_model_location()
)

def get_face_encodings(rgb_image):
    detections = face_detector(rgb_image, 1)
    encodings = []
    locations = []
    for det in detections:
        top    = max(det.top(), 0)
        bottom = min(det.bottom(), rgb_image.shape[0])
        left   = max(det.left(), 0)
        right  = min(det.right(), rgb_image.shape[1])
        face_chip = rgb_image[top:bottom, left:right]
        if face_chip.size == 0 or face_chip.shape[0] < 10 or face_chip.shape[1] < 10:
            continue
        face_chip = cv2.resize(face_chip, (150, 150))
        encoding = np.array(face_encoder.compute_face_descriptor(face_chip, 0))
        encodings.append(encoding)
        locations.append((top, right, bottom, left))
    return encodings, locations

# ── Paths ──────────────────────────────────────────────────────────────
ENCODINGS_FILE = r"E:\VS CODE\AI-Projects\Advance Faceial Recognition\FaceLoginSystem\encodings.pkl"

# ── Load known faces ───────────────────────────────────────────────────
print("✅ Loading registered faces...")
with open(ENCODINGS_FILE, 'rb') as f:
    known_encodings, known_names = pickle.load(f)
print(f"✅ Loaded {len(known_names)} face(s): {known_names}")

# ── Start Webcam ───────────────────────────────────────────────────────
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("⏳ Warming up camera...")
for _ in range(10):
    cap.read()
time.sleep(1)
print("🎥 Webcam started. Press Q to quit.")

# ── Shared state ───────────────────────────────────────────────────────
face_locations = []
face_names     = []
face_colors    = []
face_statuses  = []
lock           = threading.Lock()
processing     = False
process_every  = 3
frame_count    = 0
ACCESS_GRANTED = False

def process_faces(small_rgb):
    global processing
    encs, locs = get_face_encodings(small_rgb)
    names, colors, statuses = [], [], []
    for enc in encs:
        if len(known_encodings) > 0:
            distances = face_recognition.face_distance(known_encodings, enc)
            best_match = np.argmin(distances)
            if distances[best_match] < 0.5:
                names.append(known_names[best_match])
                colors.append((0, 210, 0))
                statuses.append("GRANTED")
            else:
                names.append("Unknown")
                colors.append((0, 0, 220))
                statuses.append("DENIED")
        else:
            names.append("Unknown")
            colors.append((0, 0, 220))
            statuses.append("DENIED")
    with lock:
        global face_locations, face_names, face_colors, face_statuses
        face_locations = locs
        face_names     = names
        face_colors    = colors
        face_statuses  = statuses
    processing = False

# ── Helper: draw semi-transparent rectangle ────────────────────────────
def draw_transparent(frame, x1, y1, x2, y2, color, alpha=0.5):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame_count += 1
    if frame_count >= process_every and not processing:
        frame_count = 0
        processing  = True
        small       = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        small_rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        t = threading.Thread(target=process_faces, args=(small_rgb,))
        t.daemon = True
        t.start()

    with lock:
        locs     = list(face_locations)
        names    = list(face_names)
        colors   = list(face_colors)
        statuses = list(face_statuses)

    h, w = frame.shape[:2]
    any_granted = any(s == "GRANTED" for s in statuses)
    any_denied  = any(s == "DENIED"  for s in statuses)

    # ── Top status banner ──────────────────────────────────────────────
    if len(locs) == 0:
        draw_transparent(frame, 0, 0, w, 55, (40, 40, 40), 0.6)
        cv2.putText(frame, "🔍  Scanning for faces...", (14, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (200, 200, 200), 1)
    else:
        # Show summary: how many granted / denied
        granted_names = [n for n, s in zip(names, statuses) if s == "GRANTED"]
        denied_count  = sum(1 for s in statuses if s == "DENIED")
        banner_color  = (0, 140, 0) if any_granted and not any_denied else \
                        (0, 0, 160) if not any_granted else (120, 80, 0)
        draw_transparent(frame, 0, 0, w, 55, banner_color, 0.65)

        if granted_names:
            text = f"✅  ACCESS GRANTED — {', '.join(granted_names)}"
        else:
            text = f"❌  ACCESS DENIED"
        if denied_count > 0 and granted_names:
            text += f"   |   ⚠ {denied_count} Unknown"
        cv2.putText(frame, text, (14, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 1)

    # ── Per-face boxes ─────────────────────────────────────────────────
    for (top, right, bottom, left), name, color, status in zip(locs, names, colors, statuses):
        top *= 2; right *= 2; bottom *= 2; left *= 2

        # Face border
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Corner accents
        corner = 18
        thick  = 3
        for (x, y, dx, dy) in [(left, top, 1, 1), (right, top, -1, 1),
                                (left, bottom, 1, -1), (right, bottom, -1, -1)]:
            cv2.line(frame, (x, y), (x + dx * corner, y), color, thick)
            cv2.line(frame, (x, y), (x, y + dy * corner), color, thick)

        # Name tag below face
        label       = f"  {name}  "
        tag_h       = 32
        (tw, _), _  = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
        tag_x1      = left
        tag_x2      = left + tw + 4
        tag_y1      = bottom
        tag_y2      = bottom + tag_h

        draw_transparent(frame, tag_x1, tag_y1, tag_x2, tag_y2, color, 0.75)
        cv2.putText(frame, label, (tag_x1 + 2, tag_y1 + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        # Status badge top-right of box
        badge       = f" {status} "
        (bw, _), _  = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bx1         = right - bw - 6
        bx2         = right
        by1         = top - 24
        by2         = top
        draw_transparent(frame, bx1, by1, bx2, by2, color, 0.8)
        cv2.putText(frame, badge, (bx1 + 2, by1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if status == "GRANTED":
            ACCESS_GRANTED = True

    # ── Face count + quit hint ─────────────────────────────────────────
    count_text = f"Faces detected: {len(locs)}"
    cv2.putText(frame, count_text, (14, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(frame, "Press Q to quit", (w - 160, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.imshow("Face Login System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

granted_people = [n for n, s in zip(face_names, face_statuses) if s == "GRANTED"]
if ACCESS_GRANTED:
    print(f"\n✅ SESSION ENDED — Access was granted to: {', '.join(set(granted_people))}")
else:
    print("\n❌ SESSION ENDED — No authorized faces detected.")