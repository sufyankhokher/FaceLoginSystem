# 🔐 Face Login System

A real-time face recognition-based login system built with Python. The system registers users through photos and authenticates them live via webcam — granting or denying access based on facial identity.

---

## 📌 Project Overview

This project was developed as part of the **Artificial Intelligence** course (BS Artificial Intelligence, 2023). It demonstrates the practical application of face recognition technology using deep metric learning to build a secure, password-free login system.

---

## 🎯 Features

- 📸 **Face Registration** — Register users by simply adding their photo
- 🎥 **Real-time Detection** — Live webcam feed with instant face recognition
- ✅ **Access Granted / Denied** — Clear visual feedback with color-coded banners
- 👥 **Multi-user Support** — Register multiple users at once
- 🔒 **Threshold-based Security** — Rejects unknown faces below confidence threshold

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| face_recognition | Face detection & encoding |
| dlib | Underlying deep learning model |
| OpenCV (cv2) | Webcam capture & UI rendering |
| NumPy | Image array processing |
| Pillow | Image loading & conversion |
| Pickle | Saving/loading face encodings |

---

## 📁 Project Structure

```
FaceLoginSystem/
├── photos/                  # Registration photos (one per person)
│   └── yourname.jpg
├── face_env/                # Python virtual environment
├── register.py              # Register faces from photos/
├── login.py                 # Real-time webcam login
├── encodings.pkl            # Auto-generated saved encodings
└── README.md
```

---

## ⚙️ Installation

### 1. Clone or Download the Project

```bash
gh repo clone sufyankhokher/FaceLoginSystem
cd FaceLoginSystem
```

### 2. Create Virtual Environment

```bash
py -3.11 -m venv face_env
face_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install cmake
pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
pip install face_recognition
pip install opencv-python
pip install numpy<2.0
pip install Pillow
```

> ⚠️ dlib requires a prebuilt wheel on Windows. Download it from:
> https://github.com/z-mahmud22/Dlib_Windows_Python3.x

---

## 🚀 Usage

### Step 1 — Register Your Face

1. Add a clear, front-facing photo to the `photos/` folder
2. Name the file after the person (e.g. `Sufyan.jpg`)
3. Run the registration script:

```bash
python register.py
```

This generates `encodings.pkl` with the saved face data.

### Step 2 — Launch the Login System

```bash
python login.py
```

- 🟢 **Green banner** = ACCESS GRANTED (recognized face)
- 🔴 **Red banner** = ACCESS DENIED (unknown face)
- ⬛ **Gray banner** = Scanning (no face detected)

Press **Q** to quit.

---

## 👤 Registering Multiple Users

Simply add more photos to the `photos/` folder:

```
photos/
├── Sufyan.jpg
├── Ahmed.jpg
└── Sara.jpg
```

Then run `register.py` again — all users will be registered automatically.

---

## 📊 How It Works

1. **Registration Phase**
   - Loads each photo from `photos/`
   - Detects the face using dlib's HOG-based detector
   - Generates a 128-dimension face encoding (deep metric learning)
   - Saves all encodings to `encodings.pkl`

2. **Login Phase**
   - Captures live webcam frames
   - Detects faces in real-time
   - Compares face encoding against all registered encodings
   - Grants access if distance < 0.5 threshold (configurable)

---

## 🔧 Configuration

In `login.py`, you can adjust the recognition threshold:

```python
if distances[best_match] < 0.5:   # Lower = stricter, Higher = more lenient
```

| Threshold | Behavior |
|---|---|
| 0.4 | Strict — may reject valid users |
| 0.5 | Balanced (default) |
| 0.6 | Lenient — may accept similar faces |

---

## ⚠️ Known Limitations

- Requires good lighting for accurate detection
- Performance may vary with glasses, hats, or extreme angles
- CPU-only on Windows (no GPU acceleration with native TensorFlow)
- One photo per user may reduce accuracy — adding 2-3 photos improves results

---

## 📚 References

- [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)
- [dlib library](http://dlib.net/)
- [OpenCV documentation](https://docs.opencv.org/)
- Artificial Intelligence — A Modern Approach, Stuart J. Russell & Peter Norvig

---

## 👨‍💻 Author

**Sufyan**
BSCS @ NUML
