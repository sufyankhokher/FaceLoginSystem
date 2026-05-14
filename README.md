# 🔐 Face Login System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)
![dlib](https://img.shields.io/badge/dlib-19.24-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)

**A real-time face recognition login system — no passwords, just your face.**

</div>

---

## 🎬 Demo

> 🟢 **ACCESS GRANTED** — Recognized face detected  
> 🔴 **ACCESS DENIED** — Unknown face detected  
> ⬛ **Scanning...** — No face in frame  

---

## ✨ Features

- 📸 Register users with a single photo
- 🎥 Real-time webcam face detection & recognition
- 👥 Multi-user support
- 🔒 Threshold-based security to reject unknown faces
- ⚡ Fast processing with frame resizing optimization

---

## 🛠️ Built With

| Library | Purpose |
|---|---|
| `face_recognition` | Face detection & 128D encoding |
| `dlib` | Deep metric learning backbone |
| `OpenCV` | Webcam capture & UI |
| `NumPy` | Image array processing |
| `Pillow` | Image loading & conversion |
| `Pickle` | Saving/loading encodings |

---

## 📁 Project Structure

```
FaceLoginSystem/
├── 📂 photos/             # Add registration photos here
│   └── yourname.jpg
├── 📄 register.py         # Register faces from photos/
├── 📄 login.py            # Real-time webcam login
├── 📄 encodings.pkl       # Auto-generated face encodings
└── 📄 README.md
```

---

## ⚙️ Installation

### Prerequisites
- Windows 10/11
- Python 3.11
- Webcam

### Setup

**1. Clone the repository**
```bash
git clone https://github.com/sufyankhokher/FaceLoginSystem.git
cd FaceLoginSystem
```

**2. Create virtual environment**
```bash
py -3.11 -m venv face_env
face_env\Scripts\activate
```

**3. Install dlib (prebuilt wheel for Windows)**

Download `dlib-19.24.1-cp311-cp311-win_amd64.whl` from:
👉 https://github.com/z-mahmud22/Dlib_Windows_Python3.x

```bash
pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
```

**4. Install remaining dependencies**
```bash
pip install face_recognition opencv-python "numpy<2.0" Pillow
```

---

## 🚀 Usage

### 1️⃣ Register Your Face

Add a clear, front-facing photo to the `photos/` folder named after the person:
```
photos/
├── Sufyan.jpg
├── Ahmed.jpg
└── Sara.jpg
```

Then run:
```bash
python register.py
```

```
📸 Scanning photos folder...
  Processing: Sufyan.jpg
  ✅ Registered: Sufyan
✅ Done! 1 face(s) registered: ['Sufyan']
```

### 2️⃣ Launch Login System

```bash
python login.py
```

Look at the camera — the system will recognize you in real time!

---

## 🧠 How It Works

```
📷 Webcam Frame
      ↓
🔍 Face Detection (dlib HOG detector)
      ↓
📐 128D Face Encoding (deep metric learning)
      ↓
📏 Distance Comparison with registered faces
      ↓
✅ < 0.5 threshold → ACCESS GRANTED
❌ ≥ 0.5 threshold → ACCESS DENIED
```

---

## 🔧 Configuration

Adjust the recognition threshold in `login.py`:

```python
if distances[best_match] < 0.5:  # tweak this value
```

| Threshold | Behavior |
|---|---|
| `0.4` | Strict — safer but may reject valid users |
| `0.5` | Balanced ✅ (recommended) |
| `0.6` | Lenient — higher false acceptance rate |

---

## ⚠️ Limitations

- Requires decent lighting for accurate detection
- CPU-only on Windows (no native GPU support)
- Performance may vary with glasses, hats, or profile views
- Adding 2–3 photos per user improves recognition accuracy

---

## 📚 References

- [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)
- [dlib](http://dlib.net/)
- [OpenCV](https://docs.opencv.org/)
- *Artificial Intelligence: A Modern Approach* — Russell & Norvig

---

## 👨‍💻 Author

**Sufyan**  
BSCS @ NUML ISLAMABAD

---

<div align="center">
⭐ Star this repo if you found it useful!
</div>
