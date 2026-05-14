from PIL import Image
import numpy as np
import face_recognition

img_path = r"E:\VS CODE\AI-Projects\Advance Faceial Recognition\FaceLoginSystem\photos\sufyan.jpg"

img = Image.open(img_path)
print("Mode:", img.mode)
print("Size:", img.size)

img = img.convert('RGB')
arr = np.array(img)
print("Shape:", arr.shape)
print("Dtype:", arr.dtype)
print("Min/Max:", arr.min(), arr.max())

# Try finding face locations first
locations = face_recognition.face_locations(arr)
print("Face locations:", locations)