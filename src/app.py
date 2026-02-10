import os
import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image


# Paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
detect_model = os.path.join(BASE_DIR, "blaze_face_short_range.tflite")
onnxmodel = os.path.join(BASE_DIR, "emotional_model.onnx")


# MediaPipe Face Detector (IMAGE mode – correct!)

base_options = python.BaseOptions(model_asset_path=detect_model)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_detection_confidence=0.6
)
detector = vision.FaceDetector.create_from_options(options)


# Emotion model

emotion_model = ort.InferenceSession(
    onnxmodel, providers=["CPUExecutionProvider"]
)
emotion_input_name = emotion_model.get_inputs()[0].name

emotion_dict = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}


# Preprocessing

def preprocess_face(face):
    face = cv2.resize(face, (48, 48)).astype(np.float32) / np.float32(255.0)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    face = (face - mean) / std
    face = face.transpose(2, 0, 1)
    face = np.expand_dims(face, axis=0).astype(np.float32)

    return face

 
# Face detection + emotion prediction

def detect_and_predict(image):
    h, w, _ = image.shape

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )

    result = detector.detect(mp_image)

    if not result.detections:
        return image

    for detection in result.detections:
        bbox = detection.bounding_box

        x1 = max(0, bbox.origin_x)
        y1 = max(0, bbox.origin_y)
        x2 = min(w, x1 + bbox.width)
        y2 = min(h, y1 + bbox.height)

        face = image[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_input = preprocess_face(face)
        outputs = emotion_model.run(
            None, {emotion_input_name: face_input}
        )[0]

        emotion = emotion_dict[int(np.argmax(outputs))]
        confidence = float(np.max(outputs))

        # Draw UI
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"{emotion} ({confidence:.2f})"
        cv2.rectangle(image, (x1, y1 - 30), (x2, y1), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 5, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    return image

# --------------------------------------------------
# Streamlit UI (Enhanced)
# --------------------------------------------------
st.set_page_config(page_title="Face Emotion Recognition", layout="centered")

st.title("😄 Real-Time Face Emotion Recognition")
st.markdown(
    """
Upload an image or take a photo.  
Faces will be detected and emotions predicted in real time.
"""
)

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.sidebar.file_uploader(
        "📂 Upload an image",
        type=["jpg", "jpeg", "png"]
    )

with col2:
    camera_image = st.sidebar.camera_input("📸 Take a photo")

file = uploaded_file or camera_image

if file:
    image = Image.open(file).convert("RGB")
    image_np = np.array(image)

    with st.spinner("Detecting faces & emotions..."):
        result = detect_and_predict(image_np.copy())

    st.image(
        cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
        caption="Emotion Detection Result",
        use_column_width=True,
    )
