

import cv2
import numpy as np
import time
import onnxruntime as ort
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Emotion Labels
#####-------------------------######
emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}


# Load ONNX Emotion Model

emotion_model = ort.InferenceSession(
    "Face-Emotion-Recoginition/src/emotional_model.onnx",
    providers=["CPUExecutionProvider"]
)
emotion_input_name = emotion_model.get_inputs()[0].name


# MediaPipe Face Detector 

base_options = python.BaseOptions(
    model_asset_path='Face-Emotion-Recoginition/src/blaze_face_short_range.tflite'
)

options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_detection_confidence=0.6
)

detector = vision.FaceDetector.create_from_options(options)


# Video Capture

cap = cv2.VideoCapture(0)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    h, w, _ = frame.shape

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    result = detector.detect_for_video(mp_image, frame_id)

    if result.detections:
        for detection in result.detections:
            bbox = detection.bounding_box

            x1 = bbox.origin_x
            y1 = bbox.origin_y
            x2 = x1 + bbox.width
            y2 = y1 + bbox.height

            # Clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # FER preprocessing
            face_resized = cv2.resize(face, (48, 48)).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
            face_input = ((face_resized - mean) / std).transpose(2, 0, 1)[np.newaxis, :]

        
            outputs = emotion_model.run(
                None, {emotion_input_name: face_input.astype(np.float32)}
            )[0]

            emotion = emotion_labels[int(np.argmax(outputs))]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, emotion, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

    cv2.imshow("MediaPipe FER", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

