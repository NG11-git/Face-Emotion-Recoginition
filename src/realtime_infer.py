

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
    "Face-Emotion-Recoginition/emotion_resnet_cbam.onnx",
    providers=["CPUExecutionProvider"]
)
emotion_input_name = emotion_model.get_inputs()[0].name


# MediaPipe Face Detector 

base_options = python.BaseOptions(
    model_asset_path='Face-Emotion-Recoginition/blaze_face_short_range.tflite'
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
    frame = cv2.flip(frame, 1)

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

            x1 = int(bbox.origin_x)
            y1 = int(bbox.origin_y)
            x2 = int(x1 + bbox.width)
            y2 = int(y1 + bbox.height)

            # Clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            box_width = x2 - x1
            box_height = y2 - y1

            pad_w = int(box_width * 0.10)

            pad_top = int(box_height * 0.15)
            pad_bottom = int(box_height * 0.05)

            x1 = max(0, x1 + pad_w)
            x2 = min(w, x2 - pad_w)
            y1 = max(0, y1 + pad_top)
            y2 = min(h, y2 - pad_bottom)

            face = frame[y1:y2, x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            if face.size == 0:
                continue

            # FER preprocessing
            face_resized = cv2.resize(face, (224, 224)).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            
            face_resized = (face_resized - mean) / std
            face_input = np.transpose(face_resized, (2, 0, 1))
            face_input = np.expand_dims(face_input, axis=0).astype(np.float32)

        
            outputs = emotion_model.run(
                None, {emotion_input_name: face_input.astype(np.float32)}
            )[0]
            
            scores = outputs[0]
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            #print(probs)

            emotion = emotion_labels[int(np.argmax(probs))]
            confidence = np.max(probs)
            label = f"{emotion} ({confidence*100:.1f}%)"

        

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

    cv2.imshow("MediaPipe FER", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

