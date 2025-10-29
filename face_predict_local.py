import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

MODEL_PATH = "skin_classifier_effnetb0.h5"   # make sure the file is in same folder
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
IMG_SIZE = (224, 224)

# Class order based on alphabetical sorting
CLASS_NAMES = ["clear skin", "dark spots", "puffy eyes", "wrinkles"]

def preprocess_face(face):
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, IMG_SIZE)
    face_arr = img_to_array(face_resized) / 255.0
    face_arr = np.expand_dims(face_arr, axis=0)
    return face_arr

def predict_face(model, face_img):
    processed = preprocess_face(face_img)
    preds = model.predict(processed)[0]
    idx = np.argmax(preds)
    confidence = preds[idx]
    return CLASS_NAMES[idx], float(confidence)

def main():
    model = load_model(MODEL_PATH)
    detector = cv2.CascadeClassifier(HAAR_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not detected")
        return
    
    print("✅ Webcam started — Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            label, conf = predict_face(model, face)
            text = f"{label}: {conf*100:.1f}%"

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Skin Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
