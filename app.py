import streamlit as st
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import tempfile
import time

# Load the model and labels
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")
offset = 20
imgSize = 300
labels = ["Hello", "I Love You", "Thanks", "Sick", "Okay", "Hurt", "Help", "Washroom", "Angry", "Play", "Home", "You",
          "No", "Yes", "GoodMorning", "GoodNight", "Book", "Beautiful", "Cute", "Water", "Sleep", "Mother", "School",
          "Where", "Ugly", "Worst", "Failure", "Victory"]

st.title("Sign Language Detection")
st.write("This app uses a trained model to detect sign language gestures in real-time video.")

# Set up video capture
cap = cv2.VideoCapture(0)
frame_window = st.image([])

# Stream video and make predictions
while True:
    success, img = cap.read()
    if not success:
        st.write("Failed to read from camera.")
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        try:
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = int(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = int(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            confidence = prediction[index] * 100
            label = f"{labels[index]} ({confidence:.2f}%)"
            
            # Draw label and bounding box
            cv2.putText(img, label, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        except Exception as e:
            st.write(f"Error: {e}")

    # Display the frame
    frame_window.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Stop stream if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
