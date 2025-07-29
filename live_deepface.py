import cv2
from deepface import DeepFace

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze frame for emotion only
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Get dominant emotion
        dominant_emotion = result[0]['dominant_emotion']

        # Put emotion text on the frame
        cv2.putText(frame, dominant_emotion, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    except Exception as e:
        print("Error:", e)

    cv2.imshow("Live Emotion Detection - DeepFace", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
