import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Initialize dlib's face detector and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate the eye aspect ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define constants for the EAR threshold and consecutive frames
EAR_THRESHOLD = 0.3
CONSECUTIVE_FRAMES = 3

# Initialize the frame counter
blink_counter = 0

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (256, 256))
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = detector(gray)
    
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Predict facial landmarks
        landmarks = predictor(gray, face)
        
        # Extract the left and right eye coordinates
        leftEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)], dtype="int")
        rightEye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)], dtype="int")
        
        # Calculate the eye aspect ratio for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        # Average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        
        # Check if the eye aspect ratio is below the blink threshold
        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            # If the eyes were closed for a sufficient number of consecutive frames, then a blink is counted
            if blink_counter >= CONSECUTIVE_FRAMES:
                print("Blink detected")
            # Reset the blink counter
            blink_counter = 0

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
