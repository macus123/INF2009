import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Parameters for gesture recognition
gesture_thresholds = {
    'open_hand': 0.5,
    'thumbs_up': 0.5,
    'point_index': 0.5
}

# Function to recognize gestures based on hand landmarks
def recognize_gestures(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    # You can add other landmarks to improve gesture recognition
    
    # Simple logic to recognize thumbs up and pointing index
    if thumb_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y:
        if index_finger_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
            return 'thumbs_up'
        else:
            return 'point_index'
    else:
        return 'open_hand'

# Function to control video player based on gesture
def control_video_player(gesture):
    if gesture == 'open_hand':
        # Send play/pause command
        print('Play/Pause')
    elif gesture == 'thumbs_up':
        # Increase volume
        print('Volume Up')
    elif gesture == 'point_index':
        # Decrease volume
        print('Volume Down')

# Initialize MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue
    
    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and find hand landmarks
    results = hands.process(rgb_frame)
    
    # Draw the hand annotations on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Recognize the gesture
            gesture = recognize_gestures(hand_landmarks)
            # Control the video player
            control_video_player(gesture)
    
    # Display the image
    cv2.imshow('Gesture Based Video Player Control', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
