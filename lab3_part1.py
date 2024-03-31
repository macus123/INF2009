import speech_recognition as sr
import os
import cv2

# Function to capture and save an image
def capture_image():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Capture one frame
    ret, frame = cap.read()

    # Save the captured image
    cv2.imwrite('captured_image.jpg', frame)

    # Release the webcam
    cap.release()

    print("Image captured and saved!")

# obtain audio from the microphone
r = sr.Recognizer()  # Initializing the Recognizer class
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)  # Identify the ambient noise and be silent during this phase
    os.system('clear')
    print("Say something!")
    audio = r.listen(source)  # Listening from microphone

# recognize speech using Google Speech Recognition
try:
    command = r.recognize_google(audio)
    print(command + "command activated!")
    
    # Check if the recognized command is the activation command
    if command.lower() == "hello":
        # print("Hello! How can I assist you?")
        # Capture and save an image
        capture_image()
    else:
        print("Sorry, I didn't understand that command.")

except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
