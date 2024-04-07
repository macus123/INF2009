import cv2

# Initialize the background subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

cap = cv2.VideoCapture(0)  # Change to the path of your video file to use a prerecorded video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Apply background subtraction
    fgMask = backSub.apply(frame)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Optional: Filter out small contours that could be just noise
    min_area = 500  # Adjust this value according to your scene
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Draw bounding boxes around the contours on the original frame
    for contour in large_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame with bounding boxes
    cv2.imshow('Motion Detection Surveillance', frame)

    # Optional: Display the foreground mask
    cv2.imshow('Foreground Mask', fgMask)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
