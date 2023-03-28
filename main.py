import cv2

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open a connection to the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Live Object Detection", frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, exit the loop
    if key == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
