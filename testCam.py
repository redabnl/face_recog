import cv2

# Initialize the camera
cap = cv2.VideoCapture(2)

# Test if the camera can be opened
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly, ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera when done
cap.release()
cv2.destroyAllWindows()
