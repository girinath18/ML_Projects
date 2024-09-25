import cv2

alg = "haarcascade_frontalface_default.xml"

# Load Haar Cascade for face detection
haar_cascade = cv2.CascadeClassifier(alg)  

# Initialize camera (0 for default camera)
cam = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

# Continuously read and display frames from the camera
while True:
    # Capture frame-by-frame
    ret, img = cam.read() 
    
    # If frame was read correctly
    if not ret:
        print("Failed to grab frame")
        break

    # Convert color image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

    # Detect faces
    faces = haar_cascade.detectMultiScale(gray_img, 1.3, 4) 

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("Face Detection", img)

    # Break loop on 'Esc' key 
    key = cv2.waitKey(10)  
    if key == 27:  
        break

# Release the camera and close all OpenCV
cam.release()
cv2.destroyAllWindows()
