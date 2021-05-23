import cv2

def detect_features(img):
    frontal_face = frontal_face_cascade.detectMultiScale(img, 1.1, 4)
    profile_face = profile_face_cascade.detectMultiScale(img, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in frontal_face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Draw the rectangle around each profile_face
    for (x, y, w, h) in profile_face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return img

def main():
    # Load the cascade

    # To capture video from webcam. 
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame
        _, img = cap.read()
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces

        img = detect_features(img)

        # Display
        cv2.imshow('img', img)

        # Stop if 'q' key is pressed
        k = cv2.waitKey(1)
        if k==ord('q'):
            break

    # Release the VideoCapture object
    cap.release()

if __name__ == '__main__':
    frontal_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    profile_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
    main()