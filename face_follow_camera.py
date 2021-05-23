import cv2
import numpy as np


def detect_features(img):
    img_cp = np.copy(img)

    # Convert to grayscale
    gray = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    frontal_face = frontal_face_cascade.detectMultiScale(gray, 1.1, 4)
    profile_face = profile_face_cascade.detectMultiScale(gray, 1.1, 4)

    detected_face = frontal_face

    if len(frontal_face) != 0 and len(profile_face) != 0:
        detected_face = frontal_face + profile_face
    elif len(frontal_face) != 0:
        detected_face = frontal_face
    elif len(profile_face) != 0:
        detected_face = profile_face
    else:
        detected_face = ()

    face_coords = [0, 0, 0, 0]
    col3 = [row[2] for row in detected_face]
    if col3:
        large_face_idx = col3.index(max(col3))
        face_coords = detected_face[large_face_idx]

    # # -----------------------------
    # # Draw the rectangle around each face
    # for (x, y, w, h) in frontal_face:
    #     cv2.rectangle(img_cp, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # # Draw the rectangle around each profile_face
    # for (x, y, w, h) in profile_face:
    #     cv2.rectangle(img_cp, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # (x, y, w, h) = face_coords
    # cv2.rectangle(img_cp, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # # Show image with face detection
    # cv2.imshow('face detection', img_cp)
    # # -----------------------------
    
    bool_detection = detected_face!=()

    return bool_detection, face_coords


def create_roi(img, face_coords):
    img_h, img_w, _ = img.shape
    (x, y, w, h) = face_coords
    center_x = x + int(w/2)
    center_y = y + int(h/2)

    roi_h = min(img_h, h*3)
    roi_w = int(roi_h*3/4)

    raw_x = int(center_x - roi_w/2 )
    raw_y = int(center_y - roi_h/2) 

    if raw_x<0:
        roi_x = 0
    elif raw_x> img_w-roi_w:
        roi_x = img_w-roi_w
    else:
        roi_x = raw_x

    if raw_y<0:
        roi_y = 0
    elif raw_y> img_h-roi_h:
        roi_y = img_h-roi_h
    else:
        roi_y = raw_y

    roi_coords = (roi_x, roi_y, roi_w, roi_h)

    return roi_coords


def main():
    # To capture video from webcam.
    cap = cv2.VideoCapture(0)

    _, img = cap.read()
    roi_coords = (
        int((img.shape[1]-img.shape[0]*3/4)/2),
        0,
        int(img.shape[0]*3/4),
        img.shape[0]
    )

    while True:
        # Read the frame
        _, img = cap.read()

        bool_detected_face, face_coords = detect_features(img)

        roi_coords_new = create_roi(img, face_coords)
        if bool_detected_face:
            roi_coords = roi_coords_new

        # print(roi_coords)

        roi = img[roi_coords[1]: roi_coords[1]+roi_coords[3], roi_coords[0]:roi_coords[0]+roi_coords[2]]
        # roi = img[roi_coords[0]:roi_coords[0]+roi_coords[2]][roi_coords[1]: roi_coords[1]+roi_coords[3]]

        final_image = cv2.resize(roi, (300, 400))

        # Display
        # cv2.imshow('img', img)
        cv2.imshow('roi', final_image)

        # Stop if 'q' key is pressed
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    # Release the VideoCapture object
    cap.release()


if __name__ == '__main__':
    frontal_face_cascade = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')
    profile_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
    main()
