import cv2 as cv
import numpy as np

# Import face and eye cascades
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

# Use webcam
cap = cv.VideoCapture(0)

# created two booleans to set true for when eye or face is detected
# they get re initialized every iteration
is_eye = False
is_face = False

while True:
    ret, img = cap.read()

    # change frame color to gray
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # find face coordinates
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

    for(x, y, w, h) in faces:
        is_face = True
        # draw blue rectangle for face coordinates
        cv.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

        # since eyes can only be in the region where face is, we can only process the
        # image pixes of the face.
        face_gray  = gray_img[y:y+h, x:x+w]
        face_color = img[y:y+h, x:x+w]

        # Find eye coordinates
        eyes = eye_cascade.detectMultiScale(face_gray)


        for(e_x, e_y, e_w, e_h) in eyes:
            # draw red rectangle for eye coordinates
            cv.rectangle(face_color, (e_x, e_y), (e_x+e_w, e_y+e_h), (0, 0, 255), 2)
            is_eye = True


    # prints "eye & face" if there was a face and eye detected
    if (is_eye and is_face):
        print("eye & face")
        is_eye = False
        is_face = False
    # prints nothing if there was a face and eye detected
    else:
        print("nothing")

    q = cv.waitKey(30) & 0xff

    # quite once escape is pressed
    if q == 27:
        break

    cv.imshow('img', img)
cap.release()
cv.destroyAllWindows()