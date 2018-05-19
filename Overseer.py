import numpy as np
import cv2

cap = cv2.VideoCapture(0)

import os

def cheminAbsolu(file):
    root = os.path.dirname(__file__)
    abs_path = os.path.join(root, file)
    return abs_path

face_cascade = cv2.CascadeClassifier('C:\Users\Vava\Documents\Fac\opencv\sources\data\haarcascades_cuda\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\Users\Vava\Documents\Fac\Recherche\\venv\FeaturesDetection\HaarCascadeMCS\haarcascade_mcs_eyepair_small.xml')
mouth_cascade = cv2.CascadeClassifier('C:\Users\Vava\Documents\Fac\Recherche\\venv\FeaturesDetection\HaarCascadeMCS\haarcascade_mcs_mouth.xml')

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10,minSize=(75, 75))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        mouth = mouth_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
               cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        for (ex, ey, ew, eh) in mouth:
               cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)



    # Display the resulting frame
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
