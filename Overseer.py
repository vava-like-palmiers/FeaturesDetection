import numpy as np
import cv2

cap = cv2.VideoCapture(0)

import os
import string
import platform
import math

def cheminAbsoluWindows(file):
    root = os.path.abspath(file)
    root = root.replace("\\ve", "\\\\ve")
    return root

def cheminAbsoluLinux(file):
    root = os.path.abspath(file)
    return root

if(platform.system() == 'Windows'):
    face_cascade = cv2.CascadeClassifier(cheminAbsoluWindows('FeaturesDetection\HaarCascadeMCS\haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(cheminAbsoluWindows('FeaturesDetection\HaarCascadeMCS\haarcascade_mcs_eyepair_big.xml'))
    mouth_cascade = cv2.CascadeClassifier(cheminAbsoluWindows('FeaturesDetection\HaarCascadeMCS\haarcascade_mcs_mouth.xml'))
elif(platform.system() == 'Linux'):
    face_cascade = cv2.CascadeClassifier(cheminAbsoluLinux('HaarCascadeMCS/haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(cheminAbsoluLinux('HaarCascadeMCS/haarcascade_mcs_eyepair_big.xml'))
    mouth_cascade = cv2.CascadeClassifier(cheminAbsoluLinux('HaarCascadeMCS/haarcascade_mcs_mouth.xml'))

safety = 5 #defini le nombre de points dans notre tableau

def distance(box1, box2):
    distp1 = math.sqrt(((box1[0]-box2[0])*(box1[0]-box2[0]))+((box1[1]-box2[1])*(box1[1]-box2[1])))
    distp2 = math.sqrt(((box1[0]+box1[2]-box2[0]-box2[2])*(box1[0]+box1[2]-box2[0]-box2[2]))+((box1[1]+box1[3]-box2[1]-box2[3])*(box1[1]+box1[3]-box2[1]-box2[3])))
    return distp1 + distp2

def push(table,value):
    if len(table)>=safety:
        del table[0]
    table.append(value)

def coefficientRegression(table):
    n=len(table)
    sumXiYi=0
    sumXi=0
    sumYi=0
    sumXipow2=0
    sumYipow2=0
    for i in range(0,n-1) :
        # calcul de la somme Xi*Yi
        sumXiYi+=i*table[i]
        # calcul de la somme Xi
        sumXi+=i
        # calcul de la somme Yi
        sumYi+=table[i]
        # calcul de la somme Xi^2
        sumXipow2+=i*i
        # calcul de la somme Yi^2
        sumYipow2+=table[i]*table[i]

    return ( (n*sumXiYi)-(sumXi*sumYi) ) / ( (n*sumXipow2) - sumXipow2 )

def coefficientCorrelation(table):
    n=len(table)
    xm=0
    ym=0
    #calcul des moyennes
    for i in range(0,n-1):
        xm+=i
        ym+=table[i]
    xm=xm/n
    ym=ym/n

    sumi_m=0
    sumXi_Xmpow2=0
    sumYi_Ympow2=0
    for i in range(0,n-1):
        #calcul de la somme (Xi-Xm)*(Yi-Ym)
        sumi_m+=(i-xm)*(table[i]-ym)
        #calcul de la somme (Xi-Xm)^2
        sumXi_Xmpow2+=(i-xm)*(i-xm)
        #calcul de la somme (Yi-Ym)^2
        sumYi_Ympow2+=(table[i]-ym)*(table[i]-ym)

    return sumi_m / math.sqrt(sumXi_Xmpow2 * sumYi_Ympow2)

def variations(table):
    if len(table)>=safety:
        return coefficientRegression(table)#*((-coefficientCorrelation(table)+1)/2)
    else:
        return 0

#seuil au-dela duquel il y a detection
seal = 0.006
#table des valeurs de la bouche
tableMouth=[]

oldval=0

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10,minSize=(75, 75))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        roi_mouth = gray[y+(h/2):y + h, x:x + w]
        roi_eyes = gray[y:y + (h/2), x:x + w]

        roi_color_mouth = img[y+(h/2):y + h, x:x + w]
        roi_color_eyes = img[y:y + (h/2), x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_eyes)
        mouth = mouth_cascade.detectMultiScale(roi_mouth)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color_eyes, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        #Pour detecter une seule bouche, on considere qu'elle sera la feature de largeur la plus grande
        #les erreurs de detection etant souvent les narines ou le menton


        mindist=math.sqrt((x*x)+(y*y))
        H=0
        W=0
        X=0
        Y=0
        for (ex, ey, ew, eh) in mouth:
            if(len(tableMouth)==0):
                oldbox=(ex, ey, ew, eh)
            if(distance((ex, ey, ew, eh), oldbox)<mindist):
                mindist=distance((ex, ey, ew, eh), oldbox)
                H=eh
                W=ew
                X=ex
                Y=ey

        cv2.rectangle(roi_color_mouth, (X, Y), (X + W, Y + H), (0, 255, 0), 2)
        oldbox=(X,Y,W,H)

        scale = math.sqrt((w*w*4/9)+(h*h/4))
        push(tableMouth,  (math.sqrt(W*W+H*H) - oldval)/scale )
        oldval = math.sqrt(W*W+H*H)
        break

    if math.fabs(variations(tableMouth)) > seal:
        print(variations(tableMouth))

    # Display the resulting frame
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
