import os
import cv2
import numpy as np

cap=cv2.VideoCapture(0)
cap.set(3,360)
cap.set(4,480)
face_detector=cv2.CascadeClassifier('../Cascades/haarcascade_frontalface_default.xml')
Id=input('Enter your Id: ')
sampleNum=0

def detect_face(frame):
    global sampleNum
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(gray,1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        sampleNum=sampleNum+1
        print(sampleNum)
        cv2.imwrite("data-set/user."+Id+'.'+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
    return frame

while True:
    ret,frame=cap.read()
    cv2.imshow('Frame',detect_face(frame))
    if cv2.waitKey(100) & 0xff == ord('q'):
        break
    elif sampleNum>=20:
        break

cap.release()
cv2.destroyAllWindows()