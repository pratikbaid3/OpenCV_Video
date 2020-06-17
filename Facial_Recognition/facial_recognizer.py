import cv2
import numpy as np 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
faceCascade= cv2.CascadeClassifier("../Cascades/haarcascade_frontalface_default.xml")
cam=cv2.VideoCapture(0)
p1_x=(int)(cam.get(cv2.CAP_PROP_FRAME_WIDTH)*3//4.2)
p1_y=(int)(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)*3.5//4)
p2_x=(int)(cam.get(cv2.CAP_PROP_FRAME_WIDTH)//4)
p2_y=(int)(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)//3.8)
while True:
    ret, im =cam.read()
    cv2.rectangle(im,pt1=(p2_x,p2_y),pt2=(p1_x,p1_y),color=(0,0,255),thickness=5)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf<60):
            if(Id==1):
                Id="Pratik Baid"
            elif(Id==3):
                Id="Rakesh Baid"
            elif(Id==2):
                Id="Tejas Baid"
        else:
            Id="Unknown"
        cv2.putText(im, str(Id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.putText(im,str(100-conf),(x+w,y+h),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
