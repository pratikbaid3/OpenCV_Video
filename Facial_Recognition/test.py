import cv2
import numpy as np 
cam=cv2.VideoCapture(0)
p1_x=(int)(cam.get(cv2.CAP_PROP_FRAME_WIDTH)*3//4)
p1_y=(int)(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)*3.5//4)
p2_x=(int)(cam.get(cv2.CAP_PROP_FRAME_WIDTH)//4)
p2_y=(int)(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)//4)
while True:
    ret, im =cam.read()
    cv2.rectangle(im,pt1=(p2_x,p2_y),pt2=(p1_x,p1_y),color=(0,0,255),thickness=5)
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()