import cv2
import os
import numpy as np 
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create() 
detector= cv2.CascadeClassifier("../Cascades/haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    faceSamples=[]
    Ids=[]
    imagePaths=[]
    for f in os.listdir(path):
        if not f.startswith('.'):
            imagePaths.append(os.path.join(path,f))

    #imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces=detector.detectMultiScale(imageNp)
        for(x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids

faces,Ids=getImagesAndLabels('data-set')
recognizer.train(faces,np.array(Ids))
recognizer.save('trainner/trainner.yml') 
