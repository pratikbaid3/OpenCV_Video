import cv2
import numpy as np
import os
subjects=["","Pratik Baid","Elvis Presley"]
def detect_face(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_cascade=cv2.CascadeClassifier('Cascades/lbpcascade_frontalface.xml')
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(20, 20))
    if(len(faces)==0):
        return None,None
    (x,y,w,h)=faces[0]
    return gray[y:y+w,x:x+h],faces[0]
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def prepare_training_data(data_folder_path):
    dirs=os.listdir(data_folder_path)
    faces=[]
    labels=[]
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue
        label=int(dir_name.replace("s",""))
        subject_dir_path=data_folder_path+"/"+dir_name
        subject_images_names=os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
            image_path=subject_dir_path+"/"+image_name
            print(image_path)
            image=cv2.imread(image_path)
            cv2.imshow("Training on image...",image)
            cv2.waitKey(100)
            face,rect=detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces,labels

def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    label = face_recognizer.predict(face)
    label_text = subjects[label[0]]
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    return img

print("Preparing data...")
faces,labels=prepare_training_data("training-data")
print("Data prepared")
print("Total faces: ",len(faces))
print("Tatal labels: ",len(labels))
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
print('Predicting images...')
'''cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    ret, img = cap.read()
    cv2.imshow('video',predict(img))
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break'''
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
print("Prediction complete")
predicted_img1=cv2.resize(predicted_img1,(600,800))
cv2.imshow(subjects[1], predicted_img1)
cv2.imshow(subjects[2], predicted_img2)
#cap.release
cv2.waitKey(0)
cv2.destroyAllWindows()