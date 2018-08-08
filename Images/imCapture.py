import cv2
import numpy as np

cap = cv2.VideoCapture(0)

type = ".jpg";

front_dir = "newFace\\"

file = open(front_dir+"currNum.txt","r")
pics = open(front_dir+"myPics.txt","a")

imNum=int(file.read())

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#face = cv2.CascadeClassifier('haarcascade_profileface.xml')

x=0; y=0; w=0; h=0

while True:


    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, 2, 5)


    if len(faces):
        (x,y,w,h)=faces[0]

    if (x):

        imNum += 1
        if imNum < 820:
            continue
        path=front_dir + "im_{}".format(imNum) + type
        frame = frame[y:y+h, x:x+w]
        frame = cv2.resize(frame, (24,24))
        cv2.imwrite(path, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        pics.write(path + " {} {} {}".format(24,24,0)+"\n")
        print imNum
    else:
        cv2.imshow('frame', frame)



    if cv2.waitKey(200) & 0xFF == ord('q'):
        file=open(front_dir + "currNum.txt","w")
        file.write(str(imNum))

        break


