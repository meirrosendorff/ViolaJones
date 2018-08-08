import cv2
import numpy as np

cap = cv2.VideoCapture(0)

type = ".jpeg"

front_dir = "faceNew\\"

file = open(front_dir+"currNum.txt","r")
pics = open(front_dir+"myPics.txt","a")

imNum=int(file.read())


while True:


    _, frame = cap.read()

    imNum += 1
    path=front_dir + "im_{}".format(imNum) + type

    cv2.imwrite(path, frame)
    cv2.imshow('frame', frame)
    pics.write(path+"\n")

    print imNum

    if cv2.waitKey(200) & 0xFF == ord('q'):
        file=open(front_dir + "currNum.txt","w")
        file.write(str(imNum))

        break


