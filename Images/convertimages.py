import cv2
import numpy as np

n=0
false = 0
location = "imgSet\\"
type = ".jpeg"

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

open("imgSet.txt", 'w').close()

for fileName in ("myPicsPos.txt", "myPicsNeg.txt"):
    myFile = open(fileName, "r+")
    newFile = open("imgSet.txt", "a+")
    pos = fileName == "myPicsPos.txt"


    for line in myFile:
        line = line.split()
        if len(line) == 0:
            false +=1
            continue
        img = cv2.imread(line[0], 0)


        if (pos):
            faces = face.detectMultiScale(img, 2, 5)
        else:
            faces = []



        if pos:
            x=int(line[2])
            y=int(line[3])
            w=int(line[4])
            h=int(line[5])
            img = img[y:y + h, x:x + w]
            n += 1
            img = cv2.resize(img, (24, 24))
            newFile.write(location + "_{}".format(n) + type + " {} {} {}\n".format(24, 24, 1))

            cv2.imwrite(location + "_{}".format(n) + type, img)

        else:
            n += 1
            img = cv2.resize(img, (24, 24))
            newFile.write(location + "_{}".format(n) + type + " {} {} {}\n".format(24, 24, 0))

            cv2.imwrite(location + "_{}".format(n) + type, img)

print false