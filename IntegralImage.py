import numpy as np
import math
import cv2
import time


class feature:
    """
    featureType 1 is white grey
                2 is grey/white
                3 is white grey white
                4 is white/grey grey/white
                5 is white/grey/white
                6 is grey white
                7 is white/grey
                8 is grey white grey
                9 is grey/white white/grey
                10 is grey/white/grey
    """
    def __init__(self, (featureType, x,y, w, h), beta):

        self.myFeature = (featureType, x, y, w, h)
        self.threshold = 0.5
        if beta == 0:
            self.alpha = 1
        else:
            self.alpha = math.log(1/beta, 10)


class featureSet:

    """contains a set of tuples relating to features in the form {featureType, w, h}
    where featureType 1 is white grey
                    2 is grey/white
                    3 is white grey white
                    4 is white/grey grey/white
                    5 is white/grey/white
                    6 is grey white
                    7 is white/grey
                    8 is grey white grey
                    9 is grey/white white/grey
                    10 is grey/white/grey
    """

    def __init__(self):

        self.myFeatures = []
        self.numLevels = -1

    def addFeature(self, cascadeStage, stage):

        if stage > self.numLevels:
            self.myFeatures.append([])
            self.numLevels += 1

        self.myFeatures[stage] = cascadeStage

    def unused(self, (featureType, x, y, w, h)):

        for stage in self.myFeatures:
            for currFeature in stage:
                if (featureType, x, y, w, h) == currFeature.myFeature:
                    return False

        return True

    def adjustThreshold(self, change):

        for currfeature in self.myFeatures[self.numLevels]:
            currfeature.threshold += change


class integralImage:
    """Creates an integral Image of a given image"""

    def __init__(self, imageWindow, rows=24, cols=24, isPos=0):

        self.isPos = bool(isPos)
        self.available = True
        self.correctID = False
        self.weight = 1
        self.rows = rows
        self.cols = cols


        #innitializes intImg 2D list
        self.intImg = []
        for y in range(rows):
            self.intImg.append([])
            for x in range(cols):
                self.intImg[y].append([])

        for x in range(cols):
            for y in range(rows):

                if (y-1<0):
                    above = 0
                else:
                    above = self.intImg[y-1][x]

                if (x-1<0):
                    left = 0
                else:
                    left = self.intImg[y][x-1]

                if y-1<0 or x-1<0:
                    old = 0
                else:
                    old = self.intImg[y-1][x-1]

                curr = imageWindow[y][x]

                self.intImg[y][x] = left + above + curr - old

    def rectVal(self, x, y, w, h):
        # returns intImg val at bottom right + intImg val at top left - bottom left - top right

        if (y-1<0):
            topRight = 0
        else:
            topRight = self.intImg[y - 1][x + w]

        if (x-1<0):
            bottomLeft = 0
        else:
            bottomLeft = self.intImg[y + h][x - 1]

        if (x-1<0 or y-1<0):
            topLeft = 0
        else:
            topLeft = self.intImg[y - 1][x - 1]

        bottomRight = self.intImg[y + h][x + w]

        return topLeft + bottomRight - bottomLeft - topRight

    def condition(self, num):
        if 800>num>300:   # left is 800>num>200 right is 1000>num>200
            return True
        return False

    def featureIsFound(self, (featureType, x, y, w, h)):

        # featureType 1 is white grey
        #             2 is grey/white
        #             3 is white grey white
        #             4 is white/grey grey/white
        #             5 is white/grey/white
        #             6 is grey white
        #             7 is white/grey
        #             8 is grey white grey
        #             9 is grey/white white/grey
        #             10 is grey/white/grey

        if featureType == 1 and w%2 == 0:
            if x + w < self.cols and y + h < self.rows:
                w = int(w / 2)
                white = self.rectVal(x, y, w, h)
                grey = self.rectVal((x + w), y, w, h)
                total = grey - white
                if self.condition(total):
                    return True


        elif featureType == 2 and h%2 == 0:
            if x + w < self.cols and y + h < self.rows:
                h = int(h / 2)
                grey = self.rectVal(x, y, w, h)
                white = self.rectVal(x, (y + h), w, h)
                total = grey - white
                if self.condition(total):
                    return True


        elif featureType == 3 and w%3 == 0:
            if x + w < self.cols and y + h < self.rows:
                w = int(w / 3)
                white1 = self.rectVal(x, y, w, h)
                white2 = self.rectVal((x + 2 * w), y, w, h)
                grey = self.rectVal((x + w), y, w, h)
                total = grey - white1 - white2
                if self.condition(total):
                    return True


        elif featureType == 4 and w%2 == 0  and h%2 == 0:
            if x + w < self.cols and y + h < self.rows:
                w = int(w / 2)
                h = int(h / 2)
                white1 = self.rectVal(x, y, w, h)
                white2 = self.rectVal((x + w), (y + h), w, h)
                grey1 = self.rectVal((x + w), y, w, h)
                grey2 = self.rectVal(x, (y + h), w, h)
                total = grey1 + grey2 - (white1 + white2)
                if self.condition(total):
                    return True

        elif featureType == 5 and h%3 == 0:
            if x + w < self.cols and y + h < self.rows:
                h = int(h / 3)
                white1 = self.rectVal(x, y, w, h)
                white2 = self.rectVal(x, (y + 2 * h), w, h)
                grey = self.rectVal(x, (y + h), w, h)
                total = grey - white1 - white2
                if self.condition(total):
                    return True


        elif featureType == 6 and w%2 == 0:
            if x + w < self.cols and y + h < self.rows:
                w = int(w / 2)
                grey = self.rectVal(x, y, w, h)
                white = self.rectVal((x + w), y, w, h)
                total = grey - white
                if self.condition(total):
                    return True


        elif featureType == 7 and h%2 == 0:
            if x + w < self.cols and y + h < self.rows:
                h = int(h / 2)
                white = self.rectVal(x, y, w, h)
                grey = self.rectVal(x, (y + h), w, h)
                total = grey - white
                if self.condition(total):
                    return True


        elif featureType == 8 and w%3 == 0:
            if x + w < self.cols and y + h < self.rows:
                w = int(w / 3)
                white1 = self.rectVal(x, y, w, h)
                white2 = self.rectVal((x + 2 * w), y, w, h)
                grey = self.rectVal((x + w), y, w, h)
                total = grey - white1 - white2
                if self.condition(total):
                    return True


        elif featureType == 9 and w%2 == 0  and h%2 == 0:
            if x + w < self.cols and y + h < self.rows:
                w = int(w / 2)
                h = int(h / 2)
                grey1 = self.rectVal(x, y, w, h)
                grey2 = self.rectVal((x + w), (y + h), w, h)
                white1 = self.rectVal((x + w), y, w, h)
                white2 = self.rectVal(x, (y + h), w, h)
                total = grey1 + grey2 - (white1 + white2)
                if self.condition(total):
                    return True

        elif featureType == 10 and h%3 == 0:
            if x + w < self.cols and y + h < self.rows:
                h = int(h / 3)
                grey1 = self.rectVal(x, y, w, h)
                grey2= self.rectVal(x, (y + 2 * h), w, h)
                white = self.rectVal(x, (y + h), w, h)
                total = grey1 + grey2 - white
                if self.condition(total):
                    return True

        return False


class integralImageList:
    """Creates a list of integral images from a given text document
        The text dcument must contain the name of each image, its height,
        width and a 1 or 0 depending on if its a positive image or not.
        with each image on its own line
        i.e the doc must be in the form:
           imageName w h 1/0
        e.g img.jpg 24 48 1
        """

    def __init__(self, fileName):

        self.myFeatures = featureSet()

        myFile = open(fileName, "r")

        self.intImgList=[]
        self.classifierLevel = 0

        for line in myFile:
            line = line.split()
            self.intImgList.append(integralImage(cv2.imread(line[0],0), int(line[1]),int(line[2]),int(line[3])))

        myFile.close()

        self.initializeWeights()
        self.normalizeWeights()

        self.rows = self.intImgList[0].rows
        self.cols = self.intImgList[0].cols

    def initializeWeights(self):

        size = 0

        for intImg in self.intImgList:
            if intImg.available:
                size += 1


        for intImg in self.intImgList:
            if intImg.available:
                intImg.weight = 1.0/size
        return

    def normalizeWeights(self):

        totalWeight=0.0

        for intImg in self.intImgList:
            if intImg.available:
                totalWeight += intImg.weight

        if totalWeight == 0:
            return
        for intImg in self.intImgList:
            if intImg.available:
                intImg.weight /= totalWeight

    def evalFeature(self, (featureType, x, y, w, h)):

        # featureType 1 is white grey
        #             2 is grey/white
        #             3 is white grey white
        #             4 is white/grey grey/white
        error = 0.0

        if (featureType == 1 or featureType == 4 or featureType == 6 or featureType == 9) and w % 2 != 0:
            return 1.0
        if (featureType == 2 or featureType == 4 or featureType == 7 or featureType == 9) and h % 2 != 0:
            return 1.0
        if (featureType == 3 or featureType == 8) and w % 3 != 0:
            return 1.0
        if (featureType == 5 or featureType == 10) and w % 3 != 0:
            return 1.0

        for intImg in self.intImgList:
            if not intImg.available:
                continue

            found = intImg.featureIsFound((featureType, x, y, w, h)) # bool true if feature is found in image
            error += intImg.weight * abs(found - intImg.isPos)

        return error

    def setCorrectID(self, feature):
        featureType, x, y, w, h = feature.myFeature
        for intImg in self.intImgList:
            intImg.correctID = (intImg.featureIsFound((featureType, x, y, w, h)) == intImg.isPos) # sets correctID to true if the img is correctly identified

    def findBestFeature(self):
        bestFeature = ((1, 0, 0, 3, 3))
        lowestErr = self.evalFeature((1, 0, 0, 3, 3))
        for w in range(3, self.cols):
            for h in range(3, self.rows):
                for x in range(self.cols-w):
                    for y in range(self.rows-h):
                        for featureType in range(1,11):
                            currErr = self.evalFeature((featureType, x, y, w, h))
                            if currErr <= lowestErr and self.myFeatures.unused((featureType, x, y, w, h)):
                                lowestErr = currErr
                                bestFeature = (featureType, x, y, w, h)


        beta = lowestErr / (1 - lowestErr)

        return feature(bestFeature, beta)

    def updateWeights(self, feature):

        featureType, x, y, w, h =feature.myFeature

        error = self.evalFeature((featureType, x, y, w, h))
        self.setCorrectID(feature)

        beta = error/(1-error)

        for intImg in self.intImgList:
            intImg.weight = intImg.weight * (beta**(1-int(not intImg.correctID)))


class cascade:

    def __init__(self, fileName):

        self.intImgList = integralImageList(fileName)
        self.intImgList.initializeWeights()
        self.roundLimit = []

    def setFeatureAvailable(self):

        for intImg in self.intImgList.intImgList:
            if intImg.available == False:
                continue

            found = self.evalImg(intImg)

            if not found and not intImg.isPos:
                intImg.available = False

    def fd(self):

        f = 1.0
        d = 1.0

        for stage in self.intImgList.myFeatures.myFeatures:
            if len(stage)==0:
                continue

            totalNeg = 0.0
            falsePos = 0.0
            totalPos = 0.0
            correct = 0.0

            for intImg in self.intImgList.intImgList:
                if intImg.available == False:
                    continue

                sum = 0.0
                alphaSum = 0.0
                found = False

                for feature in stage:
                    sum += feature.alpha * intImg.featureIsFound(feature.myFeature)
                    alphaSum += feature.threshold * feature.alpha

                if sum >= alphaSum:
                    found = True


                if intImg.isPos:
                    totalPos += 1
                    if found==True:
                        correct += 1
                else:
                    totalNeg += 1
                    if found == True:
                        falsePos += 1


            f *= falsePos/totalNeg

            d *= correct/totalPos

        print "Fals Positive: {}".format(f)
        print "True Positive: {}".format(d)

        return (f,d)

    def evalImg(self, integralImage):

        for stage in self.intImgList.myFeatures.myFeatures:
            sum = 0.0
            alphaSum = 0.0
            for feature in stage:
                sum += feature.alpha * integralImage.featureIsFound(feature.myFeature)
                alphaSum += feature.threshold * feature.alpha

            if sum <= alphaSum:
                return False

        return True

    def getNextFeature(self):
        feature = self.intImgList.findBestFeature()
        self.intImgList.updateWeights(feature)
        self.intImgList.normalizeWeights()
        return feature

    def createCascade(self, ftarget, fMax, dMin, stageLim):

        stage = 0
        fOld = 1.0
        dOld = 1.0
        fNew = 1.0
        roundLimit = [5,10,10,20,20,25,30]

        while fNew > ftarget:
            print "**********************************"
            print "ROUND {}".format(stage)
            print "**********************************"
            n = 0
            fNew = fOld

            currClassifier = []

            roundStart = time.clock()

            self.intImgList.initializeWeights()
            featureCount = 0
            while fNew > fOld * fMax:
                start = time.clock()
                print "____________"
                print "Feature {}".format(featureCount)
                print "____________"


                currClassifier.append(self.getNextFeature())
                self.intImgList.myFeatures.addFeature(currClassifier, stage)

                if len(currClassifier) <= 1:
                    currClassifier.append(self.intImgList.findBestFeature())
                    self.intImgList.myFeatures.addFeature(currClassifier, stage)

                fNew, dNew = self.fd()

                while dNew < dMin * dOld:
                    self.intImgList.myFeatures.adjustThreshold(-0.001)
                    fNew, dNew = self.fd()


                # if fNew == 1.0 and dNew == 1.0:
                #     currClassifier.pop(featureCount)
                #     if featureCount == 0:
                #         currClassifier.pop(featureCount)
                # else :
                #     featureCount += 1
                featureCount += 1
                print "Time For Feature: {}".format((time.clock()-start)/60.0)
                if (stage < 7 and featureCount == roundLimit[stage]):
                    break

            self.setFeatureAvailable()
            fOld=fNew
            dOld=dNew

            print "**********************************"
            print "ROUND COMPLETE {}".format(stage)
            print "Num Features: {}".format(featureCount)
            print "Time For Round: {}".format((time.clock() - roundStart)/60.0)
            print "**********************************"
            self.printCascade()
            print "**********************************"
            print "**********************************"

            if stage == stageLim:
                return

            stage += 1

        print "Ended Goal Achieved"

    def saveCascade(self, fileName):

        open(fileName, "w").close

        myFile = open(fileName, "a+")

        for stage in self.intImgList.myFeatures.myFeatures:

            myFile.write("stage\n")
            for feature in stage:
                t, x, y, w, h = feature.myFeature
                beta = math.exp(-feature.alpha)
                threshold = feature.threshold
                myFile.write("{} {} {} {} {} {} {}\n".format(t, x, y, w, h, beta, threshold))
        myFile.close()

    def readCascade(self, fileName):
        myFile = open(fileName, "r+")
        currCascade = []
        currStage = -1

        for line in myFile:
            if line == "stage\n":
                if len(currCascade) > 0:
                    self.intImgList.myFeatures.addFeature(currCascade, currStage)
                currCascade = []
                currStage += 1
            else:
                line = line.split()
                currCascade.append(feature((int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4])),
                                           float(line[5])))
                currCascade[len(currCascade)-1].threshold = float(line[6])

        self.intImgList.myFeatures.addFeature(currCascade, currStage)
        myFile.close()

    def printCascade(self):

        for stage in self.intImgList.myFeatures.myFeatures:
            for f in stage:
                print f.myFeature

            print "_________________"


class detector:

    def __init__(self, cascadeFile):

        self.cascade = featureSet()

        myFile = open(cascadeFile, "r+")
        currCascade = []
        currStage = -1

        for line in myFile:
            if line == "stage\n":
                if len(currCascade) > 0:
                    self.cascade.addFeature(currCascade, currStage)
                currCascade = []
                currStage += 1
            else:
                line = line.split()
                currCascade.append(feature((int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4])),
                                           float(line[5])))
                currCascade[len(currCascade) - 1].threshold = float(line[6])

        self.cascade.addFeature(currCascade, currStage)
        myFile.close()

    def evalImage(self, integralImage, scale, xpos, ypos):

        for stage in self.cascade.myFeatures:
            sum = 0.0
            alphaSum = 0.0

            for feature in stage:
                type,x,y,w,h = feature.myFeature
                currFeature = (type, int(x*scale+xpos),int(y*scale+ypos),int(w*scale),int(h*scale))
                # print currFeature
                sum += feature.alpha * integralImage.featureIsFound(currFeature)
                # print integralImage.featureIsFound(currFeature)
                alphaSum += feature.threshold * feature.alpha

            if sum < alphaSum:
                return False

        return True

    def detectFace(self, image):


        faceLocations = []

        resize = 2
        scale = 1
        winSize = 24*scale
        stages = 0
        delta = 1
        height, width = image.shape
        start = time.clock()

        currIntImg = integralImage(image, height, width)

        print "Time For Int Image: {}".format(time.clock()-start)

        start = time.clock()
        while width >= winSize and height >= winSize:
            for x in range(0, width - winSize +1, delta):
                for y in range(0, height - winSize+1, delta):
                    #print "x: {}, y: {} of w: {}, h: {}".format(x, y, width, height)
                    containsFace = self.evalImage(currIntImg, scale, x, y)
                    if containsFace:
                        faceLocations.append((x, y, winSize, winSize))



            stages += 1
            winSize = int (winSize * resize)
            scale = scale*resize
            # delta = int(scale)


        print "time to check for face: {}".format(time.clock()-start)
        return faceLocations


def timeFunctions(fileName):
    start = time.clock()

    PosArr = integralImageList(fileName)

    print "Time for Creating intImgList : {}".format(time.clock() - start)

    start = time.clock()

    PosArr.initializeWeights()

    print "Time for InnitializeWeights : {}".format(time.clock() - start)

    start = time.clock()

    PosArr.normalizeWeights()
    print "Time for NormalizeWeights : {}".format(time.clock() - start)

    start = time.clock()

    f = PosArr.findBestFeature()

    print "Time for findBestFeature: {}".format(time.clock() - start)

    start = time.clock()

    PosArr.setCorrectID(f)

    print "Time for setCorrectID: {}".format(time.clock() - start)

    start = time.clock()

    numFeatures = 178760
    totalTime = 0

    bestFeature = ((0,0,0, 3, 3))
    lowestErr = PosArr.evalFeature((0,0,0, 3, 3))

    for w in range(3, 24):
        for h in range(3, 24):
            for x in range (24-w):
                for y in range (24 - h):
                    for featureType in range(11):
                        PosArr.evalFeature((featureType,x, y, w, h))


    print "Time for Average Feature Evaluation over Full Set: {}".format((time.clock() - start)/numFeatures)

    start = time.clock()

    PosArr.updateWeights(f)

    print "Time for updateWeights: {}".format(time.clock() - start)
    #
    # start = time.clock()
    # PosArr.myFeatures.myFeatures[0] = []
    # PosArr.createStrongClassifier(2)
    # print "Time for Create Strong Classifier with 2 weak Classifiers: {}".format(time.clock() - start)
    #
    #
    # start = time.clock()
    #
    # "Time for updateWeights: {}".format(time.clock() - start)

    t,x,y,w,h = f.myFeature
    print "Best Feature Found: ({}, {}, {})".format(t,x,y, w, h)

    print "Total Number of Images Used: {}".format(len(PosArr.intImgList))

    print "Total number of Features checked: {}".format(numFeatures)


# newCascade = cascade("face.txt")
#
# newCascade.createCascade(0.005, 0.5, 0.98, 10)
#
# newCascade.saveCascade("myCascade.txt")
# #

# timeFunctions("imgSet.txt")
# Time for Creating intImgList : 8.13775648898
# Time for InnitializeWeights : 0.000972773407804
# Time for NormalizeWeights : 0.00167884411444
# 178760
# Time for findBestFeature: 7407.70003611
# Time for setCorrectID: 0.0298710976458
#
# intImgList = integralImageList("imgSet.txt")
#
# correct = 0.0
# falsePos = 0.0
# falseNeg = 0.0
# total = 0.0
#
# test = detector("bananaCascade.txt")
#
# doc = open("banana.txt")
#
# for line in doc:
#
#     total += 1
#
#     line = line.split()
#
#     image = cv2.imread(line[0],0)
#
#
#     found = test.detectFace(image)
#
#     line[3]="0"
#     if (int(line[3]) == 1 and len(found)>0) or (int(line[3]) == 0 and len(found)==0):
#         correct += 1
#
#     if int(line[3]) == 0 and len(found)>0:
#         falsePos += 1
#
#     if int(line[3]) == 1 and len(found)==0:
#         falseNeg += 1
#
# print "\n\n\tCascade Accuracy"
# print "Total Images: {}".format(total)
# print "Percentage Correct: {}".format(correct*100/total)
# print "percentage False Positive: {}".format(falsePos*100/total)
# print "percentage False Negative: {}".format(falseNeg*100/total)


# image = cv2.imread("20.jpg",0)
# test = detector("myCascade.txt")
# myBanana = test.detectFace(image)
#
# print "Length: {}".format(len(myBanana))
#
# for (x, y, w, h) in myBanana:
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 1)
# cv2.imwrite("face.jpg",image)

# detectTest = detector("trained150_200.txt")
# count = 0
# for i in range(0, 2041):
#     image = cv2.imread("imgSet/_{}.jpeg".format(i), 0)
#
#
#     rows, cols= image.shape
#
#     face = detectTest.detectFace(image)
#
#
#     for (x, y, w, h) in face:
#             cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 1)
#
#     if len(face):
#         count += 1
#
# print count
# cv2.imwrite("withRect.jpeg", image)



# file = open("myPicsPos.txt", "r+")
#
# for line in file:
#     line = line.split()
#     image = cv2.imread(line[0], 0)
#     N=2
#     height, width = image.shape
#     image = cv2.resize(image, (width/N,height/N))
#     face = detectTest.detectFace(image)
#
#     print "line"
#     if len(face):
#         correct += 1
#     total += 1
#
#     for (x, y, w, h) in face:
#         x *= N
#         y *= N
#         w *= N
#         h *= N
#         cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)
#
#     cv2.imshow('image', image)
#     cv2.waitKey(1000)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# print correct/total




detectTest = detector("trained100Imgs.txt")
# detectTest = detector("myCascade.txt")
cap = cv2.VideoCapture(0)

N=7

while(True):

    print "Entered"
        # Capture frame-by-frame, ret is boolian, false if frame
        # not read correctly
    ret, frame = cap.read()

    temp  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows, cols = temp.shape

    gray=cv2.resize(temp,(cols/N,rows/N))

    faces = detectTest.detectFace(gray)

    for (x, y, w, h) in faces:
        x *= N
        y *= N
        w *= N
        h *= N
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

    cv2.imshow('frame', frame)
    # cv2.imshow('frame',cv2.flip(gray,1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# myCascade = cascade("pos.txt")
#
#
# ftarget = 0.1
# fmax = 0.50
# dMin = 0.98
# stageLim = 1
# myCascade.createCascade(ftarget, fmax, dMin, stageLim)
#
# myCascade.saveCascade("myCascade.txt")
#
#
# myCascade.printCascade()


#test it on the full Set

# test = cascade("banana.txt")
#
# test.readCascade("bananaCascade.txt")
#
# test.printCascade()
#
#
#
# falsePos = 0.0
# falseNeg = 0.0
# totalP = 0.0
# totalN = 0.0
# correctP = 0.0
# correctN = 0.0
# total =0
# for intImg in test.intImgList.intImgList:
#     total +=1
#     if intImg.isPos == True:
#         totalP +=1
#     else:
#         totalN+=1
#
#     found = test.evalImg(intImg)
#
#     if intImg.isPos == True and found:
#         correctP += 1
#
#     if intImg.isPos == False and not found:
#         correctN += 1
#
#     if intImg.isPos == False and found:
#         falsePos += 1
#
#     if intImg.isPos == True and not found:
#         falseNeg += 1
#
#
#
# print "\n\n\tCascade Accuracy"
# print "Total Images: {}".format(totalP+totalN)
# print "Percentage Pos Correct: {}".format((correctP)*100/(totalP))
# print "Percentage Neg Correct: {}".format(correctN*100/totalN)
# print "percentage False Positive: {}".format(falsePos*100/totalN)
# print "percentage False Negative: {}".format(falseNeg*100/totalP)
# print total
