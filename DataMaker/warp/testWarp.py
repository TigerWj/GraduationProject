import numpy as np
import cv2 as cv
from img_warp_utils import *
import sys
from util import *

img = None
start = None
end = None
mode = 1

def testLocalScalingWarps(imgDir):
    img = cv.imread(imgDir)
    cv.imshow("image", img)
    out = LocalScalingWarps(img, np.array([68, 88]), np.array([91, 75]), 10)
    cv.imshow("out", out)

    if cv.waitKey(0):
        return

    return

def testLocalTranslationWarps(imgDir):
    img = cv.imread(imgDir)
    cv.imshow("image", img)
    out = LocalTranslationWarps(img, np.array([233, 63]), 200, np.array([195, 79]))
    cv.imshow("out", out)

    if cv.waitKey(0):
        return

def testVerticalScaleWarps():
    global img
    cv.namedWindow("image")
    cv.setMouseCallback("image", onMouseVerticalScaleWarps)
    cv.imshow("image", img)

    if cv.waitKey(0):
        return

def onMouseWarps(event, x, y, flags, param):
    global img, start, end, mode

    if event == cv.EVENT_LBUTTONDOWN:
        start = np.array([x, y])

    elif event == cv.EVENT_LBUTTONUP:
        end = np.array([x, y])
        if mode == 1:
            img = LocalTranslationWarps(img, start, int(pointDistance(start[0], start[1], end[0], end[1])), end)
        else:
            img = verticalScaleWarps(img, start[1], end[1], 1.3)

    elif event == cv.EVENT_LBUTTONDBLCLK:
        start = np.array([x, y])
        end = np.array([x+10, y+10])
        img = LocalScalingWarps(img, start, end, 0.9)

    cv.imshow("image", img)

def onMouseWarpsRectangle(event, x, y, flags, param):
    global img, start, end, mode

    if event == cv.EVENT_LBUTTONDOWN:
        start = np.array([x, y])

    elif event == cv.EVENT_LBUTTONUP:
        end = np.array([x, y])
        img = LocalRetangleWarps(img, [370, 469, 270, 300], 290, 30, 320)
    cv.imshow("image", img)
    

def testWarps():
    global img, mode
    cv.namedWindow("image")
    cv.setMouseCallback("image", onMouseWarps)
    cv.imshow("image", img)

    while(1):
        k = cv.waitKey(0)
        
        if k == ord("c"):
            mode = ~mode
        else:
            break
    
    cv.destroyAllWindows()



def multiTest(imgDir, silDir, keyPointsFileDir):
    silImg = cv.imread(silDir)
    keyPoints = loadKeypoints(keyPointsFileDir)[:, :2]

    pointSet = []
    pointSet.append([keyPoints[0], keyPoints[1]])
    pointSet.append([keyPoints[2], keyPoints[3]])
    pointSet.append([keyPoints[3], keyPoints[4]])
    pointSet.append([keyPoints[2], keyPoints[1]])
    pointSet.append([keyPoints[1], keyPoints[5]])
    pointSet.append([keyPoints[5], keyPoints[6]])
    pointSet.append([keyPoints[6], keyPoints[7]])
    pointSet.append([keyPoints[9], keyPoints[8]])
    pointSet.append([keyPoints[8], keyPoints[12]])
    pointSet.append([keyPoints[9], keyPoints[10]])
    pointSet.append([keyPoints[10], keyPoints[11]])
    pointSet.append([keyPoints[12], keyPoints[13]])
    pointSet.append([keyPoints[13], keyPoints[14]])
    pointSet.append([keyPoints[2], keyPoints[9]])
    pointSet.append([keyPoints[5], keyPoints[12]])

    contours, category, realContours, targetContours = clusterPoint(silImg, pointSet, set([1, 2]), 0.1)
    afterProcessRes = processAfterCluster(realContours, targetContours, set([1, 2]), pointSet, 50, sampleRate=0.1)

    colorBar = []
    for i in range(len(pointSet)):
        randColor = np.random.randint(0, 255, size=[3], dtype=np.uint8).tolist()
        colorBar.append(randColor)

    cv.namedWindow("image1")
    cv.namedWindow("image2")
    cv.namedWindow("image2")
    img1 = copy.deepcopy(silImg)
    img2 = copy.deepcopy(silImg)
    img3 = copy.deepcopy(silImg)

    for i in range(keyPoints.shape[0]):
        cv.drawMarker(img1, (int(keyPoints[i][0]), int(keyPoints[i][1])), (255, 0, 0), markerType=cv.MARKER_STAR ,markerSize=3)
    for i in range(keyPoints.shape[0]):
        cv.drawMarker(img2, (int(keyPoints[i][0]), int(keyPoints[i][1])), (255, 0, 0), markerType=cv.MARKER_STAR ,markerSize=3)
    for i in range(keyPoints.shape[0]):
        cv.drawMarker(img3, (int(keyPoints[i][0]), int(keyPoints[i][1])), (255, 0, 0), markerType=cv.MARKER_STAR ,markerSize=3)


    for idx, c in enumerate(contours):
        cv.drawMarker(img1, (round(c[0]), round(c[1])), colorBar[category[idx]], markerType=cv.MARKER_STAR ,markerSize=3)
    
    for c in realContours:
        cv.drawMarker(img2, (round(c[0]), round(c[1])), colorBar[round(c[2])], markerType=cv.MARKER_STAR ,markerSize=3)

    for c in targetContours:
        cv.drawMarker(img2, (round(c[0]), round(c[1])), (255, 0, 0), markerType=cv.MARKER_STAR ,markerSize=3)
    
    sourceIndex = []

    for _, v in afterProcessRes.items():
        for i in range(len(v)):
            sourceIndex += v[i]
            for t in v[i]:
                print(keyPoints.shape, realContours.shape, len(v[i]))
                c1 = realContours[t]
                cv.drawMarker(img3, (round(c1[0]), round(c1[1])), colorBar[round(c1[2])], markerType=cv.MARKER_STAR ,markerSize=3)
                c2 = targetContours[t]
                cv.drawMarker(img3, (round(c2[0]), round(c2[1])), (255, 0, 0), markerType=cv.MARKER_STAR ,markerSize=3)


    cv.imshow("image1", img1)
    cv.imshow("image2", img2)
    cv.imshow("image3", img3)


    img = cv.imread(imgDir)
    points = realContours[sourceIndex, :2]

    
    for i in range(len(sourceIndex)):
        r = realContours[i, :2]
        r1 = np.array([int(r[0]), int(r[1])])
        t = targetContours[i]
        t1 = np.array([int(t[0]), int(t[1])])
        img = LocalTranslationWarps(img, r1, 5, t1)

   
    cv.imshow("image4", img)

    cv.waitKey(0)
    cv.destroyAllWindows()



if __name__ == "__main__":
    print(sys.argv)
    print("Press \"c\" to change mode, holding click and move to get tansformation ")
    imgDir = "/home/wj/workspace/GraduationProject/DataMaker/testData/images/{}.jpg".format(sys.argv[1])
    img = cv.imread(imgDir)
    testWarps()
    # imgDir = "/home/wj/workspace/GraduationProject/DataMaker/testData/images/{}.jpg".format(sys.argv[1])
    # silDir = "/home/wj/workspace/GraduationProject/DataMaker/testData/u2net_results/{}.png".format(sys.argv[1])
    # keyPointDir = "/home/wj/workspace/GraduationProject/DataMaker/testData/openpose_keypoints/{}_keypoints.json".format(sys.argv[1])
    # multiTest(imgDir, silDir, keyPointDir)


