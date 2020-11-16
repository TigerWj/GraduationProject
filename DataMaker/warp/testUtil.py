import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import json
import logging
import sys
import copy

from util import getParaScatters, getIntersection, loadKeypoints, clusterPoint, processAfterCluster
from img_transform_utils import mls_rigid_deformation


logging.basicConfig(level=logging.DEBUG)

def showKeyPointInOutLine(imageDir, keyPointsFileDir):
    img= cv.imread(imageDir)

    keyPoints = loadKeypoints(keyPointsFileDir)
    for i in range(keyPoints.shape[0]):
        cv.drawMarker(img, (int(keyPoints[i][0]), int(keyPoints[i][1])), (0, 0, 255), markerType=cv.MARKER_STAR ,markerSize=5)

    cv.namedWindow("image")
    cv.imshow("image", img)
    cv.waitKey(0)


def getParaScattersTest(imageDir, keyPointsFileDir):
    img= cv.imread(imageDir)

    keyPoints = loadKeypoints(keyPointsFileDir)
    res1, res2 = getParaScatters(keyPoints[2, :2], keyPoints[3, :2], 5, 5)

    for i in range(keyPoints.shape[0]):
        cv.drawMarker(img, (int(keyPoints[i][0]), int(keyPoints[i][1])), (255, 0, 0), markerType=cv.MARKER_STAR ,markerSize=5)
    
    for i in range(res1.shape[0]):
        cv.drawMarker(img, (int(res1[i][0]), int(res1[i][1])), (0, 0, 255), markerType=cv.MARKER_STAR ,markerSize=3)
        cv.drawMarker(img, (int(res2[i][0]), int(res2[i][1])), (0, 0, 255), markerType=cv.MARKER_STAR ,markerSize=3)
        
    cv.namedWindow("image")
    cv.imshow("image", img)
    cv.waitKey(0)

def getIntersectionTest(imageDir, keyPointsFileDir):
    img= cv.imread(imageDir)

    keyPoints = loadKeypoints(keyPointsFileDir)
    # logging.info("keypoints: {}".format(keyPoints))

    # 左手臂
    p1 = copy.deepcopy(keyPoints[1, :2])
    p2 = copy.deepcopy(keyPoints[8, :2])

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    cv.namedWindow("image")
    cv.imshow("image", binary)
    cv.waitKey(0)

    resPoints, disPoints = getIntersection(p1, p2, binary, leftFlag=True, rightFlag=True, ratioNum=5, ratioIndex=[4, 3], alpha=1.5)
    
    for i in range(len(resPoints)):
        l = resPoints[i]
        cv.drawMarker(img, (int(l[0]), int(l[1])), (0, 0, 255), markerType=cv.MARKER_STAR ,markerSize=5)
        
    for i in range(len(resPoints)):
        l = disPoints[i]
        cv.drawMarker(img, (int(l[0]), int(l[1])), (255, 0, 255), markerType=cv.MARKER_STAR ,markerSize=5)

    for i in range(keyPoints.shape[0]):
        cv.drawMarker(img, (int(keyPoints[i][0]), int(keyPoints[i][1])), (255, 0, 0), markerType=cv.MARKER_STAR ,markerSize=5)

    cv.imshow("image", img)
    cv.waitKey(0)

def clusterPointTest(imgDir, silDir, keyPointsFileDir):
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

    contours, category, realContours, targetContours = clusterPoint(silImg, pointSet, set([1, 2, 5, 6]), 0.8)
    afterProcessRes = processAfterCluster(realContours, targetContours, set([1, 2, 5, 6]), pointSet, 50, sampleRate=0.2)

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
    T = 0

    for _, v in afterProcessRes.items():
        sourcePoints 
        l ,r ,t ,d = 1000, 0, 1000, 0
        for i in range(len(v)):             
            sp= realContours[v[i]]
            l = min(l, round(np.min(sourcePoints[:, 0])))
            r = max(r, round(np.max(sourcePoints[:, 0])))
            t = min(t, round(np.min(sourcePoints[:, 1])))
            d = max(d, round(np.max(sourcePoints[:, 1])))

    l = l - T
    r = r + T
    t = t - T
    d = d + T

    warpImg = img[t:d, l:r]
    warpImg = mls_rigid_deformation(warpImg, np.concatenate([keyPoints, realContours[sourceIndex, :2]], axis=0) - np.array([l, t]), np.concatenate([keyPoints, targetContours[sourceIndex, :2]], axis=0) - np.array([l, t]))
    img[t:d, l:r] = warpImg

    cv.drawMarker(img, (l, t), (255, 0, 0), markerType=cv.MARKER_STAR ,markerSize=3)
    cv.drawMarker(img, (r, d), (255, 0, 0), markerType=cv.MARKER_STAR ,markerSize=3)        

    # realTarget = targetContours[sourceIndex]
    # for i in range(realTarget.shape[0]):
    #     x = round(realTarget[i][0]) 
    #     y = round(realTarget[i][1])
    #     warpedImg[y][x] = tempImg[y][x] 

    cv.namedWindow("image4")
    cv.imshow("image4", img)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    print(sys.argv)
    imgDir = "/home/wj/workspace/GraduationProject/DataMaker/testData/images/{}.jpg".format(sys.argv[1])
    silDir = "/home/wj/workspace/GraduationProject/DataMaker/testData/u2net_results/{}.png".format(sys.argv[1])
    keyPointDir = "/home/wj/workspace/GraduationProject/DataMaker/testData/openpose_keypoints/{}_keypoints.json".format(sys.argv[1])
    #showKeyPointInOutLine(sys.argv[1], sys.argv[2])
    getIntersectionTest(silDir, keyPointDir)
    # clusterPointTest(imgDir, silDir, keyPointDir)
    # getParaScattersTest(sys.argv[1], sys.argv[2])