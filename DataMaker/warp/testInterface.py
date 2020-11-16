import os
import numpy as np
import cv2 as cv
import copy
import sys

from interface import *
from util import loadKeypoints

def testVerticalScaleBody():
    pass

corner = []
def onMouse(event, x, y, flags, param):
    global corner
    if event == cv.EVENT_LBUTTONDOWN:
        corner += [x, y]
        print(corner)

def testWarpBodyPart(imageDir, silImageDir, keyPointsFileDir):
    global corner
    img = cv.imread(imageDir)
    cv.namedWindow("source image")
    cv.imshow("source image", img)
    cv.setMouseCallback("source image", onMouse)
    cv.waitKey(0)

    silImg = cv.imread(silImageDir)
    keyPoints = loadKeypoints(keyPointsFileDir)[:, :2]

    pointSet = []
    pointSet.append([keyPoints[0], keyPoints[1]]) # 0
    pointSet.append([keyPoints[2], keyPoints[3]]) # 1
    pointSet.append([keyPoints[3], keyPoints[4]]) # 2
    pointSet.append([keyPoints[2], keyPoints[1]]) # 3
    pointSet.append([keyPoints[1], keyPoints[5]]) # 4
    pointSet.append([keyPoints[5], keyPoints[6]]) # 5
    pointSet.append([keyPoints[6], keyPoints[7]]) # 6
    pointSet.append([keyPoints[2], keyPoints[9]]) # 7
    pointSet.append([keyPoints[5], keyPoints[12]]) # 8
    pointSet.append([keyPoints[9], keyPoints[10]]) # 9
    pointSet.append([keyPoints[10], keyPoints[11]]) # 10
    pointSet.append([keyPoints[12], keyPoints[13]]) # 11
    pointSet.append([keyPoints[13], keyPoints[14]]) # 12

    inputImg = copy.deepcopy(img)
    partIndex = [1, 2, 5, 6, 9, 10, 11, 12]
    
    if len(corner)==0: corner=None
    warpedImg = warpBodyPartAcrodSil(inputImg, silImg, keyPoints[:, :2].tolist(), partIndex, pointSet,corner, alpha=1.2)

    cv.namedWindow("warped image")
    cv.imshow("warped image", warpedImg)

    cv.waitKey(0)
    cv.destroyAllWindows()

def testWarpBodyPartAcrodWidth(imageDir, keyPointsFileDir):
    img = cv.imread(imageDir)
    cv.namedWindow("source image")
    cv.imshow("source image", img)

    keyPoints = loadKeypoints(keyPointsFileDir)

    input = copy.deepcopy(img)
    inputIndex = [[1, 8]]
    weight = [5, 3, 5]
    warpedImg = warpBodyPartAcrodWidth(input, keyPoints[:, :2].tolist(), inputIndex, 0.9, 1, 20)

    cv.namedWindow("warped image")
    cv.imshow("warped image", warpedImg)

    cv.waitKey(0)
    cv.destroyAllWindows()

def testJianKuan(imageDir, keyPointsFileDir):
    img = cv.imread(imageDir)
    cv.namedWindow("source image")
    cv.imshow("source image", img)

    keyPoints = loadKeypoints(keyPointsFileDir)[:, :2]
    pointSet = []
    pointSet.append([keyPoints[0], keyPoints[1]]) # 0
    pointSet.append([keyPoints[2], keyPoints[3]]) # 1
    pointSet.append([keyPoints[3], keyPoints[4]]) # 2
    pointSet.append([keyPoints[1], keyPoints[2]]) # 3
    pointSet.append([keyPoints[1], keyPoints[5]]) # 4
    pointSet.append([keyPoints[5], keyPoints[6]]) # 5
    pointSet.append([keyPoints[6], keyPoints[7]]) # 6
    pointSet.append([keyPoints[2], keyPoints[9]]) # 7
    pointSet.append([keyPoints[5], keyPoints[12]]) # 8
    pointSet.append([keyPoints[9], keyPoints[10]]) # 9
    pointSet.append([keyPoints[10], keyPoints[11]]) # 10
    pointSet.append([keyPoints[12], keyPoints[13]]) # 11
    pointSet.append([keyPoints[13], keyPoints[14]]) # 12

    inputImg = copy.deepcopy(img)
    inputIndex = [3]
    warpedImg = jianKuan(inputImg, keyPoints.tolist(), 3, pointSet, 0.3)

    cv.namedWindow("warped image")
    cv.imshow("warped image", warpedImg)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    print(sys.argv)
    imgDir = "/home/wj/workspace/GraduationProject/DataMaker/testData/images/{}.jpg".format(sys.argv[1])
    silDir = "/home/wj/workspace/GraduationProject/DataMaker/testData/u2net_results/{}.png".format(sys.argv[1])
    keypointDir = "/home/wj/workspace/GraduationProject/DataMaker/testData/openpose_keypoints/{}_keypoints.json".format(sys.argv[1])
    
    testWarpBodyPart(imgDir, silDir, keypointDir)
    # testWarpBodyPartAcrodWidth(imgDir, keypointDir)
    #testJianKuan(imgDir, keypointDir)
