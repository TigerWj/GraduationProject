import sys
import cv2 as cv
import copy
import numpy as np

from warp.util import *
from warp.img_warp_utils import LocalTranslationWarps, verticalScaleWarps
from warp.img_transform_utils import mls_rigid_deformation

# 局部圆形区域移动变换
def demoLocalTranslationWarps(imageDir):
    img = cv.imread(imageDir)
    cv.imshow("LocalTranslationWarps", img)
    out = LocalTranslationWarps(img, np.array([233, 63]), 200, np.array([195, 79]))
    cv.imshow("out", out)

    if cv.waitKey(0):
        cv.destroyAllWindows()

# 局部垂直缩放
def demoVerticalScaleWarps(imageDir):
    img = cv.imread(imageDir)
    cv.imshow("VerticalScaleWarps", img)
    out = verticalScaleWarps(img, 100, 200, 1.5)
    cv.imshow("out", out)

    if cv.waitKey(0):
        cv.destroyAllWindows()

# 
def demoGetPoints(imageDir, silImageDir, keyPointsFileDir):

    img = cv.imread(imageDir)
    cv.namedWindow("source image")
    cv.imshow("source image", img)

    silImg = cv.imread(silImageDir)
    keyPoints = loadKeypoints(keyPointsFileDir)[:, :2]

    # 定义bodyPart类别，顺序索引值即为其类别
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

    # 需要获取其关键点的类别
    partIndex = [1, 2, 5, 6, 9, 10, 11, 12]

    # 二值化
    gray = cv.cvtColor(silImg, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    sourcePoints = []
    destiPoints = []

    for l in partIndex:
        s, d = getIntersection(np.array(pointSet[l][0]), np.array(pointSet[l][1]), binary, leftFlag=True, rightFlag=True, ratioNum=3, ratioIndex=[2,0], gap=2, alpha=1.2)
        # 聚类 筛除不属于该类别异常点
        realIndex, _ = clusterPoint(np.array(d), pointSet, np.array([l])) 
        sourcePoints += np.array(d)[realIndex].tolist()
        destiPoints += np.array(s)[realIndex].tolist()

   
    visualise(silImg, sourcePoints, destiPoints, np.array(keyPoints))

    # msl形变
    img = mls_rigid_deformation(img, np.array(sourcePoints), np.array(destiPoints))
    # img = cv.GaussianBlur(img, (9, 9), 0.6)

    cv.namedWindow("warped image")
    cv.imshow("warped image", img)
    cv.waitKey(0)

    # msl 反形变
    img = mls_rigid_deformation(img, np.array(destiPoints), np.array(sourcePoints))
    # img = cv.GaussianBlur(img, (9, 9), 0.6)

    cv.namedWindow("warped image return")
    cv.imshow("warped image return", img)
    cv.waitKey(0)

    cv.destroyAllWindows()

   
def visualise(img, sourcePoints, destiPoints, keyPoints):
    # 可视化
    cv.namedWindow("visualise keypoints")
    for i in range(len(sourcePoints)):
        l = sourcePoints[i]
        cv.drawMarker(img, (int(l[0]), int(l[1])), (0, 0, 255), markerType=cv.MARKER_STAR ,markerSize=5)
        
    for i in range(len(destiPoints)):
        l = destiPoints[i]
        cv.drawMarker(img, (int(l[0]), int(l[1])), (255, 0, 0), markerType=cv.MARKER_STAR ,markerSize=5)

    for i in range(len(keyPoints)):
        cv.drawMarker(img, (int(keyPoints[i][0]), int(keyPoints[i][1])), (0, 255, 0), markerType=cv.MARKER_STAR ,markerSize=3)
    cv.imshow("visualise", img)
    cv.waitKey(0)


if __name__ == "__main__":
    print(sys.argv)
    imgDir = "/home/wj/workspace/GraduationProject/DataMaker/testData/images/{}.jpg".format(sys.argv[1])
    silDir = "/home/wj/workspace/GraduationProject/DataMaker/testData/u2net_results/{}.png".format(sys.argv[1])
    keypointDir = "/home/wj/workspace/GraduationProject/DataMaker/testData/openpose_keypoints/{}_keypoints.json".format(sys.argv[1])
    demoLocalTranslationWarps(imgDir)
    demoVerticalScaleWarps(imgDir)
    demoGetPoints(imgDir, silDir, keypointDir)