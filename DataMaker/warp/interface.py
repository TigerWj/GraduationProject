import numpy as np
import cv2 as cv
import copy

from img_warp_utils import verticalScaleWarps
from util import getIntersection, getParaScatters, getPointAcrodLine, clusterPoint
from img_transform_utils import mls_rigid_deformation


def verticalScaleBody(img, top, down, alpha=1.5):
    """
    @description: 垂直缩放由[top, down]内的区域
    @param:
        img: 输入的图片
        top: int, pixel position in vertical direction
        down: int, pixel position in vertical direction
        alpha: float, 缩放系数
    @Returns: warped image
    """
    
    return verticalScaleWarps(img, top, down, alpha)

def warpBodyPartAcrodSil(img, silimg, keyPoints, partIndex, pointSet, corner=None, alpha=0):
    """
    @description: 对某一肢体进行形变
    @param: 
        img: 输入图片
        silimg: 轮廓图片
        keyPoints: (23, 2), 人体关键点
        pointSet: [[]], 定义由各个关键点组成的 body part
        partIndex: [], 需要进行形变的 body part 索引, 索引对应 
        corner: [x1, y1, x2, y2], 定义一个区域, 对区域内的图像,利用高斯滤波器进行平滑处理
        alpha: float, 形变程度系数
    @Returns: warped image
    """

    gray = cv.cvtColor(silimg, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    sourcePoints = []
    destiPoints = []

    if corner!=None:
        sourcePoints = [[corner[0], corner[1]],[corner[0], corner[3]], [corner[2], corner[1]],  [corner[2], corner[3]]]
        destiPoints = [[corner[0], corner[1]],[corner[0], corner[3]], [corner[2], corner[1]],  [corner[2], corner[3]]]

    for l in partIndex:
        s, d = getIntersection(np.array(pointSet[l][0]), np.array(pointSet[l][1]), binary, leftFlag=True, rightFlag=True, ratioNum=3, ratioIndex=[2,0], gap=2, alpha=alpha)
        realIndex, _ = clusterPoint(np.array(d), pointSet, np.array([l]))  # 聚类 筛除异常点
        sourcePoints += np.array(d)[realIndex].tolist()
        destiPoints += np.array(s)[realIndex].tolist()

   
    visualise(silimg, sourcePoints, destiPoints, np.array(keyPoints))
    
    # 高斯模糊
    if corner!=None:
        img = mls_rigid_deformation(img, np.array(sourcePoints), np.array(destiPoints))
        warpedImg = img[corner[1]:corner[3], corner[0]:corner[2]]
        warpedImg = cv.GaussianBlur(warpedImg, (9, 9), 0.6)
        img[corner[1]:corner[3], corner[0]:corner[2]] = warpedImg
    else:
        img = mls_rigid_deformation(img, np.array(sourcePoints), np.array(destiPoints))

    return img

def warpBodyPartAcrodWidth(img, keyPoints, partIndex, alpha, ratioNum, width, weight=None):
    sourcePoints = []
    destiPoints = []

    for l in partIndex:
        d1, d2 = getParaScatters(np.array(keyPoints[l[0]]), np.array(keyPoints[l[1]]), ratioNum, 3*width, weight)
        d1.append(keyPoints[l[0]])
        d1.append(keyPoints[l[1]])
        d2.append(keyPoints[l[0]])
        d2.append(keyPoints[l[1]])
        destiPoints += d1 + d2
    
    for l in partIndex:
        s1, s2 = getParaScatters(np.array(keyPoints[l[0]]), np.array(keyPoints[l[1]]), ratioNum, 5*width)
        s1.append(keyPoints[l[0]])
        s1.append(keyPoints[l[1]])
        s2.append(keyPoints[l[0]])
        s2.append(keyPoints[l[1]])
        sourcePoints += s1 + s2
    
    # 人体关键点作为不变的标定点，为保持整张图片
    inputImg = copy.deepcopy(img)
    warpedImg = mls_rigid_deformation(inputImg, np.array(sourcePoints), np.array(destiPoints))

    visualise(img, sourcePoints, destiPoints)

    return warpedImg

def freeTransform(img, center, radius, orient):
    warpedImg = LocalTranslationWarps(img, center, radius, orient)

    return warpedImg


def jianKuan(img, keyPoints, partIndex, pointSet, alpha):
    sourcePoints = []
    destiPoints = []

    sourcePoints.append(pointSet[partIndex][0])
    sourcePoints.append(pointSet[partIndex][1])
    destiPoints.append(pointSet[partIndex][0])
    destiPoints.append(getPointAcrodLine(np.array(pointSet[partIndex][0]), np.array(pointSet[partIndex][1]), img.shape[1], img.shape[0], 0.3))
    inputImg = copy.deepcopy(img)
    warpedImg = mls_rigid_deformation(inputImg, np.array(sourcePoints), np.array(destiPoints))

    visualise(img, sourcePoints, destiPoints, keyPoints)

    return warpedImg


# def tianEJin(img, keyPoints, partIndex, alpha):
#     pass


def visualise(img, sourcePoints, destiPoints, keyPoints):
    # 可视化
    cv.namedWindow("visualise")
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