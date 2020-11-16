import numpy as np
import cv2 as cv

from img_warp_utils import*
from util import *


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
    afterProcessRes = processAfterCluster(realContours, targetContours, set([1, 2]), pointSet, 50, sampleRate=1)

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
        cv.drawMarker(img2, (round(c[0]), round(c[1])), (255,0,0), markerType=cv.MARKER_STAR ,markerSize=3)
    

    sourceIndex = []
    for _, v in afterProcessRes.items():
        for i in range(len(v)):
            sourceIndex += v[i]
            for t in v[i]:
                c1 = realContours[t]
                cv.drawMarker(img3, (round(c1[0]), round(c1[1])), colorBar[round(c1[2])], markerType=cv.MARKER_STAR ,markerSize=3)
                c2 = targetContours[t]
                cv.drawMarker(img3, (round(c2[0]), round(c2[1])), (255, 0, 0), markerType=cv.MARKER_STAR ,markerSize=3)

    cv.imshow("image1", img1)
    cv.imshow("image2", img2)

    img = cv.imread(imgDir)
    transformInfo = xxx(img, realContours[:, :2], afterProcessRes, pointSet, alpha=60)
    print(transformInfo)
    
    for k, v in transformInfo.items():
        img = LocalTranslationWarps(img, np.array(v[1]), round(pointDistance(v[1][0], v[1][1], v[0][0], v[0][1])), np.array(v[0]))
        img = LocalTranslationWarps(img, np.array(v[4]), round(pointDistance(v[4][0], v[4][1], v[3][0], v[3][1])), np.array(v[3]))
        # cv.drawMarker(img, (round(v[0][0]), round(v[0][1])), (255, 12, 0), markerType=cv.MARKER_STAR ,markerSize=3)
        # cv.drawMarker(img, (round(v[1][0]), round(v[1][1])), (255, 120, 0), markerType=cv.MARKER_STAR ,markerSize=3)
        # cv.drawMarker(img, (round(v[3][0]), round(v[3][1])), (255, 12, 0), markerType=cv.MARKER_STAR ,markerSize=3)
        # cv.drawMarker(img, (round(v[4][0]), round(v[4][1])), (255, 120, 0), markerType=cv.MARKER_STAR ,markerSize=3)

    cv.imshow("image3", img3)

   
    cv.imshow("image4", img)

    cv.waitKey(0)
    cv.destroyAllWindows()



if __name__ == "__main__":
    print(sys.argv)

    imgDir = "/home/wj/workspace/GraduationProject/DataMaker/testData/images/{}.jpg".format(sys.argv[1])
    silDir = "/home/wj/workspace/GraduationProject/DataMaker/testData/u2net_results/{}.png".format(sys.argv[1])
    keyPointDir = "/home/wj/workspace/GraduationProject/DataMaker/testData/openpose_keypoints/{}_keypoints.json".format(sys.argv[1])
    multiTest(imgDir, silDir, keyPointDir)


