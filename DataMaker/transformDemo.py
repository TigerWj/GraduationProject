import cv2 as cv
import os
import numpy as np
import sys
import logging
import random
import json

from MovingLeastSquares.img_utils_demo import demoOut
from MovingLeastSquares.img_utils import mls_similarity_deformation, mls_affine_deformation, mls_rigid_deformation

p = []
q = []
img = None

def onMouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        p.append([x, y])
        cv.drawMarker(img, (x, y), (0,0,0), markerSize=10)
    elif event == cv.EVENT_RBUTTONDOWN:
        q.append([x, y])
        cv.drawMarker(img, (x, y), (255,255,255), markerSize=10)

    cv.imshow("image", img)


def transform(imageDir, keyPointsFileDir):
    global img
    img= cv.imread(imageDir)

    keyPoints = loadKeypoints(keyPointsFileDir)
    for i in range(keyPoints.shape[0]):
        randColor = np.random.randint(0, 255, size=[3], dtype=np.uint8).tolist()
        cv.drawMarker(img, (int(keyPoints[i][0]), int(keyPoints[i][1])), randColor, markerType=cv.MARKER_STAR ,markerSize=5)

    cv.namedWindow("image")
    cv.setMouseCallback("image", onMouse)
    cv.imshow("image", img)

    if cv.waitKey(0):
        #demoOut(mls_affine_deformation, "Affine", q, p, imageDir)
        #demoOut(mls_similarity_deformation, "Similarity", q, p, imageDir)
        demoOut(mls_rigid_deformation, "Rigid", q, p, imageDir)


def loadKeypoints(jsonDir):
    with open(os.path.join(jsonDir), 'r') as f:
        people = json.load(f)["people"]

    if len(people) != 1:
        logging.error("the number of people is not 1")
        exit(1)

    keyPoints = np.array(people[0]['pose_keypoints_2d']).reshape((-1, 3))
    
    return keyPoints


if __name__ == "__main__":
    print(sys.argv)
    print("Manually label key points set--Source, and then label some other key points set--Destination in same order")
    transform(sys.argv[1], sys.argv[2])

