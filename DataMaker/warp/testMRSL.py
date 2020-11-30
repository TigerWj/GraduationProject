import numpy as np
from img_mrsl import *
import cv2
import copy
import sys

step = 10
sigma = 5
lambdaPara = 10
alpha = 1


def MRSLTest():

    w = 578
    h = 770
    sourcePoints = np.array([[233, 396], [234, 432], []])
    destPoints = np.array([[[215,433]], [435,421]])
    MRSL(w, h, sourcePoints, destPoints)

def MRSLImageTest():
    imgdir = "/home/wj/workspace/GraduationProject/DataMaker/testData/images/5.jpg"
    image = cv2.imread(imgdir)
    sourcePoints = np.array([[233, 396], [234, 432], [241, 483], [433, 373], [432, 420]])
    destPoints = np.array([[259, 399], [264, 423], [260, 463], [390, 376], [395, 425]])

    warpImage = MRSL(image, sourcePoints, destPoints)
    cv2.imwrite("./temp.jpg", warpImage)


def getGridTest():
    step = 10
    h = 301
    w = 201

    import ipdb; ipdb.set_trace()
    sp, grid= getGrid(step,  h, w)

def norm_ind1Test():
    
    x = np.array([[1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,101.0000,201.0000,301.0000,401.0000,501.0000,101.0000,201.0000,301.0000,401.0000,501.0000,578.0000
    ,578.0000,578.0000,578.0000,578.0000,578.0000,578.0000,578.0000,578.0000,234.8395,415.4568,424.963], [1.0000,101.0000,201.0000,301.0000,401.0000,501.0000,601.0000,701.0000,770.0000,1.0000,1.0000,1.0000,1.0000,1.0000,770.0000,770.0000,770.0000,770.0000,770.0000,1.0000
    ,101.0000,201.0000,301.0000,401.0000,501.0000,601.0000,701.0000,770.0000,432.2387,421.1481,508.2881]])

    import ipdb; ipdb.set_trace()
    p, ngrid = norm_ind1(x.T)

def con_kTest():

    x = np.array([[-1.3419, -0.1753], [0.6437, -0.2660], [0.7072, 0.4413]])
    y =np.array([[-1.3419, -0.1753], [0.6437, -0.2660], [0.7072, 0.4413]])

    import ipdb; ipdb.set_trace()
    k = con_k(x, y, sigma)

image = None
p = []
q = []

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        p.append([x, y])
        cv2.drawMarker(image, (x, y), (0,0,0), markerSize=10)
    elif event == cv2.EVENT_MBUTTONDOWN:
        q.append([x, y])
        cv2.drawMarker(image, (x, y), (255,255,255), markerSize=10)

    cv2.imshow("image", image)

def MRSLImageTestPlus(imgdir):
    global image
    image = cv2.imread(imgdir)
    tempImage = copy.deepcopy(image)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", onMouse)
    cv2.imshow("image", image)

    warpImage = None
    if cv2.waitKey(0):
        warpImage = MRSL(tempImage, np.array(p), np.array(q))
        cv2.imwrite("./temp.jpg", warpImage)

    cv2.namedWindow("warped image")
    cv2.imshow("warped image", warpImage)

    if cv2.waitKey(0):
        return

    

if __name__ == "__main__":
    # getGridTest()
    # norm_ind1Test()
    # con_kTest()
    # MRSLTest()
    # MRSLImageTest()

    print(sys.argv)
    imgDir = "/home/wj/workspace/GraduationProject/DataMaker/testData/images/{}.jpg".format(sys.argv[1])
    MRSLImageTestPlus(imgDir)