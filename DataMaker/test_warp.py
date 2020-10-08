import numpy as np
import cv2 as cv
import warp


def testLocalScalingWarps(imgDir):
    img = cv.imread(imgDir)
    cv.imshow("image", img)
    out = warp.LocalScalingWarps(img, np.array([68, 88]), np.array([91, 75]), 2)
    cv.imshow("out", out)

    if cv.waitKey(0):
        return

    return

def testLocalTranslationWarps(imgDir):
    img = cv.imread(imgDir)
    cv.imshow("image", img)
    out = warp.LocalTranslationWarps(img, np.array([233, 63]), 200, np.array([195, 79]))
    cv.imshow("out", out)

    if cv.waitKey(0):
        return

img = None
start = None
end = None

def onMouseWarps(event, x, y, flags, param):
    global img, start, end
    if event == cv.EVENT_LBUTTONDOWN:
        start = np.array([x, y])
    elif event == cv.EVENT_LBUTTONUP:
        end = np.array([x, y])
        img = warp.LocalTranslationWarps(img, start, 100, end)
    elif event == cv.EVENT_LBUTTONDBLCLK:
        start = np.array([x, y])
        end = np.array([x+10, y+10])
        img = warp.LocalScalingWarps(img, start, end, 0.7)
    else:
        pass

    cv.imshow("image", img)
    

def testWarps():
    global img
    cv.namedWindow("image")
    cv.setMouseCallback("image", onMouseWarps)
    cv.imshow("image", img)

    if cv.waitKey(0):
        return


def onMouseVerticalScaleWarps(event, x, y, flags, param):
    global img, start, end
    if event == cv.EVENT_LBUTTONDOWN:
        start = np.array([x, y])
    elif event == cv.EVENT_LBUTTONUP:
        end = np.array([x, y])
        img = warp.verticalScaleWarps(img, start[1], end[1], 1.5)
    
    cv.imshow("image", img)


def testVerticalScaleWarps():
    global img
    cv.namedWindow("image")
    cv.setMouseCallback("image", onMouseVerticalScaleWarps)
    cv.imshow("image", img)

    if cv.waitKey(0):
        return

if __name__ == "__main__":
    imgDir = "./2.jpeg"
    img = cv.imread(imgDir)
    # testWarps()
    # testLocalScalingWarps(imgDir)
    # testLocalTranslationWarps(imgDir)
    testVerticalScaleWarps()

