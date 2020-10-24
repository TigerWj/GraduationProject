import numpy as np
import cv2 as cv
import warp
import sys

img = None
start = None
end = None
mode = 1

def testLocalScalingWarps(imgDir):
    img = cv.imread(imgDir)
    cv.imshow("image", img)
    out = warp.LocalScalingWarps(img, np.array([68, 88]), np.array([91, 75]), 10)
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
            img = warp.LocalTranslationWarps(img, start, 100, end)
        else:
            img = warp.verticalScaleWarps(img, start[1], end[1], 1.3)

    elif event == cv.EVENT_LBUTTONDBLCLK:
        start = np.array([x, y])
        end = np.array([x+10, y+10])
        img = warp.LocalScalingWarps(img, start, end, 0.9)

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


if __name__ == "__main__":
    print(sys.argv)
    print("Press \"c\" to change mode, holding click and move to get tansformation ")
    imgDir = sys.argv[1]
    img = cv.imread(imgDir)
    testWarps()


