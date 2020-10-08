import copy
import sys
import numpy as np
import cv2 as cv
import logging

logging.basicConfig(level=logging.DEBUG)

def normL1(vector):
    return np.sum(vector*vector, axis=0, keepdims=False)**0.5

def normL2(vector):
    return np.sum(vector*vector, axis=0, keepdims=False)

# local scaling
# center (x,y)
# orient (x,y)
def LocalScalingWarps(img, center, orient, level):
    h,w = img.shape[0], img.shape[1]
    resImg = copy.deepcopy(img)
    radius = int(normL1(orient - center)) # float
    top = max(0, center[1] - radius)
    down = min(center[1] + radius, h)
    left = max(0, center[0] - radius)
    right = min(center[0]+ radius, w)

    logging.debug("LocalScalingWarps")
    logging.debug("img shape: {}".format(img.shape))
    logging.debug("center: {}, size: {}".format(center, center.size))
    logging.debug("radius: {}".format(radius))
    logging.debug("top: {}, down: {}, left: {}, right: {}".format(top, down, left, right))

    try:
        for i in range(top, down, 1):
            for j in range(left, right, 1):
                dest = np.array([j, i])
                destInCircle = dest - center
                destR = normL1(destInCircle)
                if destR > radius or (dest == center).all(): 
                    continue

                sourceR = (1 - ((destR/radius -1)**2)*level)*destR
                sourceInCircle = sourceR / destR * destInCircle
                sourceInCircle = np.rint(sourceInCircle).astype(np.int)
                source = sourceInCircle + center

                if (source[0] >= 0 and source[0] < w) and (source[1] >= 0 and source[1] < h) \
                    and normL1(source - center) <= radius:
                    #print(dest, source)
                    resImg[dest[1]][dest[0]] = img[source[1]][source[0]]
    except:
        print("Unexpected error:", sys.exc_info()[0])
        assert False
    
    return resImg


# local translation
# center (x, y)
# orient (x, y)
def LocalTranslationWarps(img, center, radius, orient):
    h,w = img.shape[0], img.shape[1]
    resImg = copy.deepcopy(img)
    top = max(0, center[1] - radius)
    down = min(center[1] + radius, h)
    left = max(0, center[0] - radius)
    right = min(center[0] + radius, w)

    logging.debug("LocalTranslationWarps")
    logging.debug("img shape: {}".format(img.shape))
    logging.debug("center: {}, size: {}".format(center, center.size))
    logging.debug("radius: {}".format(radius))
    logging.debug("top: {}, down: {}, left: {}, right: {}".format(top, down, left, right))

    try:
        for i in range(top, down, 1):
            for j in range(left, right, 1):
                dest = np.array([j, i])
                temp1 = normL2(dest - center)
                temp2 = normL2(orient - center)
                temp3 = radius * radius

                if temp1 > temp3: continue
                if temp2 < 4*temp3:
                    temp2 = 6.25*temp3

                e = (1 - temp2*1.0 / (temp3 - temp1 + temp2))**2
                source = dest - e*(orient - center)
                source = np.rint(source).astype(np.int)
                
                if source[0] >= 0 and source[0] < w and source[1] >= 0 and source[1] < h :
                    #resImg[dest[1]][dest[0]] = np.array([0,0,0])
                    #print(dest, source)
                    resImg[dest[1]][dest[0]] = img[source[1]][source[0]]
    except:
        print("Unexpected error:", sys.exc_info()[0])
        assert False

    return resImg

# vertical scale
# top int
# down int
# level float [0, 2]
def verticalScaleWarps(img, top, down, level):
    if top == down : return img
    h,w = img.shape[0], img.shape[1]
    logging.debug(img.dtype)
    resImg = copy.deepcopy(img)

    if top < 0 or top >=h or down < 0 or down >= h:
        logging.error("wrong (top, down) which big than img size")
        assert False

    if level < 0 or level > 2:
        logging.error("wrong level which should between [0, 2]")
        assert False
    
    if (top > down): 
        top, down = down, top

    oldDistance = down - top
    newDistance = int(oldDistance*level)
    newHeight = h - oldDistance + newDistance
    newDown = top + newDistance

    resImg = np.zeros((newHeight, w, 3), dtype=img.dtype)
    logging.debug("new image shape {}".format(resImg.shape))

    # fill the part that same as before
    resImg[:top, :, :] = img[:top, :, :]
    resImg[newDown:, :, :] = img[down:, :, :]

    # fill the new part
    oldPart = img[top:down, : , :]
    newPart = cv.resize(oldPart, (w, newDistance), interpolation=cv.INTER_CUBIC)
    resImg[top:newDown, :, :] = newPart

    return resImg


def horizontalScaleWarps(img, left, right, level):
    if left == right: return img
    h,w = img.shape[0], img.shape[1]
    logging.debug(img.dtype)
    resImg = copy.deepcopy(img)

    if left < 0 or left >=w or right < 0 or right >= w:
        logging.error("wrong (left, right) which big than img size")
        assert False

    if level < 0 or level > 2:
        logging.error("wrong level which should between [0, 2]")
        assert False
    
    if (left > right): 
        left, right = right, left
        
    oldDistance = down - top
    newDistance = int(oldDistance*level)
    newHeight = h - oldDistance + newDistance
    newDown = top + newDistance

    resImg = np.zeros((newHeight, w, 3), dtype=img.dtype)
    logging.debug("new image shape {}".format(resImg.shape))

    # fill the part that same as before
    resImg[:top, :, :] = img[:top, :, :]
    resImg[newDown:, :, :] = img[down:, :, :]

    # fill the new part
    oldPart = img[top:down, : , :]
    newPart = cv.resize(oldPart, (w, newDistance), interpolation=cv.INTER_CUBIC)
    resImg[top:newDown, :, :] = newPart

    return resImg