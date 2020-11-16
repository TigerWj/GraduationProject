import numpy as np
import math
import json
import os
import cv2 as cv

def getIntersection(start, orient, img, leftFlag, rightFlag, ratioNum, ratioIndex, gap=3, alpha=0):
    if (start == orient).all():
        return None, None

    """
    @description : get intersection between line and img outline
    @param : 
    @Returns : 
    """

    if (start == orient).all():
        return None, None
    
    h,w = img.shape[0], img.shape[1]
    gapDistance = gap

    sList = []
    dList = []
    temp = orient - start

    if temp[0] == 0: # slope not exist
        print("1")
        xlist = np.array([start[0]*ratioNum])[1:-1].reshape(-1, 1)
        ylist = np.linspace(start[1], orient[1], ratioNum+2)[1:-1].reshape(-1, 1)
        samplePoints = np.concatenate([xlist, ylist], axis=1)

        for i in ratioIndex:
            p = samplePoints[i, :]
            x1 = round(p[0])
            x2 = x1
            y = round(p[1])

            if (rightFlag):
                while x1<w-1 and img[y][x1] > 127:
                    x1 += 1

                x1 += gapDistance
                sList.append([x1, y])
                x1 = min(round(p[0] + alpha*(x1 - p[0])), w-1)
                dList.append([x1, y])

            if (leftFlag):
                while x2>0 and img[y][x2] > 127:
                    x2 -= 1
                
                x2 -= gapDistance
                sList.append([x2, y])
                x2 = max(0, round(p[0] - alpha*(p[0]-x2)))
                dList.append([x2, y])

    elif temp[1] == 0: # slope = 0
        print("2")
        ylist = np.array(start[1]*ratioNum).reshape(-1, 1)
        xlist = np.linspace(start[0], orient[0], ratioNum+2)[1:-1].reshape(-1, 1)
        samplePoints = np.concatenate([xlist, ylist], axis=1)

        for i in ratioIndex:
            p = samplePoints[i, :]
            x = round(p[0])
            y1 = round(p[1])
            y2 = y1

            if (rightFlag):
                while y1>0 and img[y1][x]>127:
                    y1 -= 1
                
                y1 -= gapDistance
                sList.append([x, y1])
                y1 = max(0, round(p[1] - alpha*(p[1]-y1)))
                dList.append([x, y1])

            if (leftFlag):
                while y2<h-1 and img[y2][x]>127:
                    y2 += 1
                    sList.append([x, y2])
                    y2 = min(h-1, round(p[1] + alpha*(y2 - p[1])))
                    dList.append([x, y2])

    else:
        k1 = temp[1]*1.0/temp[0]
        b1 = start[1] - start[0]*k1
        k2 = 1.0/k1*-1
        samplePoints = None
        
        if abs(k1) >= 1:
            yList = np.linspace(start[1], orient[1], ratioNum+2)[1:-1].reshape(-1, 1)
            samplePoints = np.concatenate([(yList - b1)/k1, yList], axis=1) # 采样点
        else:
            xList = np.linspace(start[0], orient[0], ratioNum+2)[1:-1].reshape(-1, 1)
            samplePoints = np.concatenate([xList, xList*k1 + b1], axis=1) # 采样点

        # 利用了计算机图形学中的直线生成算法
        delta = k2 if abs(k2) <= 1 else k1*-1
        if abs(k2) > 1:
            for i in ratioIndex:
                p = samplePoints[i, :]
                x1, y1 = round(p[0]), round(p[1])
                x2, y2 = x1, y1
                
                if (rightFlag):
                    while y1>0 and x1>0 and img[y1][round(x1)] > 127:
                        y1 -= 1
                        x1 -= delta
                    y1 -= gapDistance
                    x1 -= gapDistance*delta

                    sList.append([x1, y1])
                    y1 = max(0, round(p[1] - alpha*(p[1] - y1)))
                    x1 = max(0, round(p[0] - alpha*(p[1] - y1)*delta))
                    dList.append([x1, y1])

                if (leftFlag):
                    while y2<h-1 and x2<w-1 and img[y2][round(x2)] > 127:
                        y2 += 1
                        x2 += delta
                    y2 += gapDistance
                    x2 += gapDistance*delta

                    sList.append([x2, y2])
                    y2 = min(h-1, round(p[1] + alpha*(y2 - p[1])))
                    x2 = min(w-1, round(p[0] + alpha*(y2 - p[1])*delta))
                    dList.append([x2, y2])
        else:
            for i in ratioIndex:
                p = samplePoints[i, :]
                x1, y1 = round(p[0]), round(p[1])
                x2, y2 = x1, y1
                
                if (leftFlag):
                    while x1>0 and y1>0 and img[round(y1)][x1] > 127:
                        x1 -= 1
                        y1 -= delta
                    x1 -= gapDistance
                    y1 -= gapDistance*delta

                    sList.append([x1, y1])
                    y1 = max(0, round(p[1] - alpha*(p[0] - x1)*delta))
                    x1 = max(0, round(p[0] - alpha*(p[0] - x1)))
                    dList.append([x1, y1])

                if (rightFlag):
                    while x2<w-1 and y2<h-1 and img[round(y2)][x2] > 127:
                        x2 += 1
                        y2 += delta
                    x2 += gapDistance
                    y2 += gapDistance*delta

                    sList.append([x2, y2])
                    y2 = min(h-1, round(p[1] + alpha*(x2 - p[0])*delta))
                    x2 = min(w-1, round(p[0] + alpha*(x2 - p[0])))
                    dList.append([x2, y2])

    return sList, dList



def getParaScatters(p1, p2, ratioNum, width, weight=None):
    if (p1==p2).all():
        return None
    
    if p1[0] == p2[0]:
        yList = np.linspace(p1[1], p2[1], ratioNum+2)[1:-1]
        x1List = np.array([p1[0] - width]*ratioNum)
        x2List = np.array([p1[0] + width]*ratioNum)
        return np.concatenate(x1List, yList), np.concatenate(x2List, yList)

    elif p1[1] == p2[1]:
        xList = np.linspace(p1[0], p2[0], ratioNum+2)[1:-1]
        y1List = np.array([p1[1] - width]*ratioNum)
        y2List = np.array([p1[1] + width]*ratioNum)
        return np.concatenate(xList, y1List), np.concatenate(xList, y2List)
        
    else:
        temp = p2 - p1
        k = temp[1] / temp[0]
        b = p1[1] - k*p1[0]

        xList = np.linspace(p1[0], p2[0], ratioNum+2)[1:-1].reshape(-1, 1)
        yList = k*xList + b
        
        if weight==None: # 所有点等距离
            A = 1
            B = -2*p1[1]
            C = p1[1]*p1[1] - (width*width*1.0) / (k*k +1)                                                                                                                                                         

            y1, y2 = quadratic(A, B, C)
            x1, x2 = k*(p1[1]-y1)+p1[0], k*(p1[1]-y2)+p1[0]
            
            return np.concatenate([xList + x1-p1[0], yList + y1-p1[1]], axis=1).tolist(), np.concatenate([xList + x2-p1[0], yList + y2-p1[1]], axis=1).tolist()
        else: # 按权重分配点的距离
            res1 = []
            res2 = []

            for i in range(xList.shape[0]):
                print(yList[i])
                A = 1
                B = -2*yList[i][0]
                C = yList[i][0]*yList[i][0] - (width*width*1.0*weight[i]*weight[i]) / (k*k +1)

                y1, y2 = quadratic(A, B, C)
                x1, x2 = k*(yList[i][0]-y1)+p1[0], k*(yList[i][0]-y2)+p1[0]
                res1.append([x1, y1])
                res2.append([x2, y2])
                
            return res1, res2


def beforeCluster(img):
    # 二值化 获取轮廓点
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    contours, cnt = cv.findContours(binary.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(binary, contours, -1, 127)

    contours = np.where(binary == 127)
    contoursMatrix = np.concatenate([contours[1].reshape(-1, 1), contours[0].reshape(-1, 1)], axis=1)

    return contoursMatrix

def clusterPoint(contoursMatrix, pointSet, expectCategory):
    """
    @description : 
    @param :
        pointSet: list[], 定义类别的关键点集合
    @Returns :
    """

    distanceMatrix = np.zeros((contoursMatrix.shape[0], len(pointSet)))

    # 预处理 点集合 聚类
    for idx, e in enumerate(pointSet):
        d = np.sqrt(np.sum(np.square(contoursMatrix - np.array(e[0])), axis=1))
        d += np.sqrt(np.sum(np.square(contoursMatrix - np.array(e[1])), axis=1))
        distanceMatrix[:, idx] = d
    category = np.argmin(distanceMatrix, axis=1)
    return np.where(np.isin(category, expectCategory)), category[np.where(np.isin(category, expectCategory))]

def processAfterCluster(realContours, targetContours, expectCategory, pointSet, distanceThreshold, sampleRate=0.15):
    """
    @description: 对于每一个类别的点集合，判别endpoints构成的线段是否可以将集合分成相对均匀的两个部分，以此区分单侧形变和双侧形变； 通过一个固定的距离阈值，决定该类别中的点集合是否实施形变
    @param:
    @Returns:
    """

    res = {}

    for c in expectCategory:
        pset = np.where(realContours[:, 2] == c)
        endPoints = pointSet[c]
        pset1Idx = []
        pset2Idx = []

        tempArray = realContours[pset]
        tempArray[:, 2] = pset[0]
        tempArray = tempArray.tolist()
        tempArray.sort(key = lambda x : (x[0], x[1]))
        sampleIndex = np.linspace(0, len(tempArray)-1, num=int(len(tempArray)*sampleRate), dtype=np.int)
        sampleIndex = np.array(tempArray, dtype=np.int)[sampleIndex][:, 2]
        pset = sampleIndex

        k = 1.0*(endPoints[1][1] - endPoints[0][1]) / (endPoints[1][0] - endPoints[0][0])
        b = endPoints[1][1] - endPoints[1][0]*k

        # 根据直线原理分类
        for idx in range(pset.shape[0]):
            idx = pset[idx]
            t = realContours[idx, :2]
            
            if t[1]-b-k*t[0] > 0 :
                pset1Idx.append(idx)
            else:
                pset2Idx.append(idx)

        # 统计点到直线的距离
        count = 0
        temp = math.sqrt(k*k + 1)
        for i in range(len(pset1Idx)):
            l = abs((k*realContours[pset1Idx[i]][0] - realContours[pset1Idx[i]][1] + b) / temp)

            if l > distanceThreshold: 
                count += 1

            # 五分之一的点 距离超过阈值 则不可以
            if count >= len(pset1Idx)/5: 
                break
        
        if count < len(pset1Idx)/5:
            if c in res:
                res[c].append(pset1Idx)
            else:
                res[c] = [pset1Idx]
        
        count = 0
        for i in range(len(pset2Idx)):
            l = abs((k*realContours[pset2Idx[i]][0] - realContours[pset2Idx[i]][1] + b) / temp)

            if l > distanceThreshold: 
                count += 1

            # 五分之一的点 距离超过阈值 则不可以
            if count >= len(pset2Idx)/5: 
                break
        
        if count < len(pset2Idx)/5:
            if c in res:
                res[c].append(pset2Idx)
            else:
                res[c] = [pset2Idx]
    
    return res


def xxx(img, contours, res, pointSet, alpha=30):

    transformInfo = {}
    print(res)
    for ca, v in res.items():
        transformInfo[ca] = []
        for i in range(len(v)):
            index = v[i]
            endPoints = np.array(pointSet[ca])
            
            p1 = np.array([0, 0])
            p2 = np.array([1000, 0])
            c = 0

            for idx in index:
                intersectionPoint = getProjectPointOnLine(contours[idx], endPoints)
                cv.drawMarker(img, (round(intersectionPoint[0]), round(intersectionPoint[1])), (0, 255, 0), markerType=cv.MARKER_STAR ,markerSize=3)
                if intersectionPoint[0] > p1[0]:
                    p1 = intersectionPoint
                if intersectionPoint[0] < p2[0]:
                    p2 = intersectionPoint
                
                tempDistance = pointDistance(contours[idx][0], contours[idx][1], intersectionPoint[0], intersectionPoint[1])
                if tempDistance > c:
                    c = tempDistance
            
            # 垂线
            k = 1.0*(endPoints[1][1] - endPoints[0][1]) / (endPoints[1][0] - endPoints[0][0])
            b = endPoints[1][1] - endPoints[1][0]*k
            
            # 垂线的中垂线
            k1 = 1.0/k*-1
            b1 = 1.0*(p1[1]+p2[1])/2 - 1.0*k1*(p1[0]+p2[0])/2

            center = getPointAlongLine(k1, b1, (p1[0]+p2[0])/2, (p1[1]+p2[1])/2, c)
            print(center)
            # cv.drawMarker(img, (round((p1[0]+p2[0])/2), round((p1[1]+p2[1])/2)), (255, 0, 0), markerType=cv.MARKER_STAR ,markerSize=3)

            transformInfo[ca].append([round((p1[0]+p2[0])/2), round((p1[1]+p2[1])/2)]) #orient

            temp = contours[index[0]][0]*k + b - contours[index[0]][1]
            temp1 = center[0][0]*k + b - center[0][1]

            # center
            if temp*temp1 > 0:
                transformInfo[ca].append(center[0])
                # transformInfo[ca].append(center[1])
            else:
                transformInfo[ca].append(center[1])
                # transformInfo[ca].append(center[0])
            
            radius = math.sqrt(math.pow(alpha, 2) + math.pow(pointDistance(p1[0], p1[1], p2[0], p2[1])/2, 2))
            transformInfo[ca].append([radius])
    cv.namedWindow("i")
    cv.imshow("i", img)
    cv.waitKey(0)

    return transformInfo



def getPointAlongLine(k, b, x, y, distance):
    A = k*k+1
    B = 2*(k*(b-y)-x)
    C = x*x + (b-y)*(b-y) - distance*distance

    x1, x2 = quadratic(A, B, C)
    y1, y2 = k*x1+b, k*x2+b

    return [[round(x1), round(y1)], [round(x2), round(y2)]]


def getPointAcrodLine(start, orient, imgWidth, imgHeight, alpha):

    if (start==orient).all():
        return np.array([None, None])

    temp = orient - start
    k = temp[1]*1.0/temp[0]
    b = start[1] - start[0]*k
    x = 0
    y = 0

    if temp[0] == 0:
        x = orient[0]
        y = orient[1] + alpha*(orient[1]-start[1])

    elif temp[1] == 0:
        y = orient[1]
        x = orient[0] + alpha*(orient[0]-orient[0])

    else:
        if abs(k) >= 1:
            y = orient[1] + alpha*(orient[1]-start[1])
            x = (y - b)*1.0/k 
        else:
            x = orient[0] + alpha*(orient[0]-start[0])
            y = k*x + b

        x = max(0, x) if x<0 else min(x, imgWidth-1)
        y = max(0, y) if y<0 else min(y, imgHeight-1)

    return np.array([round(x), round(y)])

def getProjectPointOnLine(point, linepoint):
    """
    @description:
    @param:
        point: np.array, (2, )
        linepoint: np.array, (2, 2)
    @Returns:
    """

    temp = linepoint[1, :] - linepoint[0, :]
    k = temp[1]*1.0/temp[0]

    x = (point[1] + 1/k*point[0] - linepoint[0][1] + k*linepoint[0][0]) / (k + 1/k)
    y = k*(x-linepoint[0][0]) + linepoint[0][1]

    return np.array([round(x), round(y)])

        
# ax^2 + bx + c =0
def quadratic(a, b, c):
    if a == 0:
        raise TypeError("a can not be 0")

    if (not isinstance(a, (int, float)) or not isinstance(b, (int, float) or not isinstance(c, (int, float)))):
        raise TypeError("bad operand type")

    delta = math.pow(b, 2) - 4*a*c
    if (delta < 0):
        return None, None
    
    x1 = (math.sqrt(delta) - b)/(2*a)
    x2 = -1*(math.sqrt(delta) + b)/(2*a)
    return x1, x2

def loadKeypoints(jsonDir):
    with open(os.path.join(jsonDir), 'r') as f:
        people = json.load(f)["people"]

    if len(people) != 1:
        logging.error("the number of people is not 1")
        exit(1)

    keyPoints = np.array(people[0]['pose_keypoints_2d']).reshape((-1, 3))
    
    return keyPoints

def pointDistance(a, b, c, d):
    return math.sqrt(math.pow(a-c, 2)+math.pow(b-d, 2))
