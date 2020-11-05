import numpy as np
import math

def getScatters(p1, p2, ratioNum, width):
    if ((p1==p2).all()):
        raise TypeError("two point")
    
    ratioNum += 2

    if (p1[0] == p2[0]):
        yList = np.linspace(p1[1], p2[1], ratioNum)
        x1List = np.ones((ratioNum, 1)) * (p1[0] - width)
        x2List = np.ones((ratioNum, 1))* (p2[0] + width)
        return np.concatenate(x1List, yList), np.concatenate(x2List, yList)
    elif (p1[1] == p2[1]):
        xList = np.linspace(p1[0], p2[0], ratioNum)
        y11List = np.ones((ratioNum, 1)) * (p1[1] - width)
        y2List = np.ones((ratioNum, 1))* (p2[1] + width)
        return np.concatenate(xList, y1List), np.concatenate(xList, y2List)
    else:
        temp = p2 - p1
        k = temp[1] / temp[0]
        lb = p1[1] - k*p1[0]

        xList = np.linspace(p1[0], p2[0], ratioNum).reshape(-1, 1)
        yList = k*xList + lb
        # print(xList, yList)
        A = 1
        B = -2*p1[1]
        C = p1[1]*p1[1] - (width*width*1.0) / (k*k +1)                                                                                                                                                         

        y1, y2 = quadratic(A, B, C)
        x1, x2 = k*(p1[1]-y1)+p1[0], k*(p1[1]-y2)+p1[0]
        print(x1,y1,x2,y2)
        return np.concatenate([xList + x1-p1[0], yList + y1-p1[1]], axis=1), np.concatenate([xList + x2-p1[0], yList + y2-p1[1]], axis=1)

        

# ax^2 + bx + c =0
def quadratic(a, b, c):
    if a == 0:
        raise TypeError("a can not be 0")

    if (not isinstance(a, (int, float)) or not isinstance(b, (int, float) or not isinstance(c, (int, float)))):
        raise TypeError("bad operand type")

    delta = math.pow(b, 2) - 4*a*c
    if (delta < 0):
        return None, None
    print(delta+b)
    x1 = (math.sqrt(delta) - b)/(2*a)
    x2 = -1*(math.sqrt(delta) + b)/(2*a)
    return x1, x2