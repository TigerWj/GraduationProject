import os
import glob
import warp
import json
import logging
import cv2 as cv

class dataProducer：

    def __init__(self, datasetDir):
        self.datasetDir = datasetDir
        self.normalDatasetDir = os.path.join(datasetDir, "normal")
        self.keyPointsDir = os.path.join(datasetDir, "result")
        self.imageList = os.listdir(self.normalDatasetDir)
        self.keyPointList = glob.glob(self.keyPointsDir + "*.json")

        # 数量得一样，图片和关键点json文件需匹配
        if (len(self.imageList) != len(self.keyPointList)) :
            logging.error("dataset error: the number of images not equal to keyPoints")
            exit(1)

        sort(self.imageList)
        sort(self.keyPointList)

    def run(self):
        for i in range(len(self.imageList)):
            try:
                img = cv.imread(os.path.join(self.normalDatasetDir, self.imageList[i]))
                with open(os.path.join(self.keyPointList[i]), 'r') as f:
                    people = json.load(f)["people"]
            except:
                logging.error("fileLoad error:", sys.exc_info()[0])
                exit(1)
            
            


            # heigthenLeg
            heigthenLegDir = os.path.json(self.datasetDir, "heightenLeg")
            if os.path.exists(heigthenLegDir):
                os.makedirs(heigthenLegDir)
        

    # {8,  "MidHip"},
    # {9,  "RHip"},
    # {10, "RKnee"},
    # {11, "RAnkle"},
    # {12, "LHip"},
    # {13, "LKnee"},
    # {14, "LAnkle"},
    def heigthenLeg(self, img, imgFileName, keyPoints_2d, saveDir):
        coefficient = [0.25, 0.5, 0.75, 1.25, 1.75, 2.0]

        resImg = warp.verticalScaleWarps(img, )


def loadData()