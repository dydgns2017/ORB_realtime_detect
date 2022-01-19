import glob
from unittest.mock import patch
import cv2
import os, time, sys
import numpy as np
import matplotlib.pyplot as plt

## folder location set
Root_folder = "./"
RGB_folder = os.path.join(Root_folder, "images/RGB")
Canny_folder = os.path.join(Root_folder, "images/Canny")
Sobel_folder = os.path.join(Root_folder, "images/Sobel")
LD_folder = os.path.join(Root_folder, "images/LD")
Lapla_folder = os.path.join(Root_folder, "images/Lapla")
HSV_folder = os.path.join(Root_folder, "images/HSV")
GRAY_folder = os.path.join(Root_folder, "images/GRAY")
Denoise_folder = os.path.join(Root_folder, "images/Denoise")
folders = [Canny_folder, Sobel_folder, LD_folder, Lapla_folder, RGB_folder, HSV_folder, GRAY_folder]

class CreateDataSet:
    def __init__(self) -> None:
        self.cleanDataset()
    
    def cleanDataset(self):
        for folder in folders:
            files = glob.glob(folder + "\\*.jpg")
            for file in files:
                os.remove(file)
    
    def createHSV(self):
        global HSV_folder
        for file in self.files:
            temp_file = cv2.imread(file)
            HSV = cv2.cvtColor(temp_file, cv2.COLOR_BGR2HSV)
            cv2.imwrite(HSV_folder + "/" + file.split("\\")[-1], HSV)

    def createCanny(self):
        global Canny_folder
        for file in self.files:
            temp_file = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            # code...
            canny = cv2.Canny(temp_file, 50, 200)
            cv2.imwrite(Canny_folder + "/" + file.split("\\")[-1], canny)

    def createSobel(self): 
        global Sobel_folder
        for file in self.files:
            temp_file = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            # code...
            sobel = cv2.Sobel(temp_file, ddepth=cv2.CV_8UC1, dx=1,dy=1, ksize=5)
            cv2.imwrite(Sobel_folder + "/" + file.split("\\")[-1], sobel)

    def createLD(self): 
        global LD_folder
        for file in self.files:
            temp_file = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            # code...
            lapla = cv2.Laplacian(temp_file, cv2.CV_8UC1)
            kernel = np.ones((3,3), np.uint8)
            ld = cv2.dilate(lapla, kernel, iterations=1)
            cv2.imwrite(LD_folder + "/" + file.split("\\")[-1], ld)

    def createLapla(self):
        global Lapla_folder
        for file in self.files:
            temp_file = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            # code...
            lapla = cv2.Laplacian(temp_file, cv2.CV_8UC1)
            cv2.imwrite(Lapla_folder + "/" + file.split("\\")[-1], lapla)
    
    def createGray(self):
        global GRAY_folder
        for file in self.files:
            temp_file = cv2.imread(file)
            temp_file = cv2.cvtColor(temp_file, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(GRAY_folder + "/" + file.split("\\")[-1], temp_file)
    
    def createDenoise(self): # with histogram
        global Denoise_folder
        for file in self.files:
            temp_file = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            image = cv2.fastNlMeansDenoising(temp_file, None)
            cv2.imwrite(Denoise_folder + "/" + file.split("\\")[-1], image)
    
    def createStart(self):
        self.files = glob.glob(".\\images\\origin\\*.jpg")
        for file in self.files:
            f = cv2.imread(file)
            f = cv2.resize(f, (680, 540))
            cv2.imwrite(RGB_folder + "/" + file.split("\\")[-1], f)
        self.files = glob.glob(".\\images\\RGB\\*.jpg")
        self.createLapla()
        self.createLD()
        self.createSobel()
        self.createCanny()
        self.createHSV()
        self.createGray()
        self.createDenoise()

class ORBDetection:
    def __init__(self, select_folder) -> None:
        self.orb_detector = cv2.ORB_create(30000)
        self.cap = cv2.VideoCapture(0)
        # 클래스 변경시 변경.
        self.match_files = [[cv2.imread(file), file] for file in glob.glob(select_folder + "/*.jpg")]
        self.bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
        # target image features extract
        self.features = []
        for match_file in self.match_files:
            file, filename = match_file
            kp2, desc2 = self.orb_detector.detectAndCompute(file, None)
            self.features.append([kp2, desc2, file, filename])

        self.orb_detector = cv2.ORB_create(200)
        self.target_fps = 60
        self.stable_stack = []
        self.target_feature = None

    def mostFrequent(self):
        most = max(set(self.stable_stack), key=self.stable_stack.count)
        for feature in self.features:
            filename = feature[-1] # unpacking
            if ( os.path.basename(filename) == most ):
                self.target_feature = feature
                break
        return self.target_feature

    def featureExtractRT(self, features):
        ## real time feature extract -> each frame
        dst = []
        for feature in features:
            kp2, desc2, file, filename = feature # unpacking
            matches = self.bf.match(self.desc1, desc2)
            matches = sorted(matches[:30], key = lambda x:x.distance)
            L = [sum([x.distance for x in matches]), kp2, file, matches, filename]
            dst.append(L)
        #dst = sorted(dst, key=lambda x:x[0], reverse=True)
        dst = sorted(dst, key=lambda x:x[0])
        _, kp2, file, matches, filename = dst[0]
        filename = os.path.basename(filename)
        # print(os.path.basename(filename))
        return dst, filename
    
    def startDetect(self, detect=None):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if detect != None:
                frame = detect(frame)
            self.kp1, self.desc1 = self.orb_detector.detectAndCompute(frame, None)
            # get real time frame &  distance matching calculation
            dst, filename = self.featureExtractRT(self.features)
            self.stable_stack.append(filename)
            if ( len(self.stable_stack) >= self.target_fps ):
                self.target_feature = self.mostFrequent()
                self.stable_stack = []
            # draw image
            if self.target_feature == None:
                continue
            kp2, desc2, file, filename = self.target_feature # unpacking
            matches = self.bf.match(self.desc1, desc2)
            matches = sorted(matches, key = lambda x:x.distance)
            res = cv2.drawMatches(frame, self.kp1, file, kp2, matches, None, \
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.namedWindow("res", cv2.WINDOW_NORMAL)
            cv2.imshow("res", res)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

def getLapla(frame):
    return cv2.Laplacian(frame, cv2.CV_8UC1)

def getLD(frame):
    frame = cv2.Laplacian(frame, cv2.CV_8UC1)
    kernel = np.ones((3,3), np.uint8)
    return cv2.dilate(frame, kernel, iterations=1)

def getSobel(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Sobel(frame, ddepth=cv2.CV_8UC1, dx=1,dy=1, ksize=5)

def getCanny(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(frame, 50, 200)

def getHSV(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return frame

def getGRAY(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    return frame