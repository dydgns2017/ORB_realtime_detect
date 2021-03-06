import glob
import cv2
import os
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
Clahe_folder = os.path.join(Root_folder, "images/Clahe")
folders = [Canny_folder, Sobel_folder, LD_folder, Lapla_folder, RGB_folder, \
    HSV_folder, GRAY_folder, Clahe_folder]

## hist information get
def getHist(histb):
    if (len(histb)==2):
        histb, filename = histb
    else:
        filename = None
    histb = cv2.cvtColor(histb, cv2.COLOR_BGR2HSV)
    histb = cv2.calcHist([histb],[0,1],None,[90,256],[1,180, 1,256])
    cv2.normalize(histb, histb, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return histb, filename


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
    return cv2.Canny(frame, 80, 200)

def getHSV(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return frame

def getClahe(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(27,27))
    cl1 = clahe.apply(frame)
    return cl1

def getGRAY(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

class CreateDataSet:
    def __init__(self) -> None:
        self.cleanDataset()
    
    def cleanDataset(self):
        for folder in folders:
            files = glob.glob(folder + "\\*.jpg")
            for file in files:
                os.remove(file)
            
    def createClahe(self):
        global Clahe_folder
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(27,27))
        for file in self.files:
            temp_file = cv2.imread(file, 0)
            cl1 = clahe.apply(temp_file)
            cv2.imwrite(Clahe_folder + "/" + file.split("\\")[-1], cl1)
                
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
            canny = cv2.Canny(temp_file, 80, 200)
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
    
    def createStart(self):
        self.files = glob.glob(".\\images\\origin\\*.jpg")
        for file in self.files:
            f = cv2.imread(file)
            f = cv2.resize(f, (540, 540))
            cv2.imwrite(RGB_folder + "/" + file.split("\\")[-1], f)
        self.files = glob.glob(".\\images\\RGB\\*.jpg")
        self.createLapla()
        self.createLD()
        self.createSobel()
        self.createCanny()
        self.createHSV()
        self.createGray()
        self.createClahe()

        
class ORBDetection:
    def __init__(self, select_folder) -> None:
        self.orb_detector = cv2.ORB_create()
        self.cap = cv2.VideoCapture(0)
        temp_list = glob.glob(select_folder + "/*.jpg")
        self.match_files = [[cv2.imread(file), file] for file in temp_list]
        self.bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
        # target image features extract
        self.features = []
        for match_file in self.match_files:
            file, filename = match_file
            kp2, desc2 = self.orb_detector.detectAndCompute(file, None)
            self.features.append([kp2, desc2, file, filename])

        self.orb_detector = cv2.ORB_create(500)
        self.target_fps = 30
        self.stable_stack = []
        self.target_feature = None
        # get histogram data
        images = [[cv2.imread(file), file] for file in glob.glob('images/RGB/*.jpg')]
        self.hists = list(map(getHist, images))

    def mostFrequent(self):
        most = max(self.stable_stack, key=self.stable_stack.count)
        for feature in self.features:
            filename = feature[-1] # unpacking
            if ( os.path.basename(filename) == most ):
                self.target_feature = feature
                break
        return self.target_feature

    def featureExtractRT(self, cls, features):
        ## real time feature extract -> each frame
        dst = []
        features = [feature for feature in features if feature[-1].split("_")[1] == cls]
        for feature in features:
            kp2, desc2, file, filename = feature # unpacking
            matches = self.bf.match(self.desc1, desc2)
            matches = sorted(matches[:40], key = lambda x:x.distance)
            L = [sum([x.distance for x in matches]), kp2, file, matches, filename]
            dst.append(L)
        dst = sorted(dst, key=lambda x:x[0])
        _, kp2, file, matches, filename = dst[0]
        filename = os.path.basename(filename)
        return dst, filename

    def getClass(self, frame):
        h,w,_ = frame.shape
        fr = frame[h//4:360,w//4:480]
        h,w,_ = fr.shape
        cv2.rectangle(fr, (0,0), (w-1, h-1), (0, 0, 255), thickness=2)
        fr_hist,_ = getHist(fr)
        scores = []
        for hist in self.hists:
            hist, filename = hist
            res_compare = cv2.compareHist(fr_hist,hist,cv2.HISTCMP_BHATTACHARYYA)
            scores.append([res_compare, filename])
        scores = sorted(scores, key=lambda x:x[0])
        cls = os.path.basename(scores[0][1]).split("_")[1]
        return cls, fr
    
    def startDetect(self, detect=None):
        while self.cap.isOpened():
            _, frame = self.cap.read()
            cls, fr = self.getClass(frame)
            if detect != None:
                frame = detect(frame)
            self.kp1, self.desc1 = self.orb_detector.detectAndCompute(frame, None)
            # get real time frame &  distance matching calculation
            dst, filename = self.featureExtractRT(cls, self.features)
            self.stable_stack.append(filename)
            if dst == None and filename == None:
                self.stable_stack.pop()
            if ( len(self.stable_stack) >= self.target_fps ):
                self.target_feature = self.mostFrequent()
                self.stable_stack = []
            # draw image
            if self.target_feature == None:
                continue
            kp2, desc2, file, filename = self.target_feature # unpacking
            matches = self.bf.match(self.desc1, desc2)
            matches = sorted(matches, key = lambda x:x.distance)
            res = cv2.drawMatches(frame, self.kp1, file, kp2, matches[:50], None, \
                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
            cv2.namedWindow("res", cv2.WINDOW_NORMAL)
            fileText = filename 
            cv2.putText(res, fileText, (-220, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("res", res)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

class ORBandKNNDetection(ORBDetection):
    def __init__(self, select_folder) -> None:
        super().__init__(select_folder)
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)
        search_params=dict(checks=32)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.bf = self.matcher

    def featureExtractRT(self, cls, features):
        ## real time feature extract -> each frame
        dst = []
        print(cls)
        features = [feature for feature in features if feature[-1].split("_")[1] == cls]
        for feature in features:
            kp2, desc2, file, filename = feature # unpacking
            matches = self.bf.knnMatch(self.desc1, desc2, 2)
            ratio = 0.59
            matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * ratio]
            # matches = sorted(matches[:40], key = lambda x:x.distance)
            L = [sum([x.distance for x in matches[:50]]), kp2, file, matches, filename]
            dst.append(L)
        dst = sorted(dst, key=lambda x:x[0], reverse=True)
        if dst[0][0] == 0:
            return None, None
        #print(dst[0][0], dst[1][0], dst[0][0] - dst[1][0] )     
        _, kp2, file, matches, filename = dst[0]
        filename = os.path.basename(filename)
        # print(os.path.basename(filename))
        return dst, filename
