from locale import normalize
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt

camera = cv2.VideoCapture(0)
images = [[cv2.imread(file), file] for file in glob.glob('images/RGB/*.jpg')]
def getHist(histb):
    if (len(histb)==2):
        histb, filename = histb
    else:
        filename = None
    histb = cv2.cvtColor(histb, cv2.COLOR_BGR2HSV)
    histb = cv2.calcHist([histb],[0,1],None,[90,256],[1,180, 1,256])
    cv2.normalize(histb, histb, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return histb, filename
hists = list(map(getHist, images))

def getClass(frame):
    h,w,_ = frame.shape
    fr = frame[h//4:360,w//4:480]
    h,w,_ = fr.shape
    cv2.rectangle(fr, (0,0), (w-1, h-1), (0, 0, 255), thickness=2)
    fr_hist,_ = getHist(fr)
    scores = []
    for hist in hists:
        hist, filename = hist
        res_compare = cv2.compareHist(fr_hist,hist,cv2.HISTCMP_BHATTACHARYYA)
        scores.append([res_compare, filename])
    scores = sorted(scores, key=lambda x:x[0])
    filename = scores[0][1]
    return filename, fr

while camera.isOpened():
    ret, frame = camera.read()
    # h, w, _ = frame.shape
    filename, fr = getClass(frame)
    cv2.imshow('original', frame) #show the frame
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

## When everything done, release the capture.
camera.release()
cv2.destroyAllWindows()