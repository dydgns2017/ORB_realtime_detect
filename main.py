from lib.dataProcessing import *

if __name__ == "__main__":
    data = CreateDataSet()
    data.createStart() ## 데이터 생성 진행
    ## RGB_folder Canny_folder Sobel_folder LD_folder Lapla_folder HSV_folder GRAY_folder
    detect = ORBDetection(GRAY_folder)
    ## getCanny getLapla getSobel getLD getHSV getGRAY
    # detect.startDetect(getGRAY)