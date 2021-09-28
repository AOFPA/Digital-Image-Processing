import cv2 as cv
import numpy as np
import sys
from matplotlib import pyplot as plt
 
# Global Variable
threshold_value = 180
window_size = 1  #นิยมเป็นเลขคี่ window_size*2 +1
c_value = 10 
source_img = np.zeros((10,10,3), dtype=np.uint8)
adjusted_img = np.zeros((10,10,3), dtype=np.uint8) #global threshold
adjusted_mean_img = np.zeros((10,10,3), dtype=np.uint8) #Local threshold => mean-c algorithm
adjusted_gussian_img = np.zeros((10,10,3), dtype=np.uint8) #global threshold => gaussinWeightSum-c algorithm


hist_img = np.zeros((10,10,3), dtype=np.uint8)
 
def handler_adjustThreshold(x):
    global threshold_value,window_size,c_value
    global source_img,adjusted_img,hist_img,adjusted_mean_img,adjusted_gussian_img
    threshold_value = cv.getTrackbarPos('threshold','Binary')

    #ปรับ window_size ให้เป็นเลขคี่
    window_size = cv.getTrackbarPos('window_size','Binary') * 2 + 1

    # -10 เพื่อให้ค่าตั้งแต่ช่วง -10 ถึง 10 แสดงว่า 10 คือ 0
    c_value = cv.getTrackbarPos('c_value','Binary') - 10

    print(f"Threshold Value = {threshold_value}")
    print(f"window_size = {window_size}")
    print(f"c_value = {c_value}")
   
    _, adjusted_img = cv.threshold(source_img, threshold_value, 255, cv.THRESH_BINARY) # -----------------------------
    #ปรับค่าภาพขาวดำโดยยึดตัวแปร threshold_value เป็นหลักทั้งภาพ หากค่าสีใน pixelใดเกิน threshold_value จะกลายเป็นสีขาว
    #น้อยกว่าก็จะเป็นสีดำ
    adjusted_mean_img = cv.adaptiveThreshold(source_img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,window_size,c_value)
    adjusted_gussian_img = cv.adaptiveThreshold(source_img,255,cv.ADAPTIVE_THRESH_GAUSSIN_C,cv.THRESH_BINARY,window_size,c_value)
    

    # Update histogram
    histSize = 256
    histRange = (0, 256) # the upper boundary is exclusive
    accumulate = False
    gray_hist = cv.calcHist(source_img, [0], None, [histSize], histRange, accumulate=accumulate) # -----------------------------
    hist_w = 512
    hist_h = 400
    bin_w = int(round( hist_w/histSize ))
    hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    cv.normalize(gray_hist, gray_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    for i in range(1, histSize):
        cv.line(hist_img, ( bin_w*(i-1), hist_h - int(gray_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(gray_hist[i]) ),
                ( 255, 0, 0), thickness=2)
    cv.line(hist_img,(threshold_value*2,0),(threshold_value*2,hist_h-1),(255,255,255),3)
 
def main():
    global threshold_value,window_size,c_value
    global source_img,adjusted_img,hist_img,adjusted_gussian_img,adjusted_mean_img
 
    if(len(sys.argv)>=2):
        source_img = cv.imread(str(sys.argv[1]))
    else :
        source_img = cv.imread("./output.png", 1)
 
    source_img = cv.cvtColor(source_img,cv.COLOR_BGR2GRAY) # convert to GrayScale
 
    #named windows
    cv.namedWindow("Original", cv.WINDOW_NORMAL)
    cv.namedWindow("Binary", cv.WINDOW_NORMAL)
    cv.namedWindow("BinaryMean",cv.WINDOW_NORMAL)
    cv.namedWindow("BinaryGaussin",cv.WINDOW_NORMAL)
    cv.namedWindow("Histogram", cv.WINDOW_NORMAL)
 
    #create trackbar
    cv.createTrackbar('threshold', 'Binary', threshold_value, 255, handler_adjustThreshold)
    cv.createTrackbar('window_size', 'Binary', window_size , 20, handler_adjustThreshold)
    cv.createTrackbar('c_value', 'Binary', c_value, 10, handler_adjustThreshold)
 
    adjusted_img  = source_img.copy()

    handler_adjustThreshold(0);
    
    while(True):
        cv.imshow("Original",source_img)
        cv.imshow("Binary",adjusted_img)
        cv.imshow("BinaryMean",adjusted_mean_img)
        cv.imshow("BinaryGaussin",adjusted_gussian_img)
        cv.imshow("Histogram",hist_img)
        key = cv.waitKey(100)
        if(key==27): #ESC = Exit Program
            break
 
    cv.destroyAllWindows()
 
if __name__ == "__main__":
    main()