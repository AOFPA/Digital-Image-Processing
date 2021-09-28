# Brightness/Contrast Adjustment -> https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
# Trackbar -> https://www.life2coding.com/change-brightness-and-contrast-of-images-using-opencv-python/
# Python -> https://www.tutorialspoint.com/python/python_command_line_arguments.htm
# ใครอ่าน/ฟังอาจารย์แล้วไม่เข้าใจ Code รอบแรกไม่ต้องเครียด ผมเล่น Opencv มาเกือบ 9 ปี แล้วบางคำสั่งยังจำและทำไม่ได้เลย 555+
import cv2
import numpy as np
import sys
 
# Global Variable
beta_brightness_value = 100
alpha_contrast_value = 10
source_img = np.zeros((10,10,3), dtype=np.uint8)
adjusted_img = np.zeros((10,10,3), dtype=np.uint8)
 
 
def handler_adjustAlphaBata(x):
    global beta_brightness_value,alpha_contrast_value
    global source_img,adjusted_img
    beta_brightness_value = cv2.getTrackbarPos('beta','BrightnessContrast')
    alpha_contrast_value = cv2.getTrackbarPos('alpha','BrightnessContrast')
    alpha = alpha_contrast_value / 10
    beta = int(beta_brightness_value - 100)
    print(f"alpha={alpha} / beta={beta}")
    # for better performance, pls use -> dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    for y in range(source_img.shape[0]):
        for x in range(source_img.shape[1]):
            for c in range(source_img.shape[2]):
                adjusted_img[y,x,c] = np.clip( alpha * source_img[y,x,c] + beta , 0, 255)
 
 
 
def main():
    global beta_brightness_value,alpha_contrast_value
    global source_img,adjusted_img
 
    if(len(sys.argv)>=2):
        source_img = cv2.imread(str(sys.argv[1]))
    else :
        source_img = cv2.imread("./output.png", 1)
 
    #named windows
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("BrightnessContrast", cv2.WINDOW_NORMAL)
    #cv.namedWindow("Histogram", cv2.WINDOW_NORMAL)
 
    #create trackbar
    cv2.createTrackbar('beta', 'BrightnessContrast', beta_brightness_value, 200, handler_adjustAlphaBata)
    cv2.createTrackbar('alpha', 'BrightnessContrast', alpha_contrast_value, 50, handler_adjustAlphaBata)
 
    adjusted_img  = source_img.copy()
 
    while(True):
        cv2.imshow("Original",source_img)
        cv2.imshow("BrightnessContrast",adjusted_img)
        key = cv2.waitKey(100)
        if(key==27): #ESC = Exit Program
            break
 
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main()