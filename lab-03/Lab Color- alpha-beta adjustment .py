# Brightness/Contrast Adjustment -> https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
# Trackbar -> https://www.life2coding.com/change-brightness-and-contrast-of-images-using-opencv-python/
# Python -> https://www.tutorialspoint.com/python/python_command_line_arguments.htm
#fullName : Chokchai Jamnoi  , id:162404140008

import cv2 as cv #import library openCv โดยย่อชื่อเล่นเป็น cv
import numpy as np #import library numpy โดยย่อชื่อเล่นว่าเป็น np
import sys #import Library sys ใช้ในการรับ argument ภายนอกจาก cmd (path ของภาพ)
from matplotlib import pyplot as plt #import matplotlib ในส่วนย่อยของ pyplot โดยย่อชื่อเล่นเป็น plt
# Global Variable
beta_brightness_value = 100 #ตัวแปรaBeta เอาไว้ปรับค่า brightness (100 ค่าจริง ๆ แล้วคือ 0 ส่วน 0 คือ -100)
alpha_contrast_value = 10 #ตัวแปร alpha เอาไว้ปรับค่า contrast
source_img = np.zeros((10,10,3), dtype=np.uint8) #array 3D(10*10 ทั้งหมด 3 แผ่น) set ให้ทุก pixel เป็น 0(zeros) // dtype คือ data type // เอาไว้เก็บภาพที่นำเข้ามา 
adjusted_img = np.zeros((10,10,3), dtype=np.uint8) #เอาไว้กับภาพผลลัพธ์ที่ปรับ beta/alpha แล้ว
hist_img = np.zeros((10,10,3), dtype=np.uint8) #เก็บภาพ histogram

def handler_adjustAlphaBeta(x): 
    global beta_brightness_value,alpha_contrast_value #ประกาศตัวแปรเพื่อใช้ตัวแปรใน global
    global source_img,adjusted_img,hist_img #ประกาศตัวแปรเพื่อใช้ตัวแปรใน global
    beta_brightness_value = cv.getTrackbarPos('beta','BrightnessContrast') #เอาค่ามาจาก beta ที่ปรับ เอามาเก็บไว้ใน beta_brightness_value
    alpha_contrast_value = cv.getTrackbarPos('alpha','BrightnessContrast') #เอาค่ามาจาก alpha ที่ปรับ เอามาเก็บไว้ใน alpha_contrast_value
    alpha = alpha_contrast_value / 10 #นำมาหารด้วย 10 เพื่อให้ได้เป็น ทศนิยม 
    beta = int(beta_brightness_value - 100) #นำมาลบด้วย 100 เพื่อให้ได้ค่าที่แท้จริงของ beta (ค่าที่แท้จริง คือ -100 to 100)
    print(f"alpha={alpha} / beta={beta}") #แสดงค่า alpha และ beta ที่ terminal
    
    ## loop access each pixel -> too slow
    ''' for y in range(source_img.shape[0]):
        for x in range(source_img.shape[1]):
            for c in range(source_img.shape[2]):
                adjusted_img[y,x,c] = np.clip( alpha * source_img[y,x,c] + beta , 0, 255)
    '''
    # for better performance, pls use -> dst = cv.addWeighted(src1, alpha, beta, 0.0, src2)
    adjusted_img = cv.convertScaleAbs(source_img, alpha=alpha, beta=beta) #ปรับความสว่าง แล้วนำค่าที่ได้ไปเก็บไว้ใน adjusted_img

    # Update histogram
    bgr_planes = cv.split(adjusted_img) #แยกภาพออกมาเป็น 3 แผ่น (สี 3 แชแนล rgb)
    histSize = 256
    histRange = (0, 256) # the upper boundary is exclusive
    accumulate = False #คือไม่ต้องนำค่ามาสะสม
    #cv.calcHist(ภาพ, [แชแนล 0 คือ สีน้ำเงิน], None, [ขนาดของ histogram], ช่วงของ histogram, accumulate=accumulate)
    b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate) #คำนวณ histogram สีฟ้า
    g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate) #คำนวณ histogram สีเขียว
    r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate) #คำนวณ histogram สีแดง
    hist_w = 512 #ค่ากว้างสุดของ histogram
    hist_h = 400 #ค่าสูงสุดของ histogram
    bin_w = int(round( hist_w/histSize ))
    hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

    #normalize ให้อยู่ในช่วง 0 - 1
    cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

    for i in range(1, histSize): #วนเขียนเส้นกราฟแต่ละเส้น จาก 1 - 256
        cv.line(hist_img, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(b_hist[i]) ),
                ( 255, 0, 0), thickness=2)
        cv.line(hist_img, ( bin_w*(i-1), hist_h - int(g_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(g_hist[i]) ),
                ( 0, 255, 0), thickness=2)
        cv.line(hist_img, ( bin_w*(i-1), hist_h - int(r_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(r_hist[i]) ),
                ( 0, 0, 255), thickness=2)


def main():
    global beta_brightness_value,alpha_contrast_value #ประกาศตัวแปรว่าเราจะใช้ตัวแปรตัวไหน ใน global บ้าง
    global source_img,adjusted_img,hist_img #ประกาศตัวแปรว่าเราจะใช้ตัวแปรตัวไหน ใน global บ้าง
 

    #รับภาพเข้ามา     
    if(len(sys.argv)>=2): #นำเข้าแบบ argument value //len(length) นับความยาว argument ถ้าเท่ากับหรือมากกว่า 2 เข้าเงื่อนไงนี้
        source_img = cv.imread(str(sys.argv[1])) #อ่านภาพจาก argument ตำแหน่งที่ [1]   ชื่อไฟล์.py[0] ชื่อภาพ.jpg[1] แล้วเก็บไว้ในตัวแปร source_img
    else : #ถ้าเราไม่ได้ใส argumrnt ก็ให้เข้าเงื่อนไขนี้
        source_img = cv.imread("./output.png", 1) #อ่านภาพจากไฟล์ output.png เก็บไว้ในตัวแปร source_img

    #source_img = cv.cvtColor(source_img,cv.COLOR_BGR2GRAY) # convert to GrayScale

    #named windows
    #สร้างหน้าต่างเปล่า ๆ ขึ้นมา 3 หน้าต่าง และตั้งชื่อหน้าต่าง
    #window_normal สามมารถ resize ได้
    cv.namedWindow("Original", cv.WINDOW_NORMAL) 
    cv.namedWindow("BrightnessContrast", cv.WINDOW_NORMAL)
    cv.namedWindow("Histogram", cv.WINDOW_NORMAL)

    #create trackbar
    #สร้าง trackbar beta and alpha โดยเอาไปไว้ในหน้าต่าง BrightnessContrast
    #รูปแบบคือ cv.createTrackbar('ชื่อtrackbar', 'หน้าต่างที่จะไปเกาะ', ค่าต่ำสุด, ค่าสูงสุด, เมื่อมี Event ก็จะเรียกใช้ฟังก์ชันนี้)
    cv.createTrackbar('beta', 'BrightnessContrast', beta_brightness_value, 200, handler_adjustAlphaBeta)
    cv.createTrackbar('alpha', 'BrightnessContrast', alpha_contrast_value, 50, handler_adjustAlphaBeta)

    adjusted_img  = source_img.copy() #copy ภาพ source_img ไปไว้ใน adjusted_img (เหมือนการสร้างร่างโคลน)

    while(True): # infinity loop
        cv.imshow("Original",source_img) #แสดงภาพ source_img ในหน้าต่าง Original
        cv.imshow("BrightnessContrast",adjusted_img) #แสดงภาพ adjusted_img ในหน้าต่าง BrightnessContrast
        cv.imshow("Histogram",hist_img) #แสดงภาพ hist_img ในหน้าต่าง Histogram
        #waitKey ต้องอยู่หลัง imshow เสมอ ไม่งั้น imshow จะไม่แสดงหน้าต่างออกมา
        key = cv.waitKey(100) #รอผู้ใช้กด key 100 ms
        if(key==27): #21 = ESC = Exit Program ดูจาก ASCII Table
            break #ให้ออกจาก loop

    cv.destroyAllWindows() #ทำลายหน้าต่างที่เราสร้างขึ้นมา


#ให้เรียกฟังก์ชัน main เป็นหลัก แล้ว func main ค่อยไปเรียกใช้ func อื่น ๆ
if __name__ == "__main__":
    main()