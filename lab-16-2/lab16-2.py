import cv2 as cv
import numpy as np
 
 
def handler_thresh(x):
    thresh_value = cv.getTrackbarPos('Threshold','motion')
    alpha_percent = cv.getTrackbarPos('Alpha','motion')
    print(f"thresh_value:{thresh_value} / alpha_percent:{alpha_percent}")
def main():
    thresh_value = 4
    alpha_percent = 50
    cv.namedWindow('motion',cv.WINDOW_NORMAL)
    cv.createTrackbar('Threshold', 'motion', thresh_value, 255, handler_thresh)
    cv.createTrackbar('Alpha', 'motion', alpha_percent, 100, handler_thresh)
    vdofile = 'depth.avi'
    cap = cv.VideoCapture(vdofile)
    if not cap.isOpened():
        print("Cannot open vdo")
        exit()
    #frames = [] # contain many frame
    ret, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    background = frame.copy()
    while True:
        thresh_value = cv.getTrackbarPos('Threshold','motion')
        alpha_percent = cv.getTrackbarPos('Alpha','motion')
        alpha = alpha_percent / 100
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # if(len(frames)>20):
        #     del frames[0] # delete First Frame IN
        # # frame differencing
        # ref_frame = 13
      
        #print(len(frames))
        #ให้เพิ่มตัวลด Noise Morphological

        motion_no_thres = np.abs(frame - background)
          #Updating Background for next frame
        #background = (alpha * frame) + ((1-alpha) * background)
        background = cv.addWeighted(frame, alpha, background, (1-alpha), 0) # (alpha * frame) + ((1-alpha) * background)
        ret,motion = cv.threshold(motion_no_thres, thresh_value, 255, cv.THRESH_BINARY)
        cv.imshow('frame', frame)
        cv.imshow('motion', motion)
        cv.imshow('background', background)
        if cv.waitKey(50) == 27:
            break
    cap.release()
 
if __name__ == "__main__":
    main()