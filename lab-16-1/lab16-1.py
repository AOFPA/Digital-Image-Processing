import cv2 as cv
import numpy as np


def handler_thresh(x):
    x=1

def main():
    thresh_value = 100
    cv.namedWindow('motion',cv.WINDOW_NORMAL)
    cv.createTrackbar('Threshold', 'motion', thresh_value, 255, handler_thresh)
    # vdofile = 'depth.avi'
    vdofile = 'circle_move.mp4'
    cap = cv.VideoCapture(vdofile)
    if not cap.isOpened():
        print("Cannot open vdo")
        exit()
    frames = [] # contain many frame
    while True:
        thresh_value = cv.getTrackbarPos('Threshold','motion')
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames.append(frame) # add current to history frames
        if(len(frames)>20):
            del frames[0] # delete First Frame IN
        # frame differencing
        ref_frame = 0
        print(len(frames))
        if(len(frames)>ref_frame+1):
            motion_no_thres = np.abs(frame - frames[ref_frame]) #  |current - old[t-xx]| > Th -> is Motion parts
        else :
            motion_no_thres = np.abs(frame - frames[len(frames)-1])
        ret,motion = cv.threshold(motion_no_thres, thresh_value, 255, cv.THRESH_BINARY)
        cv.imshow('frame', frame)
        cv.imshow('motion', motion)
        if cv.waitKey(50) == 27:
            break
    cap.release()

if __name__ == "__main__":
    main()
