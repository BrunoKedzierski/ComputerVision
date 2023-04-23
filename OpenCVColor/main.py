import cv2 as cv
import numpy as np
cap = cv.VideoCapture('movingball.mp4')

while(1):
    # Take each frame
    _, frame = cap.read()
    print(frame)

    frame = cv.resize(frame,(540,960))
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #Define morphology kernel
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(10,10))

    # define range of red color in HSV
    # I define two color ranges, 0-5 HUE and 175-180 hue
    mask1 = cv.inRange(hsv, (0, 50, 20), (5, 255, 255))
    mask2 = cv.inRange(hsv, (170, 50, 20), (180, 255, 255))
    #combine masks and apply morphology
    mask = cv.bitwise_or(mask1, mask2)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)


    #find contours of the mask and apply them to the frame
    contours,hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


    #compute statistical moments and calculate center of mass
    #try:
    cnt = contours[0]
    M = cv.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    #draw a circle in the center of mass
    frame = cv.circle(frame, (cx, cy), radius=5, color=(0, 255, 0), thickness=-1)
    #except:
        #pass

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)


    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()



