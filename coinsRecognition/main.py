import numpy as np
import cv2 as cv

for i in range(1,8):
    img = cv.imread(f'tray{i}.jpg', cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    img = cv.bilateralFilter(img,14,30,30)
    cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
                                param1=50,param2=40,minRadius=0,maxRadius=80)
    circles = np.uint16(np.around(circles))

    max_rad = circles[0,:,2].max()
    min_rad = circles[0,:,2].min()

    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        print(i[2])
        dist_from_max = abs(max_rad - i.item(2))
        dist_from_min = abs(min_rad - i.item(2))
        print(dist_from_min)
        print(dist_from_max)

        if dist_from_max <= dist_from_min:
            cv.putText(cimg,'5zl', (i[0],i[1]),cv.FONT_HERSHEY_SIMPLEX,1, (255,0,0),2,cv.LINE_AA)
        else:
            cv.putText(cimg,'5gr', (i[0],i[1]),cv.FONT_HERSHEY_SIMPLEX,1, (255,0,0),2,cv.LINE_AA)



    cv.imshow('detected circles',cimg)
    cv.waitKey(0)
    cv.destroyAllWindows()