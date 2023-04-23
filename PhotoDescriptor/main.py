import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



def get_orb_matches(img,orb, okp,odes):
    img_kp, img_des= orb.detectAndCompute(img, None)
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=500)

    flann_matcher = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann_matcher.match(img_des, odes)
    return img_kp, img_des, sorted(matches, key=lambda x: x.distance)


def get_sift_matches(img,sift, skp,sdes):
    img_kp, img_des= sift.detectAndCompute(img, None)
    FLANN_INDEX_KDTREE = 1
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(img_des, sdes, k=2)
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append([m])
    return  img_kp,img_des, good


#load reference image
img = cv.resize(cv.imread('saw4.jpg'),(800,600))
ref_img = cv.resize(cv.imread('saw1.jpg'),(800,600))

#img = cv.Laplacian(img,-1)
#ref_img = cv.Laplacian(ref_img,-1)
#create surf/orb objects
orb = cv.ORB_create()

sift = cv.SIFT_create()



#detect ORB
kp_o, des_o = orb.detectAndCompute(img.astype(np.uint8),None)
kp_s, des_s  = sift.detectAndCompute(img.astype(np.uint8), None)



img_plot = cv.drawKeypoints(img, kp_o, None, color=(0,255,0), flags=0)
plt.imshow(img_plot), plt.show()



kp_ref, des_ref, matches_ref =get_orb_matches(ref_img,orb, kp_o, des_o)
img_plot = cv.drawMatches(ref_img,kp_ref,img,kp_o,matches_ref[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img_plot), plt.show()






cap = cv.VideoCapture('sawmovie.mp4')
while(1):
    # Take each frame
    _, frame = cap.read()
    frame = cv.resize(frame,(800,600))
    #frame_l = cv.Laplacian(frame, -1)

    img_kp, img_des, matches = get_orb_matches(frame,orb,kp_o,des_o)

    #frame =  cv.drawMatchesKnn(frame,img_kp,img,kp_o,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    frame =  cv.drawMatches(frame,img_kp,img,kp_o,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow("matches", frame)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()






