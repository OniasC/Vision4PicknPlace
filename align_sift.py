import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread('ppb-2000.PNG',0)          # queryImage
img2 = cv2.imread('ppb-2000-ft2.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    
    #homografia do outro codigo
    #M[0] = [3.31300858, -3.7470437, 823.91527625]
    #M[1] = [ 0.401965549, -2.04959502, 968.741219]
    #M[2] = [ 0.00585532, -0.00643926, 1.0]

    h,w = img1.shape
    pts1 = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts1,M)
    #pts2 =  np.float32(dst[0][0], dst[1][0], dst[2][0], dst[3][0])

    G = cv2.getPerspectiveTransform(dst,pts1)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    im2Reg = cv2.warpPerspective(img2, G, (w, h))

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   #matchesMask = matchesMask, # draw only inliers
                   matchesMask = None,
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img2, 'gray'),plt.show()
cv2.imwrite("aligned2.jpg", im2Reg)

#plt.imshow(img3, 'gray'),plt.show()
cv2.imwrite("align_w_sift.jpg", img3)
