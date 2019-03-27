from __future__ import print_function
import cv2
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

#THIS FILE IS TO 
 
def getBlobs(img):

    ret,thresh = cv2.threshold(img, 127,255,0)

    # find contours in the binary image
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
       # calculate moments for each contour
       M = cv2.moments(c)
 
       # calculate x,y coordinate of center
       cX = int(M["m10"] / M["m00"])
       cY = int(M["m01"] / M["m00"])
       cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
       cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow("img", img)
    
    return img
 
if __name__ == '__main__':

    # Read difference image
    refFilename = "difference.jpg"
    #refFilename = "im1.png"
    print("Reading diff image : ", refFilename)
    im1 = cv2.imread(refFilename, cv2.IMREAD_COLOR)
     
    
    time_begin = float(datetime.now().strftime("%S.%f"))
    print("Taking XOR of images ...")
    # Registered image will be resotred in imReg. 
    # The estimated homography will be stored in h. 
    im_blobs = getBlobs(im1)
    time_end_alg = float(datetime.now().strftime("%S.%f"))
      
    # Write aligned image to disk. 
    outFilename = "blobs.jpg"
    print("Saving img with blobs : ", outFilename); 
    cv2.imwrite(outFilename, im_blobs)

    
    plt.imshow(im_blobs)
    plt.show()

    print(time_end_alg-time_begin)
