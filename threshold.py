import cv2
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
 

def applyThresh(im_np, h, w, THRESHOLD):
    #retval, im_np = cv2.threshold(im_np, THRESHOLD, 255, cv2.THRESH_BINARY)
    im_np = cv2.adaptiveThreshold(im_np, 255, cv2.THRESH_BINARY, cv2.ADAPTIVE_THRESH_MEAN_C, 3, 5)
    print(type(im_np))
    #for i in range(h):
    #    for j in range(w):
    #        if (im_np[i,j] < THRESHOLD):
    #            im_np[i,j] = 0
    #        else:
    #            im_np[i,j] = 255
    return im_np
   
def Threshold(img):

    # Convert images to grayscale
    imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(imGray,'gray'),plt.show()

    im_np = np.array(imGray)

    h1, w1 = imGray.shape

    im_np = applyThresh(im_np, h1, w1, 50)

    im_th = im_np
    print("applied thresholds")
    return im_th
 
if __name__ == '__main__':

    # Read image 1
    refFilename = "ppb-2000.PNG"
    print("Reading image 1 : ", refFilename)
    im1 = cv2.imread(refFilename, cv2.IMREAD_COLOR)
     
    # Read image 2
    imFilename = "aligned2.jpg"
    print("Reading image 2 : ", imFilename);  
    im2 = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Applying threshold of images ...")
    # Registered image will be resotred in imReg. 
    # The estimated homography will be stored in h. 
    im_th1 = Threshold(im1)
    plt.imshow(im_th1,'gray'),plt.show()
    cv2.imwrite("im_th1.jpg",im_th1)

    im_th2 = Threshold(im2)
    plt.imshow(im_th2,'gray'),plt.show()
    cv2.imwrite("im_th2.jpg",im_th2)
