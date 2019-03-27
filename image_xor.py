from __future__ import print_function
import cv2
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
 

def filtImage(xor_filtered, xor_img, h1, w1):
#    for i in range(1, h1-1):
#        for j in range(1, w1-1):
#            xor_filtered[i,j] = 1/9*(np.sum(xor_img[i-1:i+1,j-1])+np.sum(xor_img[i-1:i+1,j])+np.sum(xor_img[i-1:i+1,j+1]))
#            #xor_filtered[i,j] = (1/8)*(np.sum(xor_img[i-1:i+1,j])+np.sum(xor_img[i,j-1:j+1]))+(1/4)*xor_img[i,j]
    xor_filtered = cv2.blur(xor_img,(3,3))
    return xor_filtered

def applyThresh(im_np, h, w, THRESHOLD):
    for i in range(h):
        for j in range(w):
            if (im_np[i,j] < THRESHOLD):
                im_np[i,j] = 0
            else:
                im_np[i,j] = 255
    return im_np

def findBlobs(img):
    # convert the image to grayscale
    #gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # convert the grayscale image to binary image
    #ret,thresh = cv2.threshold(gray_image, 127,255, cv2.THRESH_BINARY)
    ret,thresh = cv2.threshold(img, 127,255, cv2.THRESH_BINARY)

    # find contours in the binary image
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.circle(image, (int(x+w/2), int(y+h/2)), 2, (255, 0, 0), -1)
        print(x+w/2,y+h/2)

    return image

   
def xorImages(im1, im2):

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
    im1np = np.array(im1Gray)
    im2np = np.array(im2Gray)

    h1, w1 = im1Gray.shape
    h2, w2 = im2Gray.shape

    #apply first threshold
    im1np = applyThresh(im1np, h1, w1, 120)
    #cv2.imshow("imag", im1np)
    #cv2.waitKey(0)
    im2np = applyThresh(im2np, h2, w2, 120)
    #cv2.imshow("imag", im2np)
    #cv2.waitKey(0)
    print("applied thresholds")
    
    xor_img_org = np.bitwise_xor(im1np,im2np).astype(np.uint8)
    xor_img = np.bitwise_xor(im1np,im2np).astype(np.uint8)
    xor_filtered = xor_img

    #filtering
    print("start filtering")
    time_begin_filt = float(datetime.now().strftime("%H%M%S.%f"))
    #cv2.imshow("imag", xor_filtered)
    #cv2.waitKey(0)
    filtImage(xor_filtered, xor_img, h1, w1)
    
    time_end_filt = float(datetime.now().strftime("%H%M%S.%f"))
    print("applied filter. Took time:")
    print(time_end_filt - time_begin_filt)
    #print(xor_filtered[120])


    #threshold again:
    xor_filtered = applyThresh(xor_filtered, h1, w1, 120)
    #print(xor_filtered[120])

    return xor_filtered, xor_img_org
 
if __name__ == '__main__':

    # Read image 1
    refFilename = "bot-0318.PNG"
    #refFilename = "hallway_320.jpg"
    #refFilename = "im1.png"
    print("Reading image 1 : ", refFilename)
    im1 = cv2.imread(refFilename, cv2.IMREAD_COLOR)
     
    # Read image 2
    imFilename = "bot-0318-editado.png"
    #imFilename = "aligned.jpg"
    #imFilename = "m2.png"
    print("Reading image 2 : ", imFilename);  
    im2 = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    time_begin = float(datetime.now().strftime("%H%M%S.%f"))
    print("Taking XOR of images ...")
    # Registered image will be resotred in imReg. 
    # The estimated homography will be stored in h. 
    imReg = xorImages(im1, im2)
    time_end_alg = float(datetime.now().strftime("%H%M%S.%f"))
      
    # Write aligned image to disk. 
    outFilename = "difference.jpg"
    print("Saving XORed image : ", outFilename); 
    #cv2.imwrite(outFilename, imReg[0])

    # Write aligned image to disk. 
    outFilename = "difference_not_filtered.jpg"
    print("Saving XORed image : ", outFilename); 
    #cv2.imwrite(outFilename, imReg[1])

    image = findBlobs(imReg[0])
    cv2.imshow("image", image)
    cv2.waitKey(0)
    
    print(time_end_alg-time_begin)
