import numpy as np
import cv2
from matplotlib import pyplot as plt
from datetime import datetime

img = cv2.imread('top-0318.PNG')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

print(len(gray))
#sift = cv2.xfeatures2d.SIFT_create()
time_begin = float(datetime.now().strftime("%S.%f"))

orb = cv2.ORB_create(nfeatures=1500)
kp, d = orb.detectAndCompute(gray, None)

time_end = float(datetime.now().strftime("%S.%f"))

#kp, d = sift.detectAndCompute(gray, None)
img=cv2.drawKeypoints(gray,kp,img)
print(time_end - time_begin)
cv2.imwrite('orb_keypoints.jpg',img)
plt.imshow(img)
plt.show()
