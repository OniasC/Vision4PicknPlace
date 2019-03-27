# -*- coding: utf-8 -*-
import cv2
import numpy as np

def subPixels(px1, px2):
   if px1>px2:
      return px1 - px2
   else:
      return px2 - px1

# read image through command line
img = cv2.imread("difference.jpg")
img2 = cv2.imread("bot-0318.PNG")
img_orig = img2*1 #tenho que multiplicar por 1 para ele nao mudar img_orig qnd mudo img2

# convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# convert the grayscale image to binary image
ret,thresh = cv2.threshold(gray_image, 127,255, cv2.THRESH_BINARY)

# find contours in the binary image
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


#print(contours)
difs = []
for c in contours:
   #print(c)
   # calculate moments for each contour
   M = cv2.moments(c)
   #print(M)
   x,y,w,h = cv2.boundingRect(c)
   difs.append([x,y,w,h])
   cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
   #print(x+w/2,y+h/2)
   cv2.circle(img, (int(x+w/2), int(y+h/2)), 2, (255, 120, 120), -1)

for i in range(len(difs)):
   x,y,w,h = difs[i]
   cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),2)
   cv2.circle(img2, (int(x+w/2), int(y+h/2)), 2, (255, 120, 120), -1)

member=[0]
groups=[member]
flag = 0
for i in range(len(difs)-1):
   rbg_i = img_orig[difs[i][1],difs[i][0]]

   rbg_ii = (img_orig[difs[i+1][1],difs[i+1][0]])
   rbg_m1 = (img_orig[int(difs[i][1]/2 + difs[i+1][1]/2), int(difs[i][0]/2 + difs[i+1][0]/2)])
   rbg_m2 = (img_orig[int(difs[i][1]/2 + difs[i+1][1]/2), int(difs[i][0]/2 + difs[i+1][0]/2)])
   n = 0
   dist = np.sqrt(((difs[i][1]-difs[i+1][1])/(2**n))**2 + ((difs[i][0] - difs[i+1][0])/(2**n))**2)
   while (dist > 5.0):
      n += 1
   
      dist = np.sqrt(((difs[i][1]-difs[i+1][1])/(2**n))**2 + ((difs[i][0] - difs[i+1][0])/(2**n))**2)
      #print(int(difs[i][1]*(2**n-1)/2**n + difs[i+1][1]/2), int(difs[i][0]*(2**n-1)/2**n + difs[i+1][0]/2))
      print("rodando comeco do while")
      rbg_m1 = (img_orig[int(difs[i][1]*(2**n-1)/2**n + difs[i+1][1]/2), int(difs[i][0]*(2**n-1)/2**n + difs[i+1][0]/2)])
      rbg_m2 = (img_orig[int(difs[i][1]/2 + difs[i+1][1]*(2**n-1)/2**n), int(difs[i][0]/2 + difs[i+1][0]*(2**n-1)/2**n)])
      
      print("diff of red color", subPixels(rbg_i[0],rbg_m1[0]))
      if (subPixels(rbg_i[0],rbg_m1[0]) > 25) or (subPixels(rbg_i[1],rbg_m1[1]) > 25) or (subPixels(rbg_i[2],rbg_m1[2]) > 25):
         flag = 1
         print("parou no if", i)
         break
      elif np.abs(rbg_ii[0] - rbg_m2[0]) > 25 or np.abs(rbg_ii[1] - rbg_m2[1]) > 25 or np.abs(rbg_ii[2] - rbg_m2[2]) > 25:
         flag = 1
         print("parou no elif", i)
         break
      print (dist)
   #essa logica ta ruim#
   '''if flag = 0: #they are in the same region
      member.append(i+1)
      groups = [member]
   else: #different regions
      member=[i+1]
      groups.append([member])'''
   #print(img_orig[difs[i][1],difs[i][0]])

outFilename = "difsmarked.jpg"
cv2.imwrite(outFilename, img2)
