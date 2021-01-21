import cv2
import sys
import numpy as np
import alphashape
import matplotlib.pyplot as plt
from descartes import PolygonPatch
def roi(img1):
    kernel=        [[11,  4, 17,  1,  5],
       [ 6, 14,  0, 12, 16],
       [24, 19, 13, 18, 23],
       [ 7, 11, 11, 10,  5],
       [10, 13, 23,  3,  0]]
    kernel = np.array(kernel,np.float32)/235
    img = cv2.filter2D(img1,-1,kernel)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(gray,(5,5),0)
    ret1, output = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY_INV)
    (conts, _) = cv2.findContours(output, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnt = max(conts, key = cv2.contourArea)
    h, w = img.shape[:2]
    mask1 = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask1, [cnt],-1, 255, -1)
    edges = cv2.Canny(mask,100,200)
    (conts, _) = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnt = max(conts, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    roi=img1[y+15:y+h-15,x+15:x+w-15]
    return ((x,y),roi)
def get_coor(img):
    (x,y),img=roi(img)
    median = cv2.GaussianBlur(image,(5,5),0)
    median = cv2.medianBlur(median,15)
    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    lower_blue=np.array([91,111,68])
    upper_blue=np.array([115,255,255])
    mask=cv2.inRange(hsv,lower_blue,upper_blue)
    edges = cv2.Canny(mask,100,200)
    (contours,_) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #hull = cv2.convexHull(contours,returnPoints = False)
    #print(hull)
    arr=[]
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        arr.append([x,y])
        arr.append([x+w,y])
        arr.append([x,y+h])
        arr.append([x+w,y+h])
    alpha = 0.4 * alphashape.optimizealpha(arr)
    hull = alphashape.alphashape(arr, alpha)
    hull_pts = hull.exterior.coords.xy
    arr=[]
    for i in range(len(hull_pts[0])):
        arr.append([int(hull_pts[0][i]),int(hull_pts[1][i])])
    results=[(np.array(z)+np.array([x,y])).tolist() for z in arr]
    return results
