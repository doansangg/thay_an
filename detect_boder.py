import cv2
import sys
import numpy as np
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
def find_max(img11):
    img11=roi(img11)[1]
    kernel=        [[11,  4, 17,  1,  5],
       [ 6, 14,  0, 12, 16],
       [24, 19, 13, 18, 23],
       [ 7, 11, 11, 10,  5],
       [10, 13, 23,  3,  0]]
    kernel = np.array(kernel,np.float32)/225
    img = cv2.filter2D(img11,-1,kernel)
    img = cv2.medianBlur(img,15)
    lower = np.array([48, 0, 0])
    upper = np.array([142, 255, 255])
    lower1 = np.array([49, 32, 0])
    upper1 = np.array([179, 255, 255])
    img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #img1= cv2.cvtColor(median, cv2.COLOR_BGR2LAB)
    #img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #img1=(img+img1)//2
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    kernel = np.ones((5,5), np.uint8) 
    mask1 = cv2.dilate(mask1, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    contours1, hierarchy1 = cv2.findContours(mask1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt1 = max(contours1, key=cv2.contourArea)
    return (cnt,cnt1)

def find_1(img):
    img=roi(img)[1]
    lower=np.array([115,20,0])
    upper=np.array([169,255,255])
    img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #img1= cv2.cvtColor(median, cv2.COLOR_BGR2LAB)
    #img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #img1=(img+img1)//2
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    median = cv2.GaussianBlur(mask,(5,5),0)
    mask = cv2.medianBlur(median,15)
    # Remove hair with opening
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel, iterations = 2)

    # Combine surrounding noise with ROI
    kernel = np.ones((30,30),np.uint8)
    kernel1 = np.ones((29,29),np.uint8)
    dilate = cv2.dilate(opening,kernel,iterations=3)
    img_erosion = cv2.erode(dilate, kernel1, iterations=2) 
    # Blur the image for smoother ROI
    blur = cv2.blur(img_erosion,(50,50))

    # Perform another OTSU threshold and search for biggest contour
    ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if(len(contours)>4):
        contours=contours[:4]
    return contours
def find_2(img):
    img=roi(img)[1]
    lower=np.array([70,48,0])
    upper=np.array([145,255,255])
    img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #img1= cv2.cvtColor(median, cv2.COLOR_BGR2LAB)
    #img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #img1=(img+img1)//2
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    median = cv2.GaussianBlur(mask,(5,5),0)
    mask = cv2.medianBlur(median,15)
    # Remove hair with opening
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel, iterations = 2)

    # Combine surrounding noise with ROI
    kernel = np.ones((26,26),np.uint8)
    kernel1 = np.ones((24,24),np.uint8)
    dilate = cv2.dilate(opening,kernel,iterations=3)
    img_erosion = cv2.erode(dilate, kernel1, iterations=2) 
    # Blur the image for smoother ROI
    blur = cv2.blur(img_erosion,(50,50))

    # Perform another OTSU threshold and search for biggest contour
    ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #cnt = max(contours, key=cv2.contourArea)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if(len(contours)>4):
        contours=contours[:4]
    return contours