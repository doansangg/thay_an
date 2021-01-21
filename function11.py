import cv2
import imutils
import numpy as np
import skimage.exposure

def remove_border(image):
  
    img = cv2.resize(image, (800, 600))

    ################################################################ clean border 
    img = cv2.resize(img, (800, 600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    (_, threshold) = cv2.threshold(gradient, 225, 255, cv2.THRESH_BINARY)

    contour = cv2.findContours(threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    c = sorted(contour, key=cv2.contourArea, reverse=True)[0]
    (x, y, w, h) = cv2.boundingRect(c)

    roi = img[y:y+h, x:x+w] #### ROI
    clone = img.copy()

    cv2.drawContours(img, [c], -1, (255, 255, 255), 5)
    return (x, y), img, clone


def detect(image):
    
    X, img, clone = remove_border(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(clone, cv2.COLOR_BGR2HSV)
    lower_white = np.array([23,41,133])
    upper_white = np.array([40,150,255])
    mask5 = cv2.inRange(hsv, lower_white, upper_white)
    img[mask5 == 255] = 0

    ############################### red bounding #############################
    list_contours = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    dilated = cv2.dilate(mask2, None, iterations=3)
    erosioned = cv2.erode(dilated, None, iterations=1)
    contours = cv2.findContours(erosioned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    cnt = max(contours, key=cv2.contourArea)
    list_image1 = get_coor(X, cnt)
    print(list_image1)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        area = cv2.contourArea(c)
        # print(area)
        if area < 1200:
            continue
        list_contours.append(c)
        cv2.drawContours(clone, [c], -1, (0,0,255), 3)

    ################################ Blue bounding #########################
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    mask3 = cv2.inRange(hsv, lower_blue, upper_blue)
    dilated = cv2.dilate(mask3, None, iterations=3)
    erosioned = cv2.erode(dilated, None, iterations=1)
    contours = cv2.findContours(erosioned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    cnt = max(contours, key=cv2.contourArea)
    list_image2 = get_coor(X, cnt)
    print(list_image2)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        area = cv2.contourArea(c)
        # print(area)
        if area < 1200:
            continue
        list_contours.append(c)
        cv2.drawContours(clone, [c], -1, (255,0,0), 3)

    ####################### Green bounding ##################
    lower_green = np.array([26,56,46])
    upper_green = np.array([95, 255, 255])
    mask4 = cv2.inRange(hsv, lower_green, upper_green)
    dilated = cv2.dilate(mask4, None, iterations=3)
    erosioned = cv2.erode(dilated, None, iterations=1)
    contours = cv2.findContours(erosioned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    cnt = max(contours, key=cv2.contourArea)
    list_image3 = get_coor(X, cnt)
    print(list_image3)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        area = cv2.contourArea(c)
        # print(area)
        if area < 1200:
            continue
        list_contours.append(c)
        cv2.drawContours(clone, [c], -1, (0,255,0), 3)

    ################################## black bounding ##################
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_hue = np.array([0,0,50])
    upper_hue = np.array([50,50,100])
    mask1 = cv2.inRange(hsv, lower_hue, upper_hue)
    dilated = cv2.dilate(mask1, None, iterations=3)
    erosioned = cv2.erode(dilated, None, iterations=1)
    contours = cv2.findContours(erosioned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    cnt = max(contours, key=cv2.contourArea)
    list_image4 = get_coor(X, cnt)
    print(list_image4)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        area = cv2.contourArea(c)
        # print(area)
        if area < 1000:
            continue
        list_contours.append(c)
        cv2.drawContours(clone, [c], -1, (0,0,0), 3)

    return clone

def get_coor(X, cnt):
    list_image1 = [i.tolist()[0] for i in cnt]
    list_image1 = [[img[0]+X[0], img[1]+X[1]] for img in list_image1]
    return list_image1


