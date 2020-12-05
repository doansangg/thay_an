import cv2
import imutils
import numpy as np


def detect(image):
    ################################################################ clean border 
    image = cv2.resize(image, (800, 600))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    clone = image[y:y+h, x:x+w] #### ROI

    cv2.drawContours(image, [c], -1, (255, 255, 255), 5)
    # cv2.imshow("clone", clone)

    ################################################################################
    hsv = cv2.cvtColor(clone, cv2.COLOR_BGR2HSV)
    lower_hue = np.array([0,0,0])
    upper_hue = np.array([60,60,100])
    mask1 = cv2.inRange(hsv, lower_hue, upper_hue)
    clone[mask1 == 255] = 0 
    hsv = cv2.cvtColor(clone, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    clone[mask2 == 255] = 0
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    mask3 = cv2.inRange(hsv, lower_blue, upper_blue)
    clone[mask3 == 255] = 0
    lower_white = np.array([23,41,133])
    upper_white = np.array([40,150,255])
    mask4 = cv2.inRange(hsv, lower_white, upper_white)
    clone[mask4 == 255] = 0
    lower_green = np.array([26,56,46])
    upper_green = np.array([95, 255, 255])
    mask5 = cv2.inRange(hsv, lower_green, upper_green)
    clone[mask5 == 255] = 0
    # cv2.imshow("clone2", clone) 
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ##########################################################################################################
    list_contours = []
    blurred_2 = cv2.GaussianBlur(mask2, (3, 3), 0)
    dilated_2 = cv2.dilate(blurred_2, None, iterations=2)
    erosioned_2 = cv2.erode(dilated_2, None, iterations=1)

    gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    dilated = cv2.dilate(thresh, None, iterations=2)
    erosioned = cv2.erode(dilated, None, iterations=1)
    erosioned = erosioned + erosioned_2
    # cv2.imshow("erosioned", erosioned)
    contours = cv2.findContours(erosioned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    for c in contours:
        list_contours.append(c)
        # area = cv2.contourArea(c)
    #     cv2.drawContours(clone, [c], -1, (0,200,0), 3)
    
    # image[y:y+h, x:x+w] = clone
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return list_contours,clone,(x,y,w,h)

# if __name__ == "__main__":
#     image = cv2.imread("/home/doan/Desktop/a/ALuoi_BCv2.jpg")
#     list_contour = detect(image)
#     print(list_contour)
