import cv2
import numpy as np
from matplotlib import pyplot as plt
import pathlib

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

def DrawContour(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # mask_red1 = cv2.inRange(img_hsv, (170, 100, 20), (180, 255, 255))
    # mask_red2 = cv2.inRange(img_hsv, (0, 100, 20), (10, 255, 255))
    # mask_red = mask_red1 + mask_red2
    # mask_blue = cv2.inRange(img_hsv, (100, 100, 20), (140, 255, 255))
    mask_red = cv2.inRange(img_hsv, (-10, 254, 214), (10, 265, 294))
    mask_blue = cv2.inRange(img_hsv, (97, 245, 214), (117, 265, 294))
    mask = cv2.bitwise_or(mask_red, mask_blue)
    target = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
    # cv2.imshow("target", target)
    # cv2.waitKey()

    imgray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_draw = cv2.drawContours(imgray, contours, -1, (0,0,255), 1)
    # cv2.imshow("img_draw", img_draw)
    # cv2.waitKey()
    return contours

def Solve(im):
    # im = cv2.imread(file_name)
    im = ResizeWithAspectRatio(im, width=600)
    im = im[:, :400]
    contours = DrawContour(im)
    if len(contours) == 0:
        print("doan sang")
    else:
        cont = np.vstack(contours[i] for i in range(len(contours)))
        hull = cv2. convexHull(cont)
        print(hull)
        uni_hull = []
        uni_hull.append(hull)
        return uni_hull
    # black_color = (255, 255, 255)
    # img_draw = im.copy()
    # img_draw = cv2.drawContours(img_draw, uni_hull, -1, black_color, 1)
    # cv2.imshow("img_draw", np.hstack([im, img_draw]))
    # cv2.waitKey()


    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    # print(contours)


# file_name = 'KSKT_ST_04.jpg'
# file_name = '/home/doan/Desktop/a/KSKT_02.jpg'
# file_name = 'KSKT_04.jpg'
# file_name = 'CHA_008.jpg'
# file_name = 'KSKT_ST_06.jpg'
# file_name = 'CHA_001.png'
# file_name = 'CHA_006.png'
# file_name = 'KSKT_11.png' 
# file_name = 'CHA_001.jpg'
# Solve(file_name)


