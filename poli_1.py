import cv2
import sys
import numpy as np
def poli(img1):
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
	# cv2.circle(img, (x,y), 2, [0,0,255], thickness=100)
	# cv2.circle(img, (x+w,y+h), 2, [0,0,255], thickness=100)
	# cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 10)
	roi=img1[y+15:y+h-15,x+15:x+w-15]
	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	mask = cv2.GaussianBlur(gray,(5,5),0)
	ret1, output = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY_INV)
	corners = cv2.goodFeaturesToTrack(output, 27, 0.01, 10) 
	corners = np.int0(corners)
	return output,corners 