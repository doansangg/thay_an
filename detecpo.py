from detect_boder import roi
def de(img):
	img=roi(img)[1]
	gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	mask = cv2.GaussianBlur(gray,(5,5),0)
	ret1, output = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY_INV)
	corners = cv2.goodFeaturesToTrack(output, 27, 0.01, 10) 
	corners = np.int0(corners)
	return corners
