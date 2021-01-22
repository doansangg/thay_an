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
def de_hole(img):
    img=roi(img)[1]
    arr_X=[]
    arr_Y=[]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(gray,(5,5),0)
    ret1, output = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(output,100,200,apertureSize = 3)
    minLineLength = 50
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,2,np.pi/180,15,minLineLength=minLineLength,maxLineGap=maxLineGap)
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            if x1==x2:
                arr_X.append([x1,y1])
            if y1==y2:
                arr_Y.append([x1,y1])
    def get_X(box):
        return box[0]
    def get_Y(box):
        return box[1]
    X=sorted(arr_X,key=get_X,reverse=False)
    Y=sorted(arr_Y,key=get_Y,reverse=False)
    (x_min,x_max)=(X[0],X[-1])
    (y_min,y_max)=(Y[0],Y[-1])
    img_r=img[y_min[1]:y_max[1],x_min[0]:x_max[0]]
    return img_r
#     cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("image", 1000, 800)
#     cv2.imshow("image",img_r)
#     cv2.waitKey(0)
    #cv2.imwrite("/home/doan/Documents/data/test_thayan/image11.jpg",img_r)
    
def de_hole1(img):
    img=de_hole(img)
    arr_X=[]
    arr_Y=[]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(gray,(5,5),0)
    ret1, output = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY_INV)
#     cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("image", 1000, 800)
#     cv2.imshow("image",output)
#     cv2.waitKey(0)
    edges = cv2.Canny(output,100,200,apertureSize = 3)
    minLineLength = 50
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,2,np.pi/180,15,minLineLength=minLineLength,maxLineGap=maxLineGap)
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            if x1==x2:
                arr_X.append([[x1,y1],[x2,y2]])
            if y1==y2:
                arr_Y.append([[x1,y1],[x2,y2]])
    def sort_X(box):
        return box[1][1]-box[0][1]
    def sort_Y(box):
        return box[1][0]-box[0][0]
    def get_X(box):
        return box[0][0]
    def get_Y(box):
        return box[1][1]
    X_sort=sorted(arr_X,key=sort_X,reverse=True)
    Y_sort=sorted(arr_Y,key=sort_Y,reverse=True)
    X=sorted(X_sort,key=get_X,reverse=False)
    Y=sorted(Y_sort,key=get_Y,reverse=False)
    X_1=[X_sort[0]]
    for i in X:
        if abs(i[0][0]-X_1[-1][0][0])>10:
            X_1.append(i)
    Y_1=[Y_sort[0]]
    for i in Y:
        if abs(i[1][1]-Y_1[-1][1][1])>10:
            Y_1.append(i)
    X=sorted(X_1,key=get_X,reverse=False)
    Y=sorted(Y_1,key=get_Y,reverse=False)
    Z=[[x[0][0],y[1][1]] for x in X for y in Y]
    X1=[i[0] for i in X]
    for i in X:
        X1.append(i[1])
    Y1=[i[1] for i in Y]
    for i in Y:
        Y1.append(i[0])
    print(len(Y1))
    print(len(X1))
    print(len(Z))
    return (X1,Y1,Z)
def get_coor(image):
    X,Y,Z=de_hole1(image)
    arr_X=[]
    arr_Y=[]
    for z in Z:
        for x in X:
            if z[0]==x[0] or z[1]==z[1]:
                if check(z,x) :
                    arr_X.append(z)
    for z in Z:
        for x in Y:
            if z[0]==x[0] or z[1]==z[1]:
                if check(z,x) :
                    arr_Y.append(z)
    for x in arr:
    if x not in output:
        output.append(x)
    print(output)
    alpha = 0.005 * alphashape.optimizealpha(output)
    hull = alphashape.alphashape(output, alpha)
    hull = hull.exterior.coords.xy
    return hull