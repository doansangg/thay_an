from sorfware import predict
from PolluteRecognized import Solve,ResizeWithAspectRatio
from function import detect
from detect_boder import find_max,find_2,find_1,roi
import argparse
import os
import cv2
import numpy as np
from poli_1 import poli
# cv2.namedWindow("image", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
# cv2.resizeWindow("image", 1000, 900) 
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--folder", required=True,help="path forder image")
# ap.add_argument("-w", "--weights", required=True,help="path file weigths")
# args = vars(ap.parse_args())
path_folder="/home/doan/Desktop/a"
path_weight="weights/finalized_model_1.sav"
def process(img):
    coordinates_=[]
    predict_class=predict(img,path_weight)
    if(predict_class==1):
        
        # cv2.imshow("image", img1)
        # cv2.waitKey()
        coordinates=poli(img)
        coordinates_.append(coordinates)
    if(predict_class==2):
        coordinates1=detect(img)
        coordinates_.append(coordinates1)
    if(predict_class==3):
        coor=find_max(img)
        coor1=find_1(img)
        coor2=find_2(img)
        coordinates_.append(coor)
        coordinates_.append(coor1)
        coordinates_.append(coor2)
    if (predict_class==0):
        coordinates_=[]
    return (predict_class,coordinates_)
def show(img):
    predict_class,coordinates_=process(img)
    if(predict_class==1): #square
        # img = ResizeWithAspectRatio(img, width=600)
        # img = img[:, :400]
        # blue_color = (0,0, 255)
        # img_draw = img.copy()
        # img_draw = cv2.drawContours(img_draw, coordinates_[0], -1, blue_color, 2)
        img_draw=coordinates_[0][0]
        print(coordinates_[0][1])
        img_draw=cv2.resize(img_draw,(800,800))
        # img_draw = np.hstack([img, img_draw])
        # img_draw=cv2.resize(img_draw,(800,800))
        return img_draw


        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.show()
    if(predict_class==2): #general
        # img1=cv2.resize(img,(800,800))
        clone=coordinates_[0][1]
        for c in coordinates_[0][0]:

            cv2.drawContours(clone, [c], -1, (0,200,0), 3)
        x,y,w,h=coordinates_[0][2]
        # img[y:y+h, x:x+w] = clone
        img=cv2.resize(clone,(800,800))
        return img
    if(predict_class==3): #area

        # img1=cv2.resize(img,(800,800))
        img1=roi(img)[1]
        # cv2.imshow("image",img1)
        # cv2.waitKey(0)
        coor=coordinates_[0]
        coor1=coordinates_[1]
        coor2=coordinates_[2]
        h, w = img1.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        mask1=np.zeros((h, w), np.uint8)
        mask2 = np.zeros((h, w), np.uint8)
        mask3=np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [coor[0]],-1, 255, -1)
        cv2.drawContours(mask1, [coor[1]],-1, 255, -1)
        for cnt in coor1:
            cv2.drawContours(mask2, [cnt],-1, 255, -1)
        for cnt in coor2:
            cv2.drawContours(mask3, [cnt],-1, 255, -1)
        res = cv2.bitwise_and(img1, img1, mask=mask)
        res1 = cv2.bitwise_and(img1, img1, mask=mask1)
        res2 = cv2.bitwise_and(img1, img1, mask=mask2)
        res3 = cv2.bitwise_and(img1, img1, mask=mask3)
        #img[y_min:y_max,x_min:x_max]=res
        res1=cv2.resize(res1,(800,800))
        # cv2.imshow('image',res1)
        # cv2.waitKey(0)
        #img[y_min:y_max,x_min:x_max]=res1
        res2=cv2.resize(res2,(800,800))
        # cv2.imshow('image',res2)
        # cv2.waitKey(0)
        res3=cv2.resize(res3,(800,800))
        #img[y_min:y_max,x_min:x_max]=res2
        # cv2.imshow('image',res3)
        # cv2.waitKey(0)
        res=cv2.resize(res,(800,800))
        img=np.hstack((res1,res))
        img1=np.hstack((res2,res3))
        img=np.vstack((img,img1))
        img=cv2.resize(img,(800,800))
        # cv2.imshow("image",res)
        # cv2.waitKey(0)
        # cv2.imshow("image",res1)
        # cv2.waitKey(0)
        # cv2.imshow("image",res2)
        # cv2.waitKey(0)
        # cv2.imshow("image",res3)
        # cv2.waitKey(0)
        # cv2.imshow("image",img)
        # cv2.waitKey(0)
        #img[y_min:y_max,x_min:x_max]=res3
        # cv2.imshow('image',res)
        # cv2.waitKey(0)
        # image_data =[]
        # image_data.append(res)
        # image_data.append(res1)
        # image_data.append(res2)
        # image_data.append(res3)
        # dst = image_data[0]
        # for i in range(len(image_data)):
        #     if i == 0:
        #         pass
        #     else:
        #         alpha = 1.0 / (i + 1)
        #         beta = 1.0 - alpha
        #         dst = cv2.addWeighted(image_data[i], alpha, dst, beta, 0.0)
        return img
    if(predict_class==0): #wrongImg
        img=cv2.resize(img,(800,800))
        return img

# def test(img):
#     img1 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#     img1 = cv2.Canny(img1,100,200)
#     return img1
#
def main(path_folder):
    for image in os.listdir(path_folder):
        img=cv2.imread(os.path.join(path_folder,image))
        # img = np.asarray(img)
        # print(img.shape)
        show(img)

if __name__ == "__main__":
    main(path_folder)