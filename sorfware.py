
import cv2
import numpy as np
filename = 'weights/finalized_model_1.sav'
def load_model(filename):
    model=cv2.face.LBPHFaceRecognizer_create()
    model.read(filename)
    return model
def predict(img,filename):
    model=load_model(filename)
    img = cv2.resize(img,(500, 400))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    predict = model.predict(gray)
    return predict[0]