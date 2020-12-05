import tkinter as tk
from tkinter import filedialog
from tkinter import  *
from PIL import Image,ImageTk
import os
import cv2
import numpy as np
from main_process import *
class ImageClassifyer(tk.Frame):
    def __init__(self, parent, *args, **kwargs):

        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.filename =""
        self.root = parent
        self.root.wm_title("Find Boundaries")

        # self.next_image()
        claButton = tk.Button(self.root, text='Browser',height=2, width=10, command = self.UploadAction)
        claButton.grid(row=0, column=0, padx=2, pady=2)
        broButton = tk.Button(self.root, text='Next', height=2, width=8, command = self.next_image)
        broButton.grid(row=0, column=1, padx=2, pady=2)

        self.frame1 = tk.Frame(self.root, width=1000, height=1000, bd=2)
        self.frame1.grid(row=1, column=0)
        self.frame2 = tk.Frame(self.root, width=1000, height=1000, bd=1)
        self.frame2.grid(row=1, column=1)


        self.cv1 = tk.Canvas(self.frame1, height=800, width=800, background="white", bd=1, relief=tk.RAISED)
        self.cv1.grid(row=1,column=0)
        self.cv2 = tk.Canvas(self.frame2, height=800, width=800, background="white", bd=1, relief=tk.RAISED)
        self.cv2.grid(row=1, column=0)


    def UploadAction(self):
        self.filename = filedialog.askdirectory()
        print('Selected:', self.filename)
        src = self.filename
        self.list_images = []
        print(src)
        for d in os.listdir(src):
            self.list_images.append(d)
        print(len(self.list_images))
        self.counter = 0
        self.max_count = len(self.list_images) - 1
    def next_image(self):
        if self.counter > self.max_count:
            self.counter = 0
            self.cv1.create_image(0, 0, anchor='nw', image=self.photo)
            self.cv2.create_image(0, 0, anchor='nw', image = self.imgtk)
        else:
            im = Image.open("{}/{}".format(self.filename, self.list_images[self.counter]))
       
            # if (590-im.size[0])<(490-im.size[1]):
            width = 800
            #     height = width*im.size[1]/im.size[0]
            #     self.next_step(height, width)
            # else:
            height = 800
            #     width = height*im.size[0]/im.size[1]
            self.next_step(height, width)

    def next_step(self, height, width):
        self.im = Image.open("{}/{}".format(self.filename, self.list_images[self.counter])).convert('RGB') 
        # img = np.asarray(self.im)
        img=cv2.imread("{}/{}".format(self.filename, self.list_images[self.counter]))
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        print(img.shape)
        img = show(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(int(width) , int(height) ))
        
        img = Image.fromarray(img)

        self.im.thumbnail((width, height), Image.ANTIALIAS)
        self.root.photo = ImageTk.PhotoImage(self.im)
        self.photo = ImageTk.PhotoImage(self.im)


        self.imgtk = ImageTk.PhotoImage(img)
        if self.counter == 0:
            self.cv1.create_image(0, 0, anchor = 'nw', image = self.photo)
            self.cv2.create_image(0, 0, anchor='nw', image=self.imgtk)

        else:
            self.im.thumbnail((width, height), Image.ANTIALIAS)
            self.cv1.delete("all")
            self.cv1.create_image(0, 0, anchor = 'nw', image = self.photo)
            self.cv2.create_image(0, 0, anchor='nw', image=self.imgtk)
        self.counter += 1


if __name__ == "__main__":
    root = tk.Tk()
    MyApp = ImageClassifyer(root)
    tk.mainloop()
