#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 22:28:30 2019

@author: amie
"""


import cv2
import os
import numpy as np
import pickle




class picprocess(object):
    def __init__(self):
        pass
    
    
    def pix_singlechannel_cellload(self,picpath,kernel_size):
        databoxe = []
        data_xdirection = []
        data_ydirection = []
        with open(picpath) as file:
            for line in file.readlines():
                if line == '\n':
                    continue
                line = line.strip()
                pic = cv2.imread(line)
                pic_width,pic_high,i_channel = pic.shape
                if i_channel != 1:
                    pic = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
                a = cv2.Sobel(pic,cv2.CV_16S,1,0)
                a = cv2.convertScaleAbs(a)
                cv2.imshow('expected_picture',a)
                edges_x = cv2.Sobel(pic,cv2.CV_16S,1,0)      
                edges_y = cv2.Sobel(pic,cv2.CV_16S,0,1)

                for k in range(1,pic_width-1,1):
                    for m in range(1,pic_high-1,1):
                        data_xdirection.append(edges_x[k,m])
                        data_ydirection.append(edges_y[k,m])
                for i in range(pic_width - kernel_size[0] + 1):
                    for j in range(pic_high - kernel_size[1] + 1):
                        data = []
                        for m in range(kernel_size[0]):
                            for n in range(kernel_size[1]):
                                data.append(pic[i+m,j+n])
                        databoxe.append(data)
        databoxe = np.array(databoxe)
        data_xdirection = np.array(data_xdirection)
        data_ydirection = np.array(data_ydirection)
        
        return databoxe,data_xdirection,data_ydirection


    #convert list to numpy to show the Gx or Gy picture.
    def createGpic(self,datalist,picsize):
        pic = np.zeros((picsize[0],picsize[1]))
        k = 0
        for i in range(picsize[0]):
            for j in range(0,picsize[1]):
                pic[i,j] = datalist[k]
                k += 1
        pic = cv2.convertScaleAbs(pic)
        cv2.imshow('rebuild_picture',pic)
        cv2.waitKey(30)
        
        
        
    def rebuildflow(self,flow):
        rebuild_flow = np.zeros((120,213,2))
        count = 0
        for m in range(120):
            for n in range(213):
                rebuild_flow[m,n,1] = flow[count][0]
                rebuild_flow[m,n,0] = flow[count][1]
                count += 1
        return rebuild_flow
        
    def showFlowpic(self,flow):
        cap = cv2.VideoCapture("slow_traffic_small.mp4")
        
        ret, frame1 = cap.read()
        frame1 = cv2.resize(frame1,(213,120),0,0)

        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        print(rgb.shape)
    
        cv2.imshow('rebuild flow',rgb)
        cv2.waitKey(1000)
        
    def showexpectedflowpic(self,pklpath):
        cap = cv2.VideoCapture("slow_traffic_small.mp4")
        
        ret, frame1 = cap.read()
        frame1 = cv2.resize(frame1,(213,120),0,0)
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        with open(pklpath,'rb') as outfile:
            flow = pickle.load(outfile)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            print(rgb.shape)
        
            cv2.imshow('expected flow',rgb)
            k = cv2.waitKey(1000) & 0xff
                    
    
      
    def pix_rgb_cellload(self,pic,kernel_size):
#        pic = cv2.imread(picpath)
        pic_R = pic[:,:,2]
        pic_G = pic[:,:,1]
        pic_B = pic[:,:,0]        
        pic_dataset_R = self.pix_singlechannel_cellload(pic_R, kernel_size)
        pic_dataset_G = self.pix_singlechannel_cellload(pic_G, kernel_size)
        pic_dataset_B = self.pix_singlechannel_cellload(pic_B, kernel_size)
        return pic_dataset_R,pic_dataset_G,pic_dataset_B
        
        
    def get_picadata(self, picpath):
        
        datalist = []
        with open(picpath) as file:
            for line in file:
                line = line.split('\n')[0]
                img = cv2.imread(line)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_flat = img.flatten()  
                datalist.append(img_flat)
        datalist = np.array(datalist)
        return datalist
                

    def get_img_list(self,txtpath):
        for root, dirs, files in os.walk(txtpath):
            with open('pic_path_train.txt','a') as f:
                for name in files:
                    f.write(txtpath + name + '\n')
                
        
        
        

if __name__ == '__main__':
    p = picprocess()
    pic = cv2.imread('test_data/Facebook.jpg')
#    p.pix_rgb_cellload(pic,[32,32])
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray_image',pic)
    print(pic.shape)
    pic_box = p.pix_singlechannel_cellload('testpicpath.txt',[3,3])  # [sizex - 2,sizey - 2]
#    p.get_img_list('train/')
#    p.get_picadata('pic_path1.txt')
    
    
    
    
    
    
    
    
    
        


