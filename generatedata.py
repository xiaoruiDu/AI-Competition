#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 21:24:43 2019

@author: amie
"""

import cv2
import numpy as np



class generator(object):
    def __init__(self):
        pass
    
    def getGpic(self,datapath):
        data = cv2.imread(datapath)
        if data.shape[2] != 1:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        cv2.imshow('original_pic', data)
        #Sobel x,y direction
        edges_x = cv2.Sobel(data,cv2.CV_16S,1,0)
        edges_x = cv2.convertScaleAbs(edges_x)
        cv2.imshow('x_direction', edges_x)
        edges_y = cv2.Sobel(data,cv2.CV_16S,0,1)
        edges_y = cv2.convertScaleAbs(edges_y)
        cv2.imshow('y_direction', edges_y)
        return edges_x,edges_y
    

    def fusion_xy(self, data_x,data_y):
        a = cv2.imread('tespic/1.jpg')
        a[:,:,0] = 0
        a[:,:,1] = 0
        a[:,:,2] = data_y
        cv2.imshow('test',a)
        cv2.waitKey(30)        
        
    def getVpic(self,data1,data2):
        dataV = data2-data1
        cv2.convertScaleAbs(dataV)
        cv2.imshow('V_pic',dataV)
        cv2.waitKey(30)
        

if __name__ == '__main__':
    a = generator()
    datax, datay = a.getGpic('tespic/1.jpg')
#    data1 = cv2.imread('tespic/1.jpg')
#    data1 = cv2.cvtColor(data1, cv2.COLOR_BGR2GRAY)
#    cv2.imshow('original_1', data1)
#    data2 = cv2.imread('tespic/2.jpg')
#    data2 = cv2.cvtColor(data2, cv2.COLOR_BGR2GRAY)
#    cv2.imshow('original_2',data2)
#    a.getVpic(data1,data2)
#    a.fusion_xy(datax,datay)


