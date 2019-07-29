#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 17:34:28 2019

@author: amie
"""



import picdata as pic
import numpy as np
from numba import jit
import time 
import math
import matplotlib.pyplot as plt
#from pca import PCA_face_example
import pickle
import cv2
import os



class picprocess():
    def __init__(self):
        self.strain = 1
        self.sN_NEURONS = 100
        self.sN_NEURONS_1 = 100
        self.sETA = 1.0
        self.sXI = 1.0
        self.strain_round = 100
        self.strainpath = 'trainpicpath.txt'
        self.stestpath = 'testpicpath.txt'
        value = self.readconfig('config_FGV.txt')        
        self.strain = int(value[0])
        self.sN_NEURONS = int(value[1])
        self.sN_NEURONS_1 = int(value[2])
        self.sETA = float(value[3])
        self.sXI = float(value[4])
        self.strain_round = int(value[5])
        self.strainpath =value[6]
        self.stestpath = value[7]
        
        
        
    def generateGxGyVF(self):
        Gx = []
        Gy = []
        F = []
        V = []
        for fpath,dirname,file in os.walk('pic_Gx'):
            for f in file:
                f = 'pic_Gx/' + f
                with open(f,'rb') as outfile:
                    data = pickle.load(outfile)
                    Gx.append(data)
                    
        for fpath,dirname,file in os.walk('pic_Gy'):
            for f in file:
                f = 'pic_Gy/' + f
                with open(f,'rb') as outfile:
                    data = pickle.load(outfile)
                    Gy.append(data)
                    
        for fpath,dirname,file in os.walk('pic_F'):
            for f in file:
                f = 'pic_F/' + f
                with open(f,'rb') as outfile:
                    data = pickle.load(outfile)
                    F.append(data)
        for fpath,dirname,file in os.walk('pic_V'):
            for f in file:
                f = 'pic_V/' + f
                with open(f,'rb') as outfile:
                    data = pickle.load(outfile)
                    V.append(data)        
        
        
        return Gx,Gy,F,V
    
    
    
    def generatetraindata(self,Gx,Gy,F,V):
        Gxyv = []
        Fxy = []
        for i in range(len(Gx)):
            print('Gxshape:',Gx[i].shape)
            print('Fshape:',F[i].shape)
            for m in range(Gx[i].shape[0]):
                for n in range(Gx[i].shape[1]):
                    gxyv = [Gx[i][m,n],Gy[i][m,n],V[i][m,n]]
                    fxy = [F[i][m,n,0],F[i][m,n,1]]
                    Gxyv.append(gxyv)
                    Fxy.append(fxy)
        Gxyv = np.array(Gxyv)
        Fxy = np.array(Fxy)
                    
        return Gxyv,Fxy
        
        
        
    def readconfig(self,configpath):
        value = []
        with open(configpath,'r') as config:
            for line in config.readlines():
                line = line.split('=')
                value.append(line[1].split('\n')[0])
        return value
            
    
    def cosine_dis(self, x, y):
        num =  (x*y).sum(axis=1)
        denom = np.linalg.norm(x) * np.linalg.norm(y,axis=1)
        return num/denom
    
    def predict_corner(self):
        
        # initialize the parameters
        population_a = np.zeros((self.sN_NEURONS,1))  
        population_s = np.ones((self.sN_NEURONS,1))*0.045
        wcross = np.random.uniform(0,1,(self.sN_NEURONS,self.sN_NEURONS_1))  
        population_Wcross = wcross / wcross.sum() 
        population_Winput = np.random.random((self.sN_NEURONS,3))/10.0    
        population_Winput_1 = np.random.random((self.sN_NEURONS_1,2))/10.0    
        
        
        # load the model
        with open('Weight data/train_vgf_300_1000/populations_Wcross799.pkl','rb') as file:
            population_Wcross = pickle.load(file)
        
        with open('Weight data/train_vgf_300_1000/populations_Winput799.pkl','rb') as file1:
            population_Winput = pickle.load(file1)
            
        with open('Weight data/train_vgf_300_1000/population_Winput_1799.pkl','rb') as file2:
            population_Winput_1 = pickle.load(file2)
            
        with open('Weight data/train_vgf_300_1000/populations_s799.pkl','rb') as file3:
            population_s = pickle.load(file3)
          
        # show the HL matrix
        plt.imshow(population_Wcross)
        cap = cv2.VideoCapture('slow_traffic_small.mp4')

        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )        
        color = np.random.randint(0,255,(100,3))

        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        p_time = p0[:,0,:]
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        count = 1
        while(1):
            sensory_x = []
            count+=1
            ret,frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dx = cv2.Sobel(old_gray,cv2.CV_16S,1,0)
            dy = cv2.Sobel(old_gray,cv2.CV_16S,0,1)
            VI = frame_gray - old_gray
            
            dx = cv2.resize(dx,(6400,3600))
            dy = cv2.resize(dy,(6400,3600))
            VI = cv2.resize(VI,(6400,3600))    
            
            p0 = p0[:,0,:]
            good_old_around = (p0 * 10).astype(np.int64)
            for i in range(len(good_old_around)):
                if good_old_around[i][1] >= 3600:
                    good_old_around[i][1] = 3599
                if good_old_around[i][0] >= 6400:
                    good_old_around[i][1] = 6399
                    
                x = dx[(good_old_around[i][1]),(good_old_around[i][0])]
                y = dy[(good_old_around[i][1]),(good_old_around[i][0])]
                vi = VI[(good_old_around[i][1]),(good_old_around[i][0])]*4
                sensory_x.append(np.array([x,y,vi])/1020.0)

            sensory_x = np.array(sensory_x) 
            act_cur1 = np.zeros((100,1))
            x_drection = []            
            for i in range(sensory_x.shape[0]):
                input_sample = sensory_x[i].reshape(1,-1) 
                temp = (np.power((input_sample - population_Winput),2).sum(axis=1)/100).reshape(-1,1) 
                act_cur1 = (1/(np.sqrt(2*np.pi)*population_s))*np.exp(-temp/(2*np.power(population_s,2))) 
                act_cur_sum = act_cur1.sum()
                if act_cur_sum == 0 :
                    print('act_cur.sum() is less than 1e-323,ignore the update!')
                act_cur1 = act_cur1 / act_cur_sum
                population_a = act_cur1
                # get the position of winner neuron
                win_pos = population_a[:,0].argmax()
                pre_pos = population_Wcross[win_pos,:].argmax()
                # decode the HL matrix
                a1 = population_Winput_1[pre_pos]
                x_drection.append(a1)
            x_drection = np.array(x_drection)
            good_old = p_time
            good_new = p_time + x_drection
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel().astype(np.float32)
                c,d = old.ravel().astype(np.float32)
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            img = cv2.add(frame,mask)
            # rebuild the optical flow picture
            cv2.imshow('rebuild flow',img)
            cv2.waitKey(100)
            p_time = good_new
            p0 = good_new[:,np.newaxis,:]
            

    def calmove(self,coordiante,coordinate_new):# [x1,y1],[float1,float2]
        co = []
        
        x0y0 = [coordiante[0] - 1, coordiante[1] - 1]
        x0y1 = [coordiante[0] - 1, coordiante[1]]
        x0y2 = [coordiante[0] - 1, coordiante[1] + 1]
        x1y0 = [coordiante[0] , coordiante[1] - 1]
        x1y1 = [coordiante[0] , coordiante[1]]
        x1y2 = [coordiante[0] , coordiante[1] + 1]
        x2y0 = [coordiante[0] + 1, coordiante[1] - 1]
        x2y1 = [coordiante[0] + 1, coordiante[1]]
        x2y2 = [coordiante[0] + 1, coordiante[1] + 1]
        
        co.append(x0y0)
        co.append(x0y1)
        co.append(x0y2)
        co.append(x1y0)
        co.append(x1y1)
        co.append(x1y2)
        co.append(x2y0)
        co.append(x2y1)
        co.append(x2y2)
        
        co = np.array(co)
        x1y1 = np.array(coordinate_new)
        
        co_x1y1 = co -x1y1
        co_x1y1 = np.power(co_x1y1,2)
        co_x1y1_sum = np.sum(co_x1y1,axis=1)
        posmax = co_x1y1_sum.argmin()
        print(co[posmax])
        
        
        return co[posmax]
        
    
            
    def parametrize_learning_law(self, v0, vf, t0, tf):
        y = np.zeros((tf-t0,1))
        t = [i for i in range(1,tf+1)]
        B = (vf*tf - v0*t0)/(v0 - vf)
        A = v0*t0 + B*v0
        y = [A/(t[i]+B) for i in range(len(t))]
        return y

    

    def speed_up_som(self):
        
        # get the training data
        Gx,Gy,F,V = a.generateGxGyVF()
        gxyv,fxy = a.generatetraindata(Gx,Gy,F,V)
        DxyvUxy = np.zeros((8720,5))
        with open('DxyvUxy.pkl','rb') as file:
            load = pickle.load(file)
            DxyvUxy = np.array(load)

        # Normalize the data
        sensory_x = DxyvUxy[:,0:3] / 1020.0
        sensory_y = DxyvUxy[:,3:5] 

        # initialize the parameters
        N_NEURONS  = self.sN_NEURONS   # sensor1 
        N_NEURONS_1  = self.sN_NEURONS_1   # sensor2 
        population_s = np.ones((N_NEURONS,1))*0.045   # sensor1 tuning curve
        population_a = np.zeros((N_NEURONS,1))        # sensor1 activation value
        wcross = np.random.uniform(0,1,(N_NEURONS,N_NEURONS_1))
        population_Wcross = wcross / wcross.sum()      # sensor1 HL matrix
        train_round = self.strain_round                            
        population_Winput = np.random.random((N_NEURONS,sensory_x.shape[1]))/100.0    # sensor1 weight 
        sample_num = sensory_x.shape[0]   
        sample_demension = sensory_x.shape[1] 
        learning_sigmat = self.parametrize_learning_law(50,1,1,train_round) 
        learning_alphat = self.parametrize_learning_law(0.1,0.001,1,train_round)
        ETA = 1.0 
        XI = 1e-3
        hwi = np.zeros((N_NEURONS,1))   
 
        population_s_1 = np.ones((N_NEURONS_1,1))*0.045   # sensor2 tuning curve
        population_a_1 = np.zeros((N_NEURONS_1,1))        # sensor1 activation value
        wcross_1 = np.random.uniform(0,1,(N_NEURONS_1,N_NEURONS))
        population_Wcross_1 = wcross_1 / wcross_1.sum()   # sensor2 HL matrix
        print(sensory_y.shape)
        population_Winput_1 = np.random.random((N_NEURONS_1,sensory_y.shape[1]))/100.0    # sensor1 weight 
        sample_num_1 = sensory_y.shape[0]   
        sample_demension_1 = sensory_y.shape[1]  
        ETA = 1.0  
        XI = 1e-3  
        hwi_1 = np.zeros((N_NEURONS_1,1))  
        hl_trainround = 100
        avg_act = np.zeros((N_NEURONS,1)) 
        avg_act_1 = np.zeros((N_NEURONS_1,1)) 
        
        # training  
        for t in range(hl_trainround + train_round): 
            if t < train_round:  
                for sample_index in range(sample_num): 
                    
                    act_cur1 = np.zeros((N_NEURONS,1))
                    act_cur2 = np.zeros((N_NEURONS_1,1))
                
                    
                    input_sample = sensory_x[sample_index].reshape(1,-1) 
                    input_sample_2 = sensory_y[sample_index].reshape(1,-1)

                    temp = (np.power((input_sample - population_Winput),2).sum(axis=1)/sample_demension).reshape(-1,1)  
                    temp1 = (np.power((input_sample_2 - population_Winput_1),2).sum(axis=1)/sample_demension_1).reshape(-1,1)
                    
                    # matrix calculate.All activation values are updated together
                    act_cur1 = (1/(np.sqrt(2*np.pi)*population_s))*np.exp(-temp/(2*np.power(population_s,2)))
                    act_cur2 = (1/(np.sqrt(2*np.pi)*population_s_1))*np.exp(-temp1/(2*np.power(population_s_1,2)))
                    
                    act_cur_sum = act_cur1.sum()
                    act_cur_sum1 = act_cur2.sum()
                    
                    if act_cur_sum == 0 or act_cur_sum1 == 0:
                        print('act_cur.sum() is less than 1e-323,ignore the update!')
                        continue
                    act_cur1 = act_cur1 / act_cur_sum
                    act_cur2 = act_cur2 / act_cur_sum1
                    
                    population_a = (1-ETA)*population_a + ETA * act_cur1
                    population_a_1 = (1-ETA)*population_a_1 + ETA * act_cur2
                    
                    win_pos = population_a[:,0].argmax()
                    win_pos1 = population_a_1[:,0].argmax()
                    
                    pos_list = np.arange(0,N_NEURONS,1)
                    pos_list_1 = np.arange(0,N_NEURONS_1,1)
                    
                    hwi = (np.exp(-np.power(pos_list - win_pos, 2) / (2 * np.power(learning_sigmat[t],2)))).reshape(N_NEURONS,1)
                    hwi_1 = (np.exp(-np.power(pos_list_1 - win_pos1, 2) / (2 * np.power(learning_sigmat[t],2)))).reshape(N_NEURONS_1,1)
                    
                    # matrix calculate.All population_Winput values are updated together
                    population_Winput = population_Winput+ \
                    learning_alphat[t] * hwi * (input_sample - population_Winput)
                    
                    population_Winput_1 = population_Winput_1+ \
                    learning_alphat[t] * hwi_1 * (input_sample_2 - population_Winput_1)              
                                
                    # matrix calculate.All population_s values are updated together
                    temp_s = (np.power((input_sample - population_Winput),2).sum(axis=1)/sample_demension).reshape(-1,1)
                    population_s = population_s + \
                    learning_alphat[t] *  (1/(np.sqrt(2*np.pi)*learning_sigmat[t])) * \
                    hwi * (temp_s - np.power(population_s,2))
        
                    temp_s_1 = (np.power((input_sample_2 - population_Winput_1),2).sum(axis=1)/sample_demension_1).reshape(-1,1)
                    population_s_1 = population_s_1 + \
                    learning_alphat[t] *  (1/(np.sqrt(2*np.pi)*learning_sigmat[t])) * \
                    hwi_1 * (temp_s_1 - np.power(population_s_1,2))
                    
            print('training:',t/(train_round+hl_trainround))	                    
           
            # HL matrix training 
            for sample_index in range(sample_num):

                act_cur1 = np.zeros((N_NEURONS,1))
                act_cur2 = np.zeros((N_NEURONS_1,1))
                
                input_sample = sensory_x[sample_index].reshape(1,-1) 
                input_sample_2 = sensory_y[sample_index].reshape(1,-1)
                
                temp = (np.power((input_sample - population_Winput),2).sum(axis=1)/sample_demension).reshape(-1,1)  
                temp1 = (np.power((input_sample_2 - population_Winput_1),2).sum(axis=1)/sample_demension_1).reshape(-1,1)
                
                # matrix calculate. All activation values are updated together
                act_cur1 = (1/(np.sqrt(2*np.pi)*population_s))*np.exp(-temp/(2*np.power(population_s,2)))
                act_cur2 = (1/(np.sqrt(2*np.pi)*population_s_1))*np.exp(-temp1/(2*np.power(population_s_1,2)))
                
                
                act_cur_sum = act_cur1.sum()
                act_cur_sum1 = act_cur2.sum()
                if act_cur_sum == 0 or act_cur_sum1 == 0:
                    print('act_cur.sum() is less than 1e-323,ignore the update!')
                    continue
                act_cur1 = act_cur1 / act_cur_sum
                act_cur2 = act_cur2 / act_cur_sum1
                
                population_a = (1-ETA)*population_a + ETA * act_cur1
                population_a_1 = (1-ETA)*population_a_1 + ETA * act_cur2
                
                OMEGA = 0.002 + 0.998/(t+2)
                avg_act[:,0] = (1-OMEGA)*avg_act[:, 0] + OMEGA*population_a[:,0]
                avg_act_1[:,0] = (1-OMEGA)*avg_act_1[:, 0] + OMEGA*population_a_1[:,0]
                
                population_Wcross = (1-XI)*population_Wcross + XI*(population_a - avg_act[:, 0].reshape(N_NEURONS,1))*(population_a_1 - avg_act_1[:, 0].reshape(N_NEURONS_1,1)).T

            if t%200 == 199:
                # save the model
                with open('populations_Wcross{}.pkl'.format(t),'wb') as output:
                    pickle.dump(population_Wcross,output)
                with open('populations_Winput{}.pkl'.format(t),'wb') as output1:
                    pickle.dump(population_Winput,output1)
                with open('population_Winput_1{}.pkl'.format(t),'wb') as output2:
                    pickle.dump(population_Winput_1,output2)   
                    
                with open('populations_s{}.pkl'.format(t),'wb') as output3:
                    pickle.dump(population_s,output3)
                with open('populations_s_1{}.pkl'.format(t),'wb') as output4:
                    pickle.dump(population_s_1,output4)

if __name__ == '__main__':
    a = picprocess()
    start = time.time()
    if a.strain == 1:
        a.speed_up_som()
    else:
        a.predict_corner()
    print(time.time() - start)





















