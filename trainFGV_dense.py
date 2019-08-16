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
 

    def predict_vgf(self):
        
        # initialize the parameters
        pic_data = pic.picprocess()
        population_a = np.zeros((self.sN_NEURONS,1))  
        population_s = np.ones((self.sN_NEURONS,1))*0.045
        wcross = np.random.uniform(0,1,(self.sN_NEURONS,self.sN_NEURONS_1))  
        population_Wcross = wcross / wcross.sum() 
        population_Winput = np.random.random((self.sN_NEURONS,3))/10.0    
        population_Winput_1 = np.random.random((self.sN_NEURONS_1,2))/10.0    
        
        
        # load the test picture
#        Gx,Gy,F,V = self.generateGxGyVF()
        Gx,Gy,F,V =  self.loadGxyvUxy('training_data/video/pic_ori(250_250_uniform_motion_stablescence)_version2/DxyvUxy/train_data196.pkl')
        gxyv,fxy = self.generatetraindata(Gx,Gy,F,V)
        Gx1,Gy1,F1,V1 = self.loadGxyvUxy('training_data/video/pic_ori(250_250_(simple)_one_move_white_backgroud_xdirection)/DxyvUxy/train_data12.pkl')
        gxyv1,fxy1 = self.generatetraindata(Gx1,Gy1,F1,V1)
    
        sensory_x = gxyv / 255.0
        sensory_x1 = gxyv / 255.0
        
        # load the model
#        with open('populations_Wcross399.pkl','rb') as file:
#            population_Wcross = pickle.load(file)
#            
#        with open('populations_Winput399.pkl','rb') as file1:
#            population_Winput = pickle.load(file1)
#            
#        with open('population_Winput_1399.pkl','rb') as file2:
#            population_Winput_1 = pickle.load(file2)
#            
#        with open('populations_s399.pkl','rb') as file3:
#            population_s = pickle.load(file3)
            
        with open('weights/pic_ori(250_250_(simple)_one_move_white_backgroud_xdirection)/populations_Wcross599.pkl','rb') as file:
            population_Wcross = pickle.load(file)
            
        with open('weights/pic_ori(250_250_(simple)_one_move_white_backgroud_xdirection)/populations_Winput599.pkl','rb') as file1:
            population_Winput = pickle.load(file1)
            
        with open('weights/pic_ori(250_250_(simple)_one_move_white_backgroud_xdirection)/population_Winput_1599.pkl','rb') as file2:
            population_Winput_1 = pickle.load(file2)
            
        with open('weights/pic_ori(250_250_(simple)_one_move_white_backgroud_xdirection)/populations_s599.pkl','rb') as file3:
            population_s = pickle.load(file3)        
        
        pic_data.draw2Dpic(population_Winput_1,fxy1)
        pic_data.draw3Dpic(population_Winput,sensory_x1)

        
        # show the HL matrix
        
        fig2 = plt.figure('HL matrix')
        plt.imshow(population_Wcross)
        
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
            
        x_drection_np = np.array(x_drection)       
        # rebuild the optical flow picture 
        flow = pic_data.rebuildflow(x_drection_np)
        
        ori = cv2.imread('training_data/video/pic_ori(250_250_uniform_motion_stablescence)_version2/196.jpg')
        ori_2 = cv2.imread('training_data/video/pic_ori(250_250_uniform_motion_stablescence)_version2/197.jpg')
        
        add_frame1_frame2 = cv2.addWeighted(ori,0.5,ori_2,0.5,0)
        cv2.imshow('frame1_frame2',add_frame1_frame2)
        
        cv2.imshow('frame1',ori)
        mask = np.zeros_like(ori)
        print(mask.shape)
        cont = 1
        
        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                if abs(flow[...,0][row,col]) >= 1 or abs(flow[...,1][row,col]) >=1:
                    x_max = int(col + round(flow[...,0][row,col]))
                    y_max = int(row + round(flow[...,1][row,col]))
                    if x_max < mask.shape[1] and x_max >= 0 and y_max < mask.shape[0] and y_max >= 0:
                        ori[y_max,x_max,0] = 255
                        ori[y_max,x_max,1] = 0
                        ori[y_max,x_max,2] = 0
                        if ori[row,col,0] == 255 and ori[row,col,1] == 0 and ori[row,col,2] == 0:
#                            
                            ori[row,col,0] = 0
                            ori[row,col,1] = 0
                            ori[row,col,2] = 255
                        else:
                            ori[row,col,0] = 0
                            ori[row,col,1] = 255
                            ori[row,col,2] = 0
                        cont+=1
        print(cont)
        cv2.imwrite('Optical_Flow.jpg',ori)
        cv2.imshow('Overlap_Optical_flow',ori)
        # rebuild the optical flow picture 
        pic_data.showFlowpic(flow)
        pic_data.showexpectedflowpic('training_data/video/pic_ori(250_250_uniform_motion_stablescence)_version2/DxyvUxy/train_data196.pkl')


    def parametrize_learning_law(self, v0, vf, t0, tf):
        y = np.zeros((tf-t0,1))
        t = [i for i in range(1,tf+1)]
        B = (vf*tf - v0*t0)/(v0 - vf)
        A = v0*t0 + B*v0
        y = [A/(t[i]+B) for i in range(len(t))]
        return y
    
    
    def loadGxyvUxy(self,pklpath):
        Gx = []
        Gy = []
        F = []
        V = []
        
        with open(pklpath,'rb') as outfile:
            DxyvUxy = pickle.load(outfile)
        Gx.append(DxyvUxy[0][0])
        Gy.append(DxyvUxy[0][1])
        F.append(DxyvUxy[0][3])
        V.append(DxyvUxy[0][2])
        
        return Gx,Gy,F,V

    def speed_up_som(self):
        
#        train_da =  self.loadGxyvUxy('video/DxyvUxy/train_data2.pkl')
        
        # get the training data
        Gx,Gy,F,V =  self.loadGxyvUxy('training_data/video/pic_ori(250_250_(simple)_uniform_motion_xydirection)/DxyvUxy/train_data17.pkl')
#        Gx,Gy,F,V = self.generateGxGyVF()
        gxyv,fxy = self.generatetraindata(Gx,Gy,F,V)
        
        # Normalize the data
        sensory_x = gxyv / 255.0
        sensory_y = fxy 
          
        # initialize the parameters
        N_NEURONS  = self.sN_NEURONS   # sensor1 
        N_NEURONS_1  = self.sN_NEURONS_1  # sensor2 
        population_s = np.ones((N_NEURONS,1))*0.045  # sensor1 tuning curve
        population_a = np.zeros((N_NEURONS,1))       # sensor1 activation value
        wcross = np.random.uniform(0,1,(N_NEURONS,N_NEURONS_1))
        population_Wcross = wcross / wcross.sum()   # sensor1 HL matrix
        train_round = self.strain_round
        population_Winput = np.random.random((N_NEURONS,sensory_x.shape[1]))/100.0 # sensor1 weight 
        sample_num = sensory_x.shape[0]   
        sample_demension = sensory_x.shape[1]  
        learning_sigmat = self.parametrize_learning_law(50,1,1,train_round)  
        learning_alphat = self.parametrize_learning_law(0.1,0.001,1,train_round)
        ETA = 1.0  
        XI = 1e-3  
        hwi = np.zeros((N_NEURONS,1))
        

        population_s_1 = np.ones((N_NEURONS_1,1))*0.045  # sensor2 tuning curve
        population_a_1 = np.zeros((N_NEURONS_1,1))       # sensor1 activation value
        wcross_1 = np.random.uniform(0,1,(N_NEURONS_1,N_NEURONS))
        population_Wcross_1 = wcross_1 / wcross_1.sum() # sensor2 HL matrix
        print(sensory_y.shape)
        population_Winput_1 = np.random.random((N_NEURONS_1,sensory_y.shape[1]))/100.0    
        sample_num_1 = sensory_y.shape[0]  
        sample_demension_1 = sensory_y.shape[1]  
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
            if t%5 == 4:
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
        a.predict_vgf()
    print(time.time() - start)





















