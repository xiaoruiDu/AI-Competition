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
        value = self.readconfig('config_IG.txt')        
        self.strain = int(value[0])
        self.sN_NEURONS = int(value[1])
        self.sN_NEURONS_1 = int(value[2])
        self.sETA = float(value[3])
        self.sXI = float(value[4])
        self.strain_round = int(value[5])
        self.strainpath =value[6]
        self.stestpath = value[7]
        
        
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
    
    
    def predict(self,picpath):
        
        # initialize the parameters
        picpath = self.stestpath 
        population_a = np.zeros((self.sN_NEURONS,1))  
        population_s = np.ones((self.sN_NEURONS,1))*0.045
        wcross = np.random.uniform(0,1,(self.sN_NEURONS,self.sN_NEURONS_1)) 
        population_Wcross = wcross / wcross.sum() 
        population_Winput = np.random.random((self.sN_NEURONS,9))/10.0    
        population_Winput_1 = np.random.random((self.sN_NEURONS_1,1))/10.0    
        
        pic_data = pic.picprocess()
        picpath = ''
        with open(self.stestpath) as file:
            for line in file.readlines():
                if line == '\n':
                    continue
                picpath = line.strip()
        image = cv2.imread(picpath)
        pic_width = image.shape[0]
        pic_hight = image.shape[1]                
        
        # load the test picture
        databoxe,data_xdirection,data_ydirection = pic_data.pix_singlechannel_cellload(self.stestpath,[3,3])
        sensory_x = databoxe / 255.0
        sensory_y = data_xdirection / 255.0
        sensory_y = sensory_y.reshape(-1,1)
        
        # load the model
        with open('Weight data/car__train_1000_1000/populations_Wcross152.pkl','rb') as file:
            population_Wcross = pickle.load(file)
        
        with open('Weight data/car__train_1000_1000/populations_Winput152.pkl','rb') as file1:
            population_Winput = pickle.load(file1)
            
        with open('Weight data/car__train_1000_1000/population_Winput_1152.pkl','rb') as file2:
            population_Winput_1 = pickle.load(file2)

        with open('Weight data/car__train_1000_1000/populations_s152.pkl','rb') as file3:
            population_s = pickle.load(file3)

        # show the HL matrix
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
            a1 = population_Winput_1[pre_pos]*255
            x_drection.append(a1)
        # rebuild the optical flow picture 
        pic_data.createGpic(x_drection,[pic_width-2,pic_hight-2])
       
    def parametrize_learning_law(self, v0, vf, t0, tf):
        y = np.zeros((tf-t0,1))
        t = [i for i in range(1,tf+1)]
        B = (vf*tf - v0*t0)/(v0 - vf)
        A = v0*t0 + B*v0
        y = [A/(t[i]+B) for i in range(len(t))]
        return y

    def speed_up_som(self):
        
        pic_data = pic.picprocess()

        # get the training data
        databoxe,data_xdirection,data_ydirection = pic_data.pix_singlechannel_cellload(self.strainpath,[3,3])
        # Normalize the data
        sensory_x = databoxe / 255.0
        sensory_y = data_xdirection / 255.0
        sensory_y = sensory_y.reshape(-1,1)

            
        # initialize the parameters
        N_NEURONS  = self.sN_NEURONS   # sensor1 
        N_NEURONS_1  = self.sN_NEURONS_1  # sensor2
        population_s = np.ones((N_NEURONS,1))*0.045  # sensor1 tuning curve
        population_a = np.zeros((N_NEURONS,1))       # sensor1 activation value
        wcross = np.random.uniform(0,1,(N_NEURONS,N_NEURONS_1))
        population_Wcross = wcross / wcross.sum()    # sensor1 HL matrix
        train_round = self.strain_round                          
        population_Winput = np.random.random((N_NEURONS,sensory_x.shape[1]))/10.0    # sensor1 weight 
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
        population_Wcross_1 = wcross_1 / wcross_1.sum()  # sensor2 HL matrix
        print(sensory_y.shape)
        population_Winput_1 = np.random.random((N_NEURONS_1,sensory_y.shape[1]))/10.0    #初始权重
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
                
                input_sample = sensory_x[sample_index].reshape(1,-1) #(1,1024)
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

            if t%50 == 49:
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
        a.predict(a.stestpath)
    print(time.time() - start)





















