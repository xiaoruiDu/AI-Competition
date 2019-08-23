#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen May  8 21:27:47 2019

@author: amie
"""



import numpy as np
import math
from scipy import linalg
import matplotlib.pyplot as plt
import pickle
import sys
from PyQt5.QtCore import pyqtSignal, QObject, Qt, pyqtSlot
from PyQt5.QtWidgets import QWidget, QApplication, QGroupBox, QPushButton, QLabel, QCheckBox, QSpinBox, QHBoxLayout, QComboBox, QGridLayout
from time import ctime, sleep
from decimal import *
from numba import jit

class Func(QWidget):
    
    #定义信号，定义槽函数和信号函数的链接在thread.py中
    senddata = pyqtSignal(dict,dict,int,list)
    senddata1 = pyqtSignal(dict,dict,int,list)
    
    def __init__(self):
        super().__init__()
        
        # enables dynamic visualization on network runtime
        self.DYN_VISUAL = 0
        # number of populations in the network    (the number of sensors?)
        self.N_SOM      = 2
        # number of neurons in each population    (different sensor has different output layers?)
        self.N_NEURONS  = 100
        self.N_NEURONS_1 = 100
        # MAX_EPOCHS for SOM relaxation
        self.MAX_EPOCHS_IN_LRN = 50
        self.MAX_EPOCHS_XMOD_LRN = 100
        # number of data samples
        self.N_SAMPLES = 1500
        # decay factors
        self.ETA = 1.0 # activity decay
        self.XI = 1e-3 # weights decay
        # enable population wrap-up to cancel out boundary effects 
        self.WRAP_ON = 0
        self.SensorIndex = 1  # 在UI上显示1号感知器图像
        self.train_startflag = False  # 开始训练的标志，如果为false，则等待start_button按钮的信号
        self.readtxt_flag = False
        
        
        self.pref = [1, 6, 13, 40, 45, 85, 90, 98]
        self.population = {}
        self.sensor_data = {}




    def change_readtxt(self,text):
        print('readTxt_flat:', text)
        self.readtxt_flag = int(text)


            
    def read_txtdata(self,sensory):
        sensory_x = np.loadtxt('sensory_1.txt')
        sensory_x = sensory_x.reshape(len(sensory_x),1)
        sensory_y = np.loadtxt('sensory_2.txt')
        sensory_y = sensory_y.reshape(len(sensory_y),1)
        sensory['x'] = sensory_x
        sensory['y'] = sensory_y
        sensory['num_vals'] = len(sensory_x)
        return sensory            
            
            
    
        
        
    def winput_init(self, pop, sensory):
        if pop['idx'] == 1:
            winput = np.random.choice(sensory['x'][:,0],self.N_NEURONS).reshape(self.N_NEURONS,1)
            pop['Winput'] = winput/winput.sum()
        if pop['idx'] == 2:
            winput = np.random.choice(sensory['y'][:,0],self.N_NEURONS_1).reshape(self.N_NEURONS_1,1)
            pop['Winput'] = winput/winput.sum()


        
        ## xinhao cao 
    
    # 改变初始化的，神经元等函数值。
    def change_pref(self,text):
        pre = text.split(',')
        self.pref = [int(x) for x in pre]
        
        
    def change_trainstartflag(self):
        self.train_startflag = True
    
    
    def change_NeuronsNum(self, text):
        print('NeuronsNum:', text)
        self.N_NEURONS = int(text)
        
    def change_NeuronsNum1(self, text):
        print('NeuronsNum2:', text)
        self.N_NEURONS_1 = int(text)
#        pass
    
    def change_EpochsValue(self,text):
        print('EpochsValue:', text)
        self.MAX_EPOCHS_XMOD_LRN = int(text)
        
    
    def change_SAMPLESNum(self,text):
        print('SAMPLESNum:', text)
        self.N_SAMPLES = int(text)
    
    def change_ETAValue(self,text):
        print('ETAValue:', text)
        self.ETA = float(text)
    
    def change_XIValue(self,text):
        print('XIValue:', text)
        self.XI = float(text)
    
    def change_SensoryIndex(self,text):
        print('SensoryIndex:', text)
        self.SensorIndex = int(text)
        
    
    def changevisualize(self):
        print('You have set DYN_VISUAL = 1')
        print('please be patient, the training is very slow')
        print('It will cost about 30s to get the dynamic process on the SOM_UI')
        self.DYN_VISUAL = 1
        
        
    def changevisualize1(self):
        print('You have set DYN_VISUAL = 0')
        print('The SOM_UI is waitting for your click')
        self.DYN_VISUAL = 0 
        
        
        
    def savevalues(self,pop,sensor):
        self.population = pop
        self.sensor_data = sensor
    
    def getvalues(self):
        return self.population, self.sensor_data
    
    # add hexagonal relation for grid cells example
    def hexagonal_relation(self,sensory_data):
        n = 6
        theta = [x/self.N_SAMPLES for x in range(self.N_SAMPLES)]
        theta = np.array(theta)
        r = np.cos(math.pi/n)/np.cos(2*math.pi*(n*theta%1)/n - math.pi/n)
        sensory_data['x'] = r * np.cos(2*math.pi*theta)
        sensory_data['y'] = r * np.sin(2*math.pi*theta)
        sensory_data['x'] = sensory_data['x'].reshape(self.N_SAMPLES,1)
        sensory_data['y'] = sensory_data['y'].reshape(self.N_SAMPLES,1)
        return sensory_data
        
        
    
    def init_input_data(self):
        sensory_data = {}
        # set up the interval of interest (i.e. +/- range)ststr
        sensory_data['range'] = 1.0
        # setup the number of random input samples to generate
        sensory_data['num_vals'] = self.N_SAMPLES
        # choose between uniformly distributed data and non-uniform distribution
        sensory_data['dist'] = 'uniform' #{uniform, non-uniform}
        # generate observations distributed as some continous heavy-tailed distribution.
        # options are decpowerlaw, incpowerlaw and Gauss
        # distribution
        sensory_data['nufrnd_type'] = ''
        sensory_data['x'] = self.randnum_gen(sensory_data)
        exponent = 3
        sensory_data['y'] = sensory_data['x'] ** exponent
        return sensory_data
        
    def randnum_gen(self,sensory_data):
        # exponent of powerlaw distribution of values
        expn = 3
        # create a uniformly distributed vector of values
        x = np.random.uniform(0,1,(sensory_data['num_vals'],1)) * sensory_data['range']
        # init bounds
        vmin = 0.0
        vmax = sensory_data['range']
        if sensory_data['dist'] == 'uniform':
            y = -sensory_data['range'] + np.random.uniform(0,1,(sensory_data['num_vals'],1)) * (2 * sensory_data['range'])
        
        elif sensory_data['dist'] == 'non-uniform':
            if sensory_data['nufrnd_type'] == 'incpowerlaw':
                y = math.exp(math.log(x * (-vmin**(expn + 1) + vmax**(expn + 1)) + vmin**(expn + 1))/(expn+1))
            elif sensory_data['nufrnd_type'] == 'decpowerlaw':
                y  = -math.exp(math.log(x*(-vmin**(expn+1) + vmax**(expn+1)) + vmin**(expn+1))/(expn+1))     
            elif sensory_data['nufrnd_type'] == 'gauss':
                y  = np.random.uniform(0,1,(sensory_data['num_vals'],1))*(vmax/4)       
            elif sensory_data['nufrnd_type'] == 'convex':
                y = np.zeros((sensory_data['num_vals'],1))
                for idx in range(1,len(x)+1):
                    if idx <= len(x) / 2:
                        vmin = 0.00000005
                        y[idx] = -math.exp(math.log(x[idx]*(-vmin**(expn+1) + vmax**(expn+1)) + vmin**(expn+1))/(expn+1))
                    else:
                        vmin = -vmin
                        y[idx] = math.exp(math.log(x[idx]*(-vmin**(expn+1) + vmax**(expn+1)) + vmin**(expn+1))/(expn+1))
            else:
                print('sensory_data[''nufrnd_type''] key is error')
        else:
            print('sensory_data[''nufrnd_type''] key is error')
        return y
    def create_network_and_initialize_params(self,N_POP, N_NEURONS, N_NEURONS_1):
    #create a network of SOMs given the simulation constants
        populations = []    
        wcross = np.random.uniform(0,1,(N_NEURONS,N_NEURONS_1))
        wcross_1 = np.random.uniform(0,1,(N_NEURONS_1,N_NEURONS))
        sigma_def = 0.045 #固定为调整曲线归一化高度的一半
        for pop_idx in range(1,N_POP+1):
            
            if pop_idx == 1:
                
            
                population_dic = {}
                population_dic['idx'] = pop_idx
                population_dic['lsize'] = N_NEURONS
                population_dic['Winput'] = np.zeros((N_NEURONS,1))
                population_dic['s'] = sigma_def * np.ones((N_NEURONS,1))
                population_dic['Wcross'] = wcross / wcross.sum()
                population_dic['a'] = np.zeros((N_NEURONS,1))
                populations.append(population_dic)
        
            if pop_idx == 2:
                population_dic = {}
                population_dic['idx'] = pop_idx
                population_dic['lsize'] = N_NEURONS_1
                population_dic['Winput'] = np.zeros((N_NEURONS_1,1))
                population_dic['s'] = sigma_def * np.ones((N_NEURONS_1,1))
                population_dic['Wcross'] = wcross_1 / wcross_1.sum()
                population_dic['a'] = np.zeros((N_NEURONS_1,1))
                populations.append(population_dic)
            
        return populations
        
    def parametrize_learning_law(self,v0, vf, t0, tf, ftype):
        #% function to parametrize the learning rate of the
        #% self-organizing-population-code network
        y = np.zeros((tf-t0,1))
        t = [i for i in range(1,tf+1)]
        if ftype == 'sigmoid':
            s = -math.floor(math.log10(tf)) * 10**(-(math.floor(math.log10(tf))))
            p = abs(s*10**(math.floor(math.log10(tf)) + math.floor(math.log10(tf))/2))
            y = v0 - (v0)/(1+math.exp(s*(t-(tf/p)))) + vf
        elif ftype == 'invtime':
            B = (vf*tf - v0*t0)/(v0 - vf)
            A = v0*t0 + B*v0
            y = [A/(t[i]+B) for i in range(len(t))]
        elif ftype == 'exp':
            if v0 < 1:
                p = -math.log(v0)
            else:
                p = math.log(v0)
            y = v0 * math.exp(-t/(tf/p))
        else:
            print('pls cheack ftype in parametrize_l_law function()')
        return y
                
    def visualize_runtime(self,populations,t):
        fig = plt.figure()
        ax2 = plt.subplot()
        id_maxv = np.zeros((populations[0]['lsize'],1))
        for idx in range(populations[0]['lsize']):
            arr =  populations[0]['lsize'][idx,:]
            id_maxv[idx] = np.where(arr==np.max(arr))
        HL = (populations[0]['Wcross'].T)
        ax2.imshow(populations[0]['Wcross'].T, extent=[0,100,100,0])     
        plt.xlabel('neuron index')
        plt.ylabel('neuron index')
        plt.title('Hebbian connection matrix @ epoch {0}'.format(t))
        
    def present_tuning_curves(self, pop, sdata):
        fig = plt.figure()
        ax1 = plt.subplot(4,1,1) 
        ax2 = plt.subplot(4,1,2)
        ax3 = plt.subplot(4,1,3)
        ax4 = plt.subplot(4,1,4)

        # ax1 = plt.subplot(4,1,1) 
        # 激活ax1 以下绘图都在ax1基础上进行
        plt.sca(ax1)
        if pop['idx'] == 1:
            plt.hist(sdata['x'],bins=50,facecolor="blue", edgecolor="black", alpha=0.7)            
        elif pop['idx'] == 2:
            plt.hist(sdata['y'],bins=50,facecolor="blue", edgecolor="black", alpha=0.7)
        plt.xlabel('range pop {0} '.format(pop['idx']))
        plt.ylabel('input values distribution')
        # ax2 = plt.subplot(4,1,2)
        # 激活ax2 以下绘图都在ax1基础上进行
        plt.sca(ax2)
        # compute the tuning curve of the current neuron in the population
        # the equally spaced mean values
        x = np.linspace(-sdata['range'],sdata['range'],pop['lsize'])
        for idx in range(pop['lsize']):
            # extract the preferred values (wight vector) of each neuron
            v_pref = pop['Winput'][idx]
            fx = np.exp(-(x - v_pref)**2/(2*pop['s'][idx]**2))
            plt.plot([x for x in range(100)],fx,linewidth=3)
        pop['Winput'] = np.sort(pop['Winput'],axis=0)  #notice keng
        plt.xticks([])  # quxiao x value
        plt.yticks([])
        # 设置边框不可见（上下左右）
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        # 在图上绘图
        ax23 = fig.add_axes([0.125,0.511739,0.775,0.167391],facecolor='none')
        # 设置边框不可见
        ax23.spines['top'].set_visible(False)
        ax23.spines['right'].set_visible(False)
        ax23.spines['left'].set_linewidth(3)
        ax23.spines['bottom'].set_linewidth(3)
        # 这句话调试吧，忘记了
        plt.setp(ax23.get_xticklabels(),visible=False)
        plt.xticks(pop['Winput'])
        plt.xlim((-1,1))
        plt.xlabel('neuron preferred values')
        plt.ylabel('learned tuning curves shapes')      
        # ax3 = plt.subplot(4,1,3)
        plt.sca(ax3)
        plt.hist(pop['Winput'],bins=50,facecolor="blue", edgecolor="black", alpha=0.7)
        plt.xlabel('range pop {0} '.format(pop['idx']))
        plt.ylabel('# of allocated neurons')
        # ax4 = plt.subplot(4,1,4)
        plt.sca(ax4)
        plt.plot(pop['s'],'.r')
        plt.xlabel('neuron index')
        plt.ylabel('width of tuning curves')
    #DISPLAY ONLY SOME NEURONS
        fig2 = plt.figure()
        bx1 = plt.subplot(1,1,1)
        # print(plt.gca())
        # 排序函数需要指定坐标轴
        v_pref = np.sort(pop['Winput'],axis=0)  #notice keng
        pref= self.pref
        for idx in range(len(pref)):
            idx_pref = pref[idx]
            fx = np.exp(-(x - v_pref[idx_pref])**2/(2*pop['s'][idx_pref]**2))
            plt.plot([x for x in range(100)],fx,linewidth=3)
        plt.sca(bx1)
        plt.xticks([])  # quxiao x value
        plt.yticks([])
        # 在图上画图
        bx12 = fig2.add_axes([0.125,0.11,0.775,0.77],facecolor='none')
        bx12.spines['left'].set_linewidth(3)
        bx12.spines['bottom'].set_linewidth(3)
        v_pref_idx = np.zeros((1,len(pref)))
        for idx in range(len(pref)):
            v_pref_idx[0][idx] = v_pref[pref[idx] + 1]  #keng dian !!!
        plt.xticks(v_pref_idx[0],rotation=90)  #keng doam 
        plt.xlim(-1,1)
        plt.xlabel('preferred value')
        plt.ylabel('learned tuning curves shapes')
        bx13 = fig2.add_axes([0.125,0.11,0.774,0.77],facecolor='none')
        plt.xticks(v_pref_idx[0],pref)
        plt.xlim(-1,1)
        bx13.xaxis.tick_top()
        plt.title('neuron index')

    def visualize_results(self,populations,sensory_data,learning_params):
        fig = plt.figure()
        # ax1 图占用2个区域
        ax1 = plt.subplot(4,1,(1,2))
        ax2 = plt.subplot(4,1,(3,4))
        plt.sca(ax1)
        plt.plot(sensory_data['x'],sensory_data['y'],'.g')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Encoded relation')
        # learned realtionship encoded in the Hebbian links
        plt.sca(ax2)
        # extract the max weight on each row (if multiple the first one)
        id_maxv = np.zeros((populations[0]['lsize'],1))   #(100,1)
        for idx in range(populations[0]['lsize']):
            arr =  populations[0]['Wcross'][idx,:]
            id_maxv[idx] = np.where(arr==np.max(arr))
        
        # 画热力图
        ax2.imshow(populations[0]['Wcross'].T, extent=[0,100,100,0])
        plt.xlabel('neuron index')
        plt.ylabel('neuron index')
        fig.tight_layout()      
        fig2 = plt.figure()
        plt.plot(learning_params['alphat'],'k',linewidth=3)
        plt.xlabel('SOM training epochs')
        plt.ylabel('SOM Learning rate')
        fig3 = plt.figure()
        plt.plot(self.parametrize_learning_law(populations[0]['lsize']/2,1,learning_params['t0'],learning_params['tf_learn_in'],'invtime'),'k',linewidth=3)
        plt.xlabel('SOM training epochs')
        plt.ylabel('SOM neighborhood size')
        # hebbian learning
        fig4 = plt.figure()
        etat = self.parametrize_learning_law(0.1,0.001,learning_params['t0'],learning_params['tf_learn_cross'],'invtime')
        plt.plot(etat,'m',linewidth=3)
        plt.xlabel('Hebbian learning epochs')
        plt.ylabel('Hebbian Learning rate')
        fig5 = plt.figure()
        plt.subplot(2,1,1)
        plt.plot(populations[0]['Winput'],'.g')
        plt.xlabel('neuron index in pop 1')
        plt.ylabel('preferred value')
        bx2 = plt.subplot(2,1,2)
        plt.plot(populations[1]['Winput'],'.b')
        plt.xlabel('neuron index in pop 2')
        plt.ylabel('preferred value')
        fig5.tight_layout()
        

        
    
    def main_thread(self,train=True):
        
        while 1:  # 检查pushbutton传来的信号
            sleep(1)
            if self.train_startflag: #如果信号传来，标志更新，开始训练（后面有break）
                
            # INIT INPUT DATA - RELATION IS EMBEDDED IN THE INPUT DATA PAIRS
                sensory_data = self.init_input_data()
                if self.readtxt_flag:
                    print('read data from txt....')
                    sensory_data = self.read_txtdata(sensory_data)
                
                # add hexagonal relation for grid cells example
#                sensory_data = self.hexagonal_relation(sensory_data=sensory_data)
                print('x:',np.shape(sensory_data['x']))
                print('y:',np.shape(sensory_data['y']))
          
            # CREATE NETWORK AND INITIALIZE PARAMS
                #populations = [{},{}]
                populations = self.create_network_and_initialize_params(N_POP=self.N_SOM, N_NEURONS=self.N_NEURONS, N_NEURONS_1=self.N_NEURONS_1)
                
                # initiate the winput(in order to avoid the dead node)
                self.winput_init(populations[0],sensory_data)
                self.winput_init(populations[1],sensory_data)     
                
                
                act_cur = np.zeros((self.N_NEURONS,1)) 
                hwi = np.zeros((self.N_NEURONS,1)) 
                hwi_1 = np.zeros((self.N_NEURONS_1,1))  #增加 
                
                
                
                learning_params = {}
                learning_params['t0'] = 1
                learning_params['tf_learn_in'] = self.MAX_EPOCHS_IN_LRN
                learning_params['tf_learn_cross'] = self.MAX_EPOCHS_XMOD_LRN
                sigma0 = self.N_NEURONS/2
                sigmaf = 1
                learning_params['sigmat'] = self.parametrize_learning_law(sigma0,sigmaf,learning_params['t0'],learning_params['tf_learn_in'],'invtime')
                alpha0 = 0.1
                alphaf = 0.001
                learning_params['alphat'] = self.parametrize_learning_law(alpha0,alphaf,learning_params['t0'],learning_params['tf_learn_in'],'invtime')
                # cross-modal learning rule type
                cross_learning = 'covariance' #{hebb - Hebbian, covariance - Covariance, oja - Oja's Local PCA}
                #  mean activities for covariance learning
#               avg_act = np.zeros((self.N_NEURONS,self.N_SOM))
                avg_act = np.zeros((self.N_NEURONS,1)) 
                avg_act_1 = np.zeros((self.N_NEURONS_1,1)) 
            # NETWORK SIMULATION LOOP
                if train:
                    print('Started training ...\n')
                    
                    for t in range(1,learning_params['tf_learn_cross']+1): #1 dao 100   {learning_params['tf_learn_cross']+1}
        #                if self.DYN_VISUAL == 1:
        #                    self.visualize_runtime(populations,t)
                        if  t<= learning_params['tf_learn_in']:  # t qu [0,50]
                            for didx in range(sensory_data['num_vals']):  #ci chu 0.1 wenti  (didx qu 0 dao 1449)
                                for pidx in range(1,self.N_SOM+1):  # pidx qu 1 dao 2
                                    act_cur = np.zeros((populations[pidx-1]['lsize'],1)) #ci chu 0,1wenti 
                                    if pidx == 1:
                                        input_sample = sensory_data['x'][didx]
                                    elif pidx == 2:
                                        input_sample = sensory_data['y'][didx]
                                    else:
                                        print('pidx is out of range')
                                        
                                        
                                    # compute new activity given the current input sample
#                                    for idx in range(populations[pidx-1]['lsize']): # idx qu 0 dao 99
#                                        act_cur[idx] = (1/(math.sqrt(2*math.pi)*populations[pidx-1]['s'][idx]))*math.exp(-(input_sample - populations[pidx-1]['Winput'][idx])**2/(2*populations[pidx-1]['s'][idx]**2))
                                    
                                    #矩阵运算，加速语句,获得同时获得100个激活值
                                    act_cur = (1/(np.sqrt(2*np.pi)*populations[pidx-1]['s']))*np.exp(-np.power((input_sample - populations[pidx-1]['Winput']),2)/(2*np.power(populations[pidx-1]['s'],2)))
                                    #  normalize the activity vector of the population
                                    act_cur_sum = act_cur.sum()
                                    if act_cur_sum == 0:
#                                        print('pidx11111:',pidx)
                                        print('act_cur.sum() is less than 1e-323,ignore the update!')
                                        continue
                                    act_cur = act_cur / act_cur_sum
                                    # update the activity for the next iteration
                                    populations[pidx-1]['a'] = (1-self.ETA)*populations[pidx-1]['a'] + self.ETA*act_cur
                                    arr = populations[pidx-1]['a']
                                    win_act = np.max(populations[pidx-1]['a'])
                                    win_pos = np.where(arr==np.max(arr))[0][0]
                                    
                                    
                                    
                                    pos_list = np.arange(0,self.N_NEURONS,1)
                                    pos_list_1 = np.arange(0,self.N_NEURONS_1,1)
                                    if self.WRAP_ON == 1:
#                                   hwi[idx] = math.exp(-linalg.norm(min([linalg.norm(idx-win_pos), self.N_NEURONS - linalg.norm(idx-win_pos)]))**2/(2*learning_params['sigmat'][t-1]**2))
                                        print('waiting to process')
                                    else:
#                                   hwi[idx] = math.exp(-np.linalg.norm(pos-win_pos)**2/(2*learning_params['sigmat'][t-1]**2))
                                        if pidx == 1:
                                            #矩阵运算，加速语句，同时获得100个hwi值
                                            hwi_temp = np.exp(-np.power(pos_list - win_pos, 2) / (2 * np.power(learning_params['sigmat'][t-1],2)))
#                                            hwi_temp = np.exp(-np.power(np.linalg.norm((pos_list - win_pos).reshape(self.N_NEURONS,1),axis=1),2)/(2 * np.power(learning_params['sigmat'][t-1],2)))
                                            hwi = hwi_temp.reshape(self.N_NEURONS,1)
                                        elif pidx == 2:
                                            #矩阵运算，加速语句，同时获得100个hwi_1值
                                            hwi_temp1 = np.exp(-np.power(pos_list_1 - win_pos, 2) / (2 * np.power(learning_params['sigmat'][t-1],2)))
#                                            hwi_temp1 = np.exp(-np.power(np.linalg.norm((pos_list_1 - win_pos).reshape(self.N_NEURONS_1,1),axis=1),2)/(2 * np.power(learning_params['sigmat'][t-1],2)))
                                            hwi_1 = hwi_temp1.reshape(self.N_NEURONS_1,1)
                                    
                                    if pidx == 1:
                                        #矩阵运算，加速语句，同时更新100个权重
                                        populations[pidx - 1]['Winput'] = populations[pidx - 1]['Winput']+ \
                                        learning_params['alphat'][t-1] * hwi * (input_sample - populations[pidx - 1]['Winput'])
                            
                                        # 同时更新100个 tuning curve值
                                        populations[pidx-1]['s'] = populations[pidx-1]['s'] + \
                                        learning_params['alphat'][t-1] *  (1/(np.sqrt(2*np.pi)*learning_params['sigmat'][t-1])) * \
                                        hwi * (np.power((input_sample - populations[pidx-1]['Winput']),2) - np.power(populations[pidx-1]['s'],2))  

                                    if pidx == 2:
                                        #矩阵运算，加速语句
                                        populations[pidx - 1]['Winput'] = populations[pidx - 1]['Winput']+ \
                                        learning_params['alphat'][t-1] * hwi_1 * (input_sample - populations[pidx - 1]['Winput'])
                            
                                        populations[pidx-1]['s'] = populations[pidx-1]['s'] + \
                                        learning_params['alphat'][t-1] *  (1/(np.sqrt(2*np.pi)*learning_params['sigmat'][t-1])) * \
                                        hwi_1 * (np.power((input_sample - populations[pidx-1]['Winput']),2) - np.power(populations[pidx-1]['s'],2))  
                
#                                    for idx in range(populations[pidx-1]['lsize']): #0 dao 99,go through neurons in the population
#                                        if self.WRAP_ON == 1:
#                                            hwi[idx] = math.exp(-linalg.norm(min([linalg.norm(idx-win_pos), self.N_NEURONS - linalg.norm(idx-win_pos)]))**2/(2*learning_params['sigmat'][t-1]**2))
#                                        else:
#                                            # simple Gaussian kernel with no boundary compensation
#                                            if pidx == 1:
#                                                
#                                                hwi[idx] = math.exp(-linalg.norm(idx-win_pos)**2/(2*learning_params['sigmat'][t-1]**2))
#                                            elif pidx == 2:
#                                                hwi_1[idx] = math.exp(-linalg.norm(idx-win_pos)**2/(2*learning_params['sigmat'][t-1]**2))                                            # simple Gaussian kernel with no boundary compensation
#                                        if pidx == 1:
#                                            
#                                            populations[pidx - 1]['Winput'][idx] = populations[pidx - 1]['Winput'][idx] + \
#                                            learning_params['alphat'][t-1] * hwi[idx] * (input_sample - populations[pidx - 1]['Winput'][idx])
#                                            
#                                            populations[pidx-1]['s'][idx] = populations[pidx-1]['s'][idx] + \
#                                            learning_params['alphat'][t-1] *  (1/(math.sqrt(2*math.pi)*learning_params['sigmat'][t-1])) * \
#                                            hwi[idx] * ((input_sample - populations[pidx-1]['Winput'][idx])**2 - populations[pidx-1]['s'][idx]**2)
#                                        
#                                        elif pidx == 2:
#                                            populations[pidx - 1]['Winput'][idx] = populations[pidx - 1]['Winput'][idx] + \
#                                            learning_params['alphat'][t-1] * hwi_1[idx] * (input_sample - populations[pidx - 1]['Winput'][idx])
#                                            
#                                            populations[pidx-1]['s'][idx] = populations[pidx-1]['s'][idx] + \
#                                            learning_params['alphat'][t-1] *  (1/(math.sqrt(2*math.pi)*learning_params['sigmat'][t-1])) * \
#                                            hwi_1[idx] * ((input_sample - populations[pidx-1]['Winput'][idx])**2 - populations[pidx-1]['s'][idx]**2)                                
                                
                                if self.DYN_VISUAL == 1 and didx % 100 == 99 and self.SensorIndex == 1:
                                    self.senddata.emit(populations[0],sensory_data,self.N_NEURONS_1,self.pref)
        #                            self.senddata1.emit(populations[1],sensory_data,1)
                                    sleep(1.5)
                                elif self.DYN_VISUAL == 1 and didx % 100 == 99 and self.SensorIndex == 2:
                                    self.senddata.emit(populations[1],sensory_data,self.N_NEURONS,self.pref)
                                    sleep(1.5)
                                        
                                        
                        # learn the cross-modal correlation
                        
                        for didx in range(sensory_data['num_vals']): #(0-1449)
                            for pidx in range(1,self.N_SOM+1):
                                act_cur = np.zeros((populations[pidx-1]['lsize'],1))
                                #  pick a new sample from the dataset and feed it to the current layer
                                if pidx == 1:
                                    input_sample = sensory_data['x'][didx]
                                elif pidx == 2:
                                    input_sample = sensory_data['y'][didx]
                                else:
                                    print('pidx is out of range')
                                    
#                                for idx in range(populations[pidx-1]['lsize']): # idx qu 0 dao 99
#                                     act_cur[idx] = (1/(math.sqrt(2*math.pi)*populations[pidx-1]['s'][idx]))*math.exp(-(input_sample - populations[pidx-1]['Winput'][idx])**2/(2*populations[pidx-1]['s'][idx]**2))
                                #矩阵运算，加速语句
                                act_cur = (1/(np.sqrt(2*np.pi)*populations[pidx-1]['s']))*np.exp(-np.power((input_sample - populations[pidx-1]['Winput']),2)/(2*np.power(populations[pidx-1]['s'],2)))
                                
                                #  normalize the activity vector of the population
                                act_cur_sum = act_cur.sum()
                                if act_cur_sum == 0:
#                                    print('pidx2222:',pidx)
                                    print('act_cur.sum() is less than 1e-323,ignore the update!')
                                    continue
                                act_cur = act_cur / act_cur_sum
                                # update the activity for the next iteration
                                populations[pidx-1]['a'] = (1-self.ETA)*populations[pidx-1]['a'] + self.ETA*act_cur
                            if cross_learning == 'hebb':
                                populations[0]['Wcross'] = (1-self.XI)*populations[0]['Wcross'] + self.XI*populations[0]['a']*populations[1]['a'].T;
                                populations[1]['Wcross'] = (1-self.XI)*populations[1]['Wcross'] + self.XI*populations[1]['a']*populations[0]['a'].T;
                            elif cross_learning == 'covariance':
                                # compute the mean value computation decay
                                OMEGA = 0.002 + 0.998/(t+2)
                                for pidx in range(self.N_SOM): # 0 de wen ti 
                                    if pidx == 0:
                                        avg_act[:, 0] = (1-OMEGA)*avg_act[:, 0] + OMEGA*populations[pidx]['a'][:,0]
                                        
                                    elif pidx == 1:
                                        avg_act_1[:, 0] = (1-OMEGA)*avg_act_1[:, 0] + OMEGA*populations[pidx]['a'][:,0]
                                    
                                # cross-modal Hebbian covariance learning rule: update the synaptic weights
                                populations[0]['Wcross'] = (1-self.XI)*populations[0]['Wcross'] + self.XI*(populations[0]['a'] - avg_act[:, 0].reshape(self.N_NEURONS,1))*(populations[1]['a'] - avg_act_1[:, 0].reshape(self.N_NEURONS_1,1)).T
                                populations[1]['Wcross'] = (1-self.XI)*populations[1]['Wcross'] + self.XI*(populations[1]['a'] - avg_act_1[:, 0].reshape(self.N_NEURONS_1,1))*(populations[0]['a'] - avg_act[:, 0].reshape(self.N_NEURONS,1)).T
                            elif cross_learning == 'oja':
                                # Oja's local PCA learning rule
                                populations[0]['Wcross'] = ((1-self.XI)*populations[0]['Wcross'] + self.XI*populations[0]['a']*populations[1]['a'].T)/math.sqrt(sum(sum((1-self.XI)*populations[0]['Wcross'] + self.XI*populations[0]['a']*populations[1]['a'].T)))
                                populations[1]['Wcross'] = ((1-self.XI)*populations[1]['Wcross'] + self.XI*populations[1]['a']*populations[0]['a'].T)/math.sqrt(sum(sum((1-self.XI)*populations[1]['Wcross'] + self.XI*populations[1]['a']*populations[0]['a'].T)));
                            else:
                                print('cross_learning type is not hebb neigher covariance')
                        
        #                self.savevalues() 
        #                print('Wcross_shape:',np.shape(populations[0]['Wcross']))
                            
                            if self.DYN_VISUAL == 1 and didx % 100 == 99 and self.SensorIndex == 1:
                                #print('go111')
                                self.senddata.emit(populations[0],sensory_data,self.N_NEURONS_1,self.pref)
        #                        self.senddata1.emit(populations[1],sensory_data,1)
                                sleep(1.5)
                            elif self.DYN_VISUAL == 1 and didx % 100 == 99 and self.SensorIndex == 2:
                                self.senddata.emit(populations[1],sensory_data,self.N_NEURONS,self.pref)
                                sleep(1.5)
                        
        
                        print('training:',t/100)    
                    print('Ended training sequence. Presenting results ...\n')                
                    # VISUALIZATION
                    with open('populations1.pkl','wb') as output:
                        pickle.dump(populations,output)
                    with open('sensory_data1.pkl','wb') as output1:
                        pickle.dump(sensory_data,output1)
                    with open('learning_params1.pkl','wb') as output2:
                        pickle.dump(learning_params,output2)       
                    self.present_tuning_curves(populations[0],sensory_data)
                    self.present_tuning_curves(populations[1],sensory_data)
                    populations[0]['Wcross'] = populations[0]['Wcross'] / np.max(populations[0]['Wcross'])  #np.max  keng dian
                    populations[1]['Wcross'] = populations[1]['Wcross'] / np.max(populations[1]['Wcross'])
                    # visualize post-simulation weight matrices encoding learned relation
                    lrn_fct = self.visualize_results(populations,sensory_data,learning_params)
                if train == False:
                    print('load .pkl  \n')
                    pkl_file1 = open('populations_100.pkl','rb')
                    pop = pickle.load(pkl_file1)
                    pkl_file2 = open('sensory_data_100.pkl','rb')
                    sensory_data = pickle.load(pkl_file2)
                    pkl_file3 = open('learning_params_100.pkl','rb')
                    learning_params1 = pickle.load(pkl_file3)
                    self.present_tuning_curves(pop[0],sensory_data)
                    self.present_tuning_curves(pop[1],sensory_data)
                    self.visualize_results(pop,sensory_data,learning_params1)  
                    
                break
if __name__ == '__main__':
    F = Func()
    # train == True: the fuction will train the data with Epoch == 2 
    # train == False: The network will load the pre_train parameters with Epoch == 100
    # Default train == True
    F.main_thread(train=False)













