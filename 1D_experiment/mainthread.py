#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:13:24 2019

@author: amie
"""

import threading
import time
import function as fun_c

'''
定义多线程

'''


class myThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.F = fun_c.Func()
        
    def run(self):
        print ("start a new thread： " )
        # 获取锁，用于线程同步
        threadLock = threading.Lock()
        
        # 上锁
        threadLock.acquire()
        
        # 跑线程，训练网络，发送中间结果
        self.F.main_thread(train=True)
        
        # 释放锁，开启下一个线程
        threadLock.release()
        
    def terminate(self):
        self._running = False




