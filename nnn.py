#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[4]:


class nn(object):
    """
    Class constructor.
    """
    def __init__(self,x0;np.ndarray,y0:np.ndarray,lr:float=0.001,n_iters:int=1000)
        """
        Constructor method.
        """
        self.x0 = x0
        self.y0 = y0
        self.lr = lr
        self.n_iters = n_iters
        self.m, self.n = np.shape(self.x0)[0], np.shape(self.x0)[1]
    
        
        self.w = None
        self.b = None
    
    def initialise(self,features):
        
        #randn takes shape of the output array from normal dist
        w = np.random.randn(len(features),2)
        b = 0
        return w,b
    
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    
    def forward_propagate(self,)
        self.sigmoid(np.matmul(self.x0,w.T)+self.b)
    
    
    
    


# In[9]:


a = np.random.randn(10,2)
a.shape[1]


# In[ ]:




