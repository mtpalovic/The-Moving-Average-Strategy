#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy import linspace, polyval, polyfit, sqrt, stats, randn, optimize

import pandas as pd
import matplotlib.pyplot as plt

import csv

import scipy
from sklearn.linear_model import LinearRegression

import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import sphinx


# In[3]:


class lr(object):
    """
    Class constructor.
    """
    N_ITERS: 1000
    
    def __init__(self,x0,y0,l):
        """
        Constructor method.
        """
        
        self.x0 = x0
        self.y0 = y0
        
        self.l = float(l)
        #forces to be float or init, otherwise error raised
        assert (type(l)==float or type(l)==int or type(x0)==None)
        
        
        
        self.n = len(self.x0)
        
        #weight,bias, not part of init
        self.a = 0
        self.b = 0
        
        self.h = np.zeros(lr.N_ITERS)
    
    
    
    
    
    def __str__(self):
        """String representation of an object. Init params are instance attributes.
        :return: init param as string
        :rtype: str
        """
        return f"init params:{self.x0},{self.y0}, {self.l}"
    
    
    
    
    
    
    def c(self,y_):
        """
        Cost function
        :param y_: init method
        :type y_: init method
        
        :return:
        :rtype: 
        
        """
        c = np.sum(np.square(self.y0 - y_))/(2*self.n)
        return c 
    
    def f(self):
        """
        Fit function
        :param y_: init method
        :type y_: init method
        
        :return:
        :rtype: 
        
        """
        for e in range(0,len(self.iters_),1):
            
            y_ = self.b*self.X + self.a
            
            dv_a = (-2/self.n)*(self.y0 - y_)
            
            dv_b = (-2/self.n)*(self.x0*(self.y0 - y_))
            
            
            self.a = self.a - dv_a*self.l
            self.b = self.b - dv_b*self.l
            
            self.history[e] = self.c(y_)
            self.mse = self.mean_se(self.y0, y_)
        
        return self.mse
    
    
    def mean_se(self,y_p,y):
        """
        Mean squared error.
        :param y_p: 
        :type y_p: 
        
        :param y: 
        :type y: 
        
        :return:
        :rtype:
        
        """
        er = y - y_p
        mse = np.sum(np.square(er))/self.n
        
        return mse
    
    
    
    def create_list(self,*args):
        self.li = [*args]
        return self.li
    
    
    def create_tuple(self):
        """Creates a tuple
        :return:
        :rtype: tuple
        """
        self.tupl = ()
        
        return self.tupl
    
    
     def create_dict(self):
        """Creates a dictionary
        :return:
        :rtype:
        """
        self.dict = {}
        
        return self.dict


# In[4]:


q = lr(100,10,800)


# In[ ]:




