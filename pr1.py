#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import random


# In[ ]:


class linreg(object):
    """
    Class constructor.
    """
    
    def __init__(self,x0:list,y0:list,l:float = 0.005,n_iters:int=10000):
        """
        Constructor method.
        """
        self.x0 = x0
        self.y0 = y0
        assert (type(x0)==list and type(y0)==list)
        
        self.l = l
        assert (type(l)==float or type(l)==int)  
        
        
        self.n_iters = n_iters
        self.n = len(self.x0) 
        self.h = np.zeros(self.n_iters)
        
        #weight,bias, not part of init
        self.a = 2
        self.b = 3
        self.mse = 0
    
    
    
    def __str__(self):
        """String representation of an object. Init params are instance attributes.
        :return: init param as string
        :rtype: str
        """
        return f"init params:{self.x0},{self.y0}, {self.l}"
    
    
    
    def _c(self,y_:float):
        """
        Cost function. Predicts cost of one estim value and all data points.
        :param y_: estimated value of y
        :type y_: float, required
        
        :return: cost associated with estimated value of y
        :rtype: float
        
        """
        
        if y_:
            if isinstance(y_,float):
                c = 0
                for i,v in enumerate(self.y0):
                    c = np.sum(np.square(self.y0[i] - y_))/(2*self.n)
                    c = float(c)
        return c
    
    
    
    def _f(self):
        """
        Fit function
        :return: mean squared error
        :rtype: float
        
        """
        
        #creates a list of length of n_iters
        self.h = np.zeros(self.n_iters)
        
        for e in range(len(self.h)):
            for i,(v,k) in enumerate(zip(self.x0,self.y0)):
                y_ = self.b*self.x0[i] + self.a
                dv_a = (-2/self.n)*(self.y0[i] - y_)
                dv_b = (-2/self.n)*(self.x0[i]*(self.y0[i] - y_))
                self.a = self.a - dv_a*self.l
                self.b = self.b -dv_b*self.l
                
                self.mse = self._mse(y_,self.y0)
            self.h[e] = self._c(y_)
        return self.mse
    
    
    
    
    
    def _mse(self,y_p:float,y0:list):
        """
        Mean squared error.
        :param y_p: estimated value of y from _f()
        :type y_p: float
        
        :param y0: y0 observations
        :type y0: list 
        
        :return: mean squared error
        :rtype: float
        
        """
        for i,v in enumerate(y0):
            mse = (1/self.n)*np.sum(np.square(y0[i] - y_p))
        return mse
    
    
    def create_tuple(self):
        """Creates a tuple
        :return:
        :rtype: tuple
        """
        self.tuple = ()
        
        return self.tuple
    
    
    def create_dict(self):
        """Creates a dictionary
        :return:
        :rtype: dict
        """
        self.dict = {}
        
        return self.dict
    
    
    def _res(self):
        
        f=plt.figure(figsize=(10,10))
        z=f.add_subplot(211)
        plt.title("error minimisation for a number of iterations")
        z.plot(self.h)

