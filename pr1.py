#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

import matplotlib.pyplot as plt


# In[ ]:


class lr(object):
    """
    Class constructor.
    """
    N_ITERS = 100
    
    def __init__(self,x0:list = None,y0:list = None,l:float = 0.7):
        """
        Constructor method.
        """
        
        self.x0 = x0
        assert type(x0)==list
        
        
        self.y0 = y0
        assert type(y0)==list
        
        
        self.l = l
        #forces to be float or init, otherwise error raised
        assert (type(l)==float or type(l)==int)  
        
        
        self.n = len(self.x0)
        self.h = np.zeros(lr.N_ITERS)
        
        #weight,bias, not part of init
        self.a = None
        self.b = None
    
    
    
    def __str__(self):
        """String representation of an object. Init params are instance attributes.
        :return: init param as string
        :rtype: str
        """
        return f"init params:{self.x0},{self.y0}, {self.l}"
    
    
    
    def c(self,y_):
        """
        Cost function
        :param y_: estimated value of y
        :type y_: int or float, required
        
        :return: cost associated with estimated value of y
        :rtype: int or float
        
        """
        
        for i,v in enumerate(self.y0):
            
            c = np.sum(np.square(self.y0[i] - y_))/(2*self.n)
            c = float(c)
        
        return c 
    
    
    
    def f(self):
        """
        Fit function
        :param y_: 
        :type y_: 
        
        :return:
        :rtype: 
        
        """
        self.h = np.zeros(lr.N_ITERS)
        
        for e in range(lr.N_ITERS):
            
            for i,(v,k) in enumerate(zip(self.x0,self.y0)):
                
                y_ = self.b*self.x0[i] + self.a
            
                dv_a = (-2/self.n)*(self.y0[i] - y_)
            
                dv_b = (-2/self.n)*(self.x0[i]*(self.y0[i] - y_))
            
                
                
                self.a -= dv_a*self.l
                
                self.b -= dv_b*self.l
             
                
                self.mse = self.mean_se(y_,self.y0)
            
            
            self.h[e] = self.c(y_)
 
        
        return self.mse
    
    
    
    
    
    def mean_se(self,y_p,y0):
        """
        Mean squared error.
        :param y_p: 
        :type y_p: 
        
        :param y: 
        :type y: 
        
        :return:
        :rtype:
        
        """
        for i,v in enumerate(y0):
            
            mse = (1/self.n)*np.sum(np.square(y0[i] - y_p))
        
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
        :rtype: dict
        """
        self.dict = {}
        
        return self.dict
    
    
    
    def res(self):
        
        f=plt.figure(figsize=(10,10))
        q=f.add_subplot(211)

        plt.title("error minimisation for a number of iterations")
        
        q.plot(self.h)

