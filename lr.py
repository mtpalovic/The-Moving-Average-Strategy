#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
random.seed(43)

import matplotlib.pyplot as plt


# In[ ]:


class lr(object):
    """
    Class constructor.
    """
    
    def __init__(self,x0:np.ndarray, y0:np.ndarray,lr:float=0.001,n_iters:int=1000):
        """
        Constructor method.
        """
        self.x0 = x0
        self.y0 = y0
        #assert(type(x0)==np.ndarray and type(y0)==np.ndarray) 
        
        
        self.lr = lr
        assert type(lr)==float
        
        
        self.n_iters = n_iters
        self.m, self.n = np.shape(self.x0)[0], np.shape(self.x0)[1]
        
        # 1D array of size n_features
        #initialise theta
        self.th = np.ones(self.n)
        
     
    
    
    
    def create_array(self):
        """Creates a list
        :return:
        :rtype: list
        """
        
        self.array = []
        
        return self.array
    
    
    
    
    
    def cost_not_vectorised(self):
        """
        Cost function. Estimated parameters and true values across all rows.
        
        :return: cost associated with estimated value of y
        :rtype: float
        
        """
        
        #this indicates that cost will be a float
        #sum cost across all rows
        c = 0
        
        for i in range(self.m):    
            
            #y_hat for each row
            y_hat = 0
            
            for j in range(self.n):
                y_hat += self.th[j]*self.x0[i][j]
            
            #np.power because of neg distance and punishing large outliers
            #cost for each obs i
            c_i = np.power((y_hat - self.y0[i]),2)
            c += c_i
        
        cost = (1/(2*self.m))*c
        
        
        return cost
    
    
    
    
    def gradient_descent_not_vectorised(self):
        """
        Gradient descent using a for loop. Non-vectorised implementation.
        
        :return: cost associated with estimated value of y
        :rtype: float
        
        """
        
        #calc cost for each iteration as gradient descent continues
        cost_l = []
        
        
        for _ in range(self.n_iters):
            
            #only one derivative for each column/feature
            
            dev_ = []
            for k in range(self.n):
                
                
                
                
                #sum derivatives from all rows
                dev_sum = 0
                for i in range(self.m):
                    
                    #estimate
                    y_hat = 0
                    for j in range(self.n):
                        y_hat += self.th[j]*self.x0[i][j]
                    
                    #derivative for each row 
                    d_i = (y_hat - self.y0[i])*self.x0[i][k]
                    dev_sum += d_i 
                
                
                
                
                
                #append to list
                #derivative for each column
                dev = (1/self.m)*dev_sum 
                dev_.append(dev)
            
            
            #update params stored as self.th acc to lr rate
            self.th = self.th - self.lr*np.array(dev_)
            
            cos = self.cost_not_vectorised()
            
            cost_l.append(cos)
            
            plt.plot(cost_l)
        
        return len(cost_l)
    
    
    
    
    
    
    
    
    def cost_vectorised(self):
        
        
        
        if np.shape(self.x0)[1] == np.shape(self.th)[0]:
        
            
            
            a = self.x0@self.th
            a = a.reshape(-1,1)
            self.th = np.reshape(self.th,(-1,1))
        
        
            if np.shape(a) == (np.size(self.x0,0),np.size(self.th,1)):
        
                b = a - self.y0
                
                if np.shape(b.T)[1] == np.shape(b)[0]:
                
                    c = b.T@b
                    cost = (1/(2*self.m))*c
                
                else:
                    print(f"{np.shape(b.T)[1]} must be equal to {np.shape(b)[0]}")
                
                
        
            else:
                print(f"vector a must be of size ({np.size(self.x0,0)},{np.size(self.th,1)})")
        
        
        else:
            print(f"{np.shape(self.x0)[1]} not equal to {np.shape(self.th)[0]}")
        
        
        return cost
    
    
    
    
    
    
    
    def gradient_descent_vectorised(self):
        
        
        e = np.reshape(self.x0@self.th,(-1,1)) - self.y0 
        
        #reshape into column vector
        self.th = np.reshape(self.th,(-1,1))
        
        
        c_k = []
        
        for i in range(self.n_iters):
            
            if np.shape(self.x0.T)[1] == np.shape(e)[0]:
                
                self.th = self.th - self.lr*(1/self.m)*(self.x0.T@e)
                
        
                cost_i = self.cost_vectorised()
                c_k.append(float(cost_i))
            
            
            else:
                print(f"num of cols {np.shape(self.x0.T)[1]} not equal to num of rows {np.shape(e)[0]}")
    
        plt.plot(c_k)
    
        return len(c_k)


# In[ ]:


#generate random data with seed for reproducibility
np.random.seed(43)
a = np.random.randn(1000,10)
a


# In[ ]:


np.random.seed(43)
b = np.random.randn(1000,1)
b


# In[ ]:


a = lr(a,b,0.001,1000)


# In[ ]:


a.cost_not_vectorised()


# In[ ]:


a.gradient_descent_not_vectorised()


# In[ ]:


a.cost_vectorised()


# In[ ]:


a.gradient_descent_vectorised()


# In[ ]:





# In[ ]:




