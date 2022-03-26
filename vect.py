#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


class non_vect(object):
    """
    Class constructor.
    """
    
    def __init__(self,x0:np.ndarray, y0:np.ndarray,lr:float=0.005,n_iters:int=1000):
        """
        Constructor method.
        """
        self.x0 = x0
        self.y0 = y0
        assert (type(x0)==np.ndarray and type(x0)==np.ndarray)
        
        
        self.lr = lr
        assert type(lr)==float
        
        
        self.n_iters = n_iters
        self.m, self.n = np.shape(self.x0)[0], np.shape(self.x0)[1]
        
        # 1D array of size n_features
        #initialise theta
        self.th = np.ones(self.n)
        
        

       
    def cost(self):
        """
        Cost function. Predicts cost of one estim value and all data points.
        
        :return: cost associated with estimated value of y
        :rtype: float
        
        """
        
        c = 0
        for i in range(self.m):    
            y_hat = 0
            
            for j in range(self.n):
                y_hat += self.th[j]*self.x0[i][j]
            
            #np.power because of neg distance and punishing large outliers
            #cost for each obs i
            c_i = np.power((y_hat - self.y0[i]),2)
            c += c_i
        
        cost = (1/2*self.m)*c
        
        return cost
    
    
    
    
    def derivative(self):
        """
        Gradient descent using a for loop.
        
        :return: cost associated with estimated value of y
        :rtype: float
        
        """
        
        #calc cost for each iteration as gradient descent continues
        cost_l = []
        for _ in range(self.n_iters):
            
            #only one derivative for each column/feature
            dev_ = []
            for k in range(self.n):
                
                #sum derivatives from all rows in a single col
                dev_sum = 0
                for i in range(self.m):
                    
                    #estimate
                    y_hat = 0
                    for j in range(self.n):
                        y_hat += self.th[j]*self.x0[i][j]
                    
                    #derivative for each row 
                    d_i = (y_hat - y[i])*self.x0[i][k]
                    dev_sum += d_i 
                
                #append to list
                #derivative for each column
                dev = (1/self.m)*dev_sum 
                dev_.append(dev)
            
            
            #update params stored as self.th acc to lr rate
            self.th = self.th - self.lr*np.array(dev_)
            
            cos = self.cost()
            
            cost_l.append(cos)
        
        return cost_l


# In[ ]:


from sklearn.datasets import load_diabetes
x,y = load_diabetes(return_X_y=True)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

X = NormalizeData(x)

a = non_vect(X,y,0.005,100)
e = a.derivative()

print(e)


plt.plot(np.arange(0, 100),e)


# In[ ]:




