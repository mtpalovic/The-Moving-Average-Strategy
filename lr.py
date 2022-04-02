#!/usr/bin/env python
# coding: utf-8

# $$h_{\theta}(x^{i}) = \theta_{0} x_{0}^i+\theta_{1} x_{1}^i+\theta_{2} x_{2}^i+\theta_{j} x_{j}^i \tag{1}$$

# $$J(\theta) = \frac{1}{2m}\sum\limits_{i=1}^{m}(h_{\theta}(x^{i})-y^{i})^2 \tag{2}$$

# $$J(\theta) = \frac{1}{2m}\sum\limits_{i=1}^{m}((\theta_{0} x_{0}^i+\theta_{1} x_{1}^i+\theta_{2} x_{2}^i+\theta_{j} x_{j}^i)-y^{i})^2 \tag{3}$$

# $$\frac{\partial J(\theta)}{\partial \theta_{j}} = \frac{1}{m}\sum\limits_{i=1}^{m}(h_{\theta}(x^{i})-y^{i})x_{j}^i \tag{4}$$

# $$\frac{\partial J(\theta)}{\partial \theta_{j}} = \frac{1}{m}\sum\limits_{i=1}^{m}((\theta_{0} x_{0}^i+\theta_{1} x_{1}^i+\theta_{2} x_{2}^i+\theta_{j} x_{j}^i)-y^{i})x_{j}^i \tag{5}$$

# $$\theta_{j} = \theta_{j} - \alpha \frac{1}{m}\sum\limits_{i=1}^{m}((\theta_{0} x_{0}^i+\theta_{1} x_{1}^i+\theta_{2} x_{2}^i+\theta_{j} x_{j}^i)-y^{i})x_{j}^i \tag{6}$$

# $$J(\theta) = \frac{1}{2m}(x \theta - y)^T(x \theta - y) \tag{7}$$

# $$\frac{\partial J(\theta)}{\partial \theta} = \frac{1}{m}x^T(x \theta - y) \tag{8}$$

# In[ ]:


import numpy as np
import random
random.seed(63)

import matplotlib.pyplot as plt


# In[ ]:


#generate random data with seed for reproducibility
np.random.seed(63)
a = np.random.randn(1000,10)


# In[ ]:


np.random.seed(63)
b = np.random.randn(1000,1)


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
        assert(type(x0)==np.ndarray and type(y0)==np.ndarray) 
        
        
        self.lr = lr
        assert type(lr)==float
        
        
        self.n_iters = n_iters
        self.m, self.n = np.shape(self.x0)[0], np.shape(self.x0)[1]
        
        # 1D array of size n_features
        #initialise theta
        #column vector of of shape(10,1)
        self.th = np.random.randn(self.n,1)
        
    
        
    
    
    def cost_not_vectorised(self, theta):
        
        #this indicates that cost will be a float
        #calculate cost at each row or sample and then sum
        c = 0
        for i in range(self.m):    
            
            #y_hat prediction for each row with all estimated parameters
            #at each row, iterate over all columns/parameters stored in theta
            y_hat = 0
            for j in range(self.n):
                y_hat += theta[j]*self.x0[i][j]
            
            #np.power because of neg distance and punishing large outliers
            #cost for row i, substract actual from prediction
            c_i = (y_hat - self.y0[i])**2
            c += c_i
            c = float(c)
        
        
        #out of for loop
        cost = (1/(2*self.m))*c
        
        return cost
    
    
    
    
    def gradient_descent_not_vectorised(self):
        
        np.random.seed(63)
        self.th = np.random.randn(self.n,1)
        
        cost_list = []
        
        
        for _ in range(self.n_iters):
            
            
            derivatives_list = []
            for k in range(self.n):
                
                
                #sum derivatives from all rows
                d_j = 0
                for i in range(self.m):
                    
                    #estimate
                    y_hat = 0
                    for j in range(self.n):
                        y_hat += self.th[j]*self.x0[i][j]
                    
                    #derivative for each row
                    #cannot be indexed by j because out of that for loop at this step
                    #index by k because derivative for k parameter 
                    
                    #derivative of theta j at row i
                    d_i = (y_hat - self.y0[i])*self.x0[i][k]
                    
                    #sum all rows and store as d_j, derivative of theta j
                    d_j += d_i 
                
                
                #once iterative over all 10 theta parameters
                #append derivative for each column (of each parameter) to a list
                derivatives = (1/self.m)*d_j
                
                #derivatives = float(derivatives)
                derivatives_list.append(derivatives)
            
            
            
            #update params stored as self.th acc to lr rate
            #both self.th and np.array(derivatives_list) are column vectors of shape (10,1)
            #indexed by z, hence result is a column vector of original shape
            
            derivatives_list = np.array(derivatives_list).reshape(-1,1)
            for z in range(len(self.th)):
                self.th[z] = self.th[z] - self.lr*derivatives_list[z]
            
            
            cost = self.cost_not_vectorised(self.th)
            cost_list.append(cost)
            
            plt.plot(cost_list)
        
        return cost_list
    
    
    
    
    
    
    
    def cost_vectorised(self, theta):
        
        
        
        #self.x0 of shape (1000,10) and self.th of shape (10,1)
        #the matrix multiplication result of shape (1000,1)
        if np.shape(self.x0)[1] == np.shape(theta)[0]:
        
            a = self.x0@theta
            a = a.reshape(-1,1) #shape (1000,1)
            
            theta = np.reshape(theta,(-1,1)) #to be sure that shape (10,1)
            
            
            #vector shape (1000,1) necessary for column vector substraction in vectorised form
            if np.shape(a) == (np.size(self.x0,0),np.size(theta,1)):
        
                #all three column vectors of shape (1000,1)
                b = a - self.y0
                
                if np.shape(b.T)[1] == np.shape(b)[0]:
                
                    #b.T of shape (1,1000) and b of shape (1000,1)
                    #matrix multiplication result of shape (1,1), which is a scalar
                    #equation 7
                    c = float(b.T@b)
                
                else:
                    #if this not true, matrix multiplication cannot be calculated
                    print(f"{np.shape(b.T)[1]} must be equal to {np.shape(b)[0]}")
                
                
        
            else:
                #check if a = x theta is of correct shape
                print(f"vector a must be of size ({np.size(self.x0,0)},{np.size(theta,1)})")
        
        
        else:
            #if this not true, matrix multiplication cannot be calculated
            print(f"{np.shape(self.x0)[1]} not equal to {np.shape(theta)[0]}")
        
        
        cost = (1/(2*self.m))*c
        
        
        return cost
    
    
    
    
    
    
    
    def gradient_descent_vectorised(self):
        
        
        np.random.seed(63)
        self.th = np.random.randn(self.n,1)
        
        
        #self.x0 shape (1000,10) and self.th (10,1)
        #the product of matrix multiplication is of shape (1000,1)
        #self.y0 shape (1000,1)
        #e hence of shape (1000,1)
        e = np.reshape(self.x0@self.th,(-1,1)) - self.y0 
        
        
        #reshape into column vector
        #to ensure that shape (10,1)
        self.th = np.reshape(self.th,(-1,1))
        
        
        cost_list1 = []
        
        for i in range(self.n_iters):
            
            #if np.shape(self.x0.T)[1] == np.shape(e)[0]:
            
                
            #update params acc to learning rate 
            #old param value minus (learning rate x derivative) 
            self.th = self.th - self.lr*(1/self.m)*(self.x0.T@e)
            
            
            #else:
                #print(f"num of cols {np.shape(self.x0.T)[1]} not equal to num of rows {np.shape(e)[0]}")
    
    
    
    
            cost_i = self.cost_vectorised(self.th)
            cost_list1.append(cost_i)
    
    
        plt.plot(cost_list1)
    
        return cost_list1


# In[ ]:


a = lr(a,b,0.001,1000)


# In[ ]:


#a.cost_not_vectorised()


# In[ ]:


a.gradient_descent_not_vectorised()


# In[ ]:


#a.cost_vectorised()


# In[ ]:


a.gradient_descent_vectorised()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




