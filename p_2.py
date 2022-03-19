#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
from scipy import linspace, polyval, polyfit, sqrt, stats, randn, optimize


import pandas as pd
import matplotlib.pyplot as plt

import csv

import scipy
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


with open(r"C:\Users\mpalovic\Desktop\d.csv") as f:
    r = csv.reader(f)
    
    table = []
    
    
    for row in r:
        print(row)
print(r)


# In[41]:


data= pd.read_csv(r"C:\Users\mpalovic\Desktop\d.csv")
data.iloc[]


# In[47]:


lp = {}
for i,v in enumerate(data.columns.values):
    lp.append(i,v)


# In[48]:


lp


# In[ ]:


class Brownian():
    """
    Class constructor.
    """
    def __init__(self,x0):
        """
        Init class.
        """
        assert (type(x0)==float or type(x0)==int or type(x0)==None)
        
        self.x0 = float(x0)

