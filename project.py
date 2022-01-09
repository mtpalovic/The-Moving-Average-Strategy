#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

from numpy.random import randint

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import random

from datetime import datetime
import seaborn as sns

import pandas as pd
import math

import time as t

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (normalize, 
                                   StandardScaler, 
                                   MinMaxScaler, 
                                   LabelEncoder, 
                                   OneHotEncoder)

from sklearn.decomposition import PCA

from sklearn.feature_selection import (VarianceThreshold, 
                                       SelectKBest, 
                                       mutual_info_classif)

from sklearn.svm import SVC

from sklearn.ensemble import (AdaBoostClassifier, 
                              BaggingClassifier, 
                              StackingClassifier)

from sklearn.metrics import (confusion_matrix,
                             classification_report,
                             accuracy_score,
                             precision_score)

from sklearn.utils import shuffle

from sklearn.model_selection import (train_test_split, 
                                     GridSearchCV, 
                                     cross_validate)

from sklearn.utils.estimator_checks import check_estimator

from sklearn.utils.validation import (check_array, 
                                      check_is_fitted, 
                                      check_X_y,
                                      _check_sample_weight)

from sklearn.base import (BaseEstimator, ClassifierMixin)

from typing import Union


# In[2]:


#low (inclusive) to high (exclusive)
np.random.randint(1,2)


# In[63]:


class s():
    
    TEST_SIZE = 0.8
    N_ITERS = 1e3
    
    
    """
    :param c: controls the trade-off between 
        classifying all the points correctly and having a straight line, 
        small c cost of misclassif low, soft margin, 
        large c high cost of misclassif, hard margin 
    :type c: float
    
    :kernel: non-linearly separable data into higher dimension of spaces (linearly separable)
    :type kernel: str
    
    :degree: only relevant for poly kernel, ignored by all other kernels
    :type degree: int, optional
    
    :param gamma: how far the influence of of a training set goes, 
        high gamma = only points near the decision line are considered to determine the direction of the line, 
        close points to the line have high weight, 
        low gamma = points far away from the line are also considered
    :type gamma: float
    
    """
    
    
    def param_check(m):
        """Checks the type of init params (float, int, str). Custom decorator.
        :param m: init method
        :type m: 
        
        :return: init method after params are checked 
        :rtype: 
        """
        #ref refers to self
        def f(ref, estimator, k, C, gamma, random_number):
            for p in [C, gamma]:
                if not isinstance(p,(int,float)):
                    raise TypeError("c and gamma must be floats")
                
                else:
                    pass
                
                
            for v in [estimator, k]:
                if not isinstance(v,str):
                    raise TypeError("estim and k must be strings")
                else:
                    pass
            
            
            return m(ref, estimator, k, C, gamma, random_number)
        
        return f
    
    
    
    @param_check
    def __init__(self,
                 estimator:str = "SVC",
                 k:str = "linear",
                 C:int = 1000,
                 gamma = 1,
                 random_number:int = None):
        
        
        
        
        
        #kernel dict, self.kernels[self.k], self.k must exactly match the key in dict
        self.kernels = {
            "linear": self.kernel_linear,
            "rbf": self.kernel_rbf,
            "poly": self.kernel_poly
        }
        
        
        self.estimator = estimator
        
        self.gamma = self.d_check(gamma)
        
        self.random_number = random_number if random_number is not None else np.random.randint(0,100,size=None)
        
        #restrictions on the init params
        if k not in self.kernels.keys():
            raise AttributeError(f"kernel {k} required to be in {self.kernels.keys()}")
        else:
            self.k = k
            
        
        
        if 0.1 <= C <= 1000:
            self.C = C
        else:
            raise AttributeError(f"Param C:{C} not within required range")
            
            
            
            
    @staticmethod        
    def d_check(val):
        """Check init param type before it is set in the init method. Only int type.
        :param val: val in the init method
        :type val: 
        """
        if not isinstance(val,int):
            raise TypeError("val must be of type int")
        return val
    
    
    
    
    
    def cls_attr(self, a = "N_ITERS", l = None):
        """Check class attribute, not init (instance) attr. Other methods: getattr, setattr, hasattr.
        Init params are instance attributes, class attributes are defined outside constructor.
        :param u: defaults to None
        :type u: bool
        
        :raises AttributeError: if default not provided in getattr
        
        :return:
        :rtype:
        """
        
        
        if(l is None):
            
            try:
                k = getattr(s,a)
            
            except AttributeError as e:
                print(e)
        
        else:
            pass
        
        return float(k)
        
    
    
    def instance_attributes(self):
        """
        :return: Returns a list of instance attributes
        :rtype: list
        """
        return self.__dict__.keys()
        
        
    
    
    def __str__(self):
        """String representation of an object. Init params are instance attributes.
        :return: init param as string
        :rtype: str
        """
        return f"init params:{self.estimator},{self.k}, {self.C}, {self.gamma}, {self.random_number}"
    
        
    
    
    def create_tuple(self):
        """Creates a tuple
        :return:
        :rtype:
        """
        self.tuple = ()
        
        return self.tuple
    
    
    
    def create_arr(self, *args):
        """Creates a list
        :return:
        :rtype:
        """
        
        self.arr = [*args]
        
        return self.arr
    
    
    
    def create_dict(self):
        """Creates a dictionary
        :return:
        :rtype:
        """
        self.dict = {}
        
        return self.dict
    
    
    
    def load_data(self):
        path = "C:/Users/mpalovic/Desktop"
        ticker = "gspc"
        file_name = "ta.{}".format(str(ticker)) + ".csv"
        data = pd.read_csv(filepath_or_buffer = "{}/{}".format(path, file_name), 
                           parse_dates=["Date"], 
                           sep = ",")
        
        df = pd.DataFrame(data)
        
        df.sample(n=5)
        
        
        #choose dtypes
        df.astype(
                {"Volume":float,"On Balance Volume":float}
            ).dtypes

        df.select_dtypes(np.number)
        
        
        return df
    
    
    
    def datetime_index(self):
        """Transforms dates into cos,sin
        :return: dataframe with dates adjusted
        :rtype: dataframe
        """
        
        pi = float(math.pi)
        
        d = self.load_data()
        mo = d["Date"].dt.month
        
        f = self.create_arr("cos", "sin")
    
        for i,v in enumerate(f):
            if i == 0:
                d[v] = np.cos(2 * pi * mo / mo.max())
            elif i == 1:
                d[v] = np.sin(2 * pi * mo / mo.max())
        
        
        d.drop(labels="Date", axis = 1, inplace = True)
        
        return d
    
    
    
    
    
    
    def mis_vals(self):
        """Check missing values
        :return: dataframe with missing vals checked
        :rtype: dataframe
        """
        d = self.datetime_index()
        
        for _ in d.select_dtypes(np.number):
            i = SimpleImputer(missing_values = np.nan, strategy = "mean")
            d = pd.DataFrame(i.fit_transform(d), 
                             columns = d.columns)
        
        return d
    
    
    
    
    def x_y_(self):
        
        df = self.mis_vals()
        
        pred = (df.shift(-7)["Close"] >= df["Close"]) #bool, if price in 14 days bigger, returns 1
        df.drop("Close", 1, inplace=True)
        pred = pred.iloc[:-7]
        df["Pred"] = pred.astype(int)
    
        df = df.dropna()
        x = pd.DataFrame(df.loc[:,df.columns != "Pred"])
        y = df.iloc[:,-1]
    
        return x,y
    
    
    
    
    
    
    def label_encoding(self):
        
        x,y = self.x_y_()
        
        if isinstance(y,np.ndarray):
            y = LabelEncoder().fit_transform(y)
        
        
        return y
    
    
    
    
    def kernel_linear(self, x, y):
        return np.dot(x,y.T)
    
    
    
    def kernel_rbf(self, x,y):
        
        #initialise k
        arr = np.zeros(x.shape[0], y.shape[0])
        
        for i,x in enumerate(x):
            for j,y in enumerate(y):
                arr[i,j] = np.exp(-1*np.linalg.norm(x-y)**2)
        return arr
    
    
    
    def kernel_poly(self,x,y,p=3):
        return (1+np.dot(x,y))**p
    
    
    
    
    
    
    
    def feature_selection(self, num: Union[float,int] = 1.0*10e-3):
        
        
        x, _ = self.x_y_()
        c = []    
    
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)
        
        if num:
            
            #actual variance of cols
            x.var()

            #columns with variance higher than threshold will remain in dataframe
            #i want to remove low variance columns
            #TRUE low variance, FALSE high variance
            selector = VarianceThreshold(num)
            selector.fit_transform(x)
            selector.get_support()
            high_var_cols = [int(i) for i,e in enumerate(selector.get_support(indices = False))]
            high_var_cols
            #returns cols with above threshold variance, that is what I want


            x = pd.DataFrame(x)
            if isinstance(x, pd.DataFrame):
                c = [x for z,x in enumerate(x.columns) if z not in high_var_cols]

            x = np.array(
                x[x.columns[~x.columns.isin(c)]])
        
        
        
        else:
            pass
        
        #if num call this func, else call self.scale()
        return x if num else self.scale()
        
    
    
    
    
    
    
    def selectK(self, n:int=10):
        if n:
            x = self.feature_selection() 
            y = self.label_encoding()
        
        
            n = int(n)
    
            #created func above that creates an empty list
            high_score_features = self.create_arr()
            
            
            
            feature_scores = mutual_info_classif(x, y, random_state=self.random_number)
            
            
            
            
            #must be because of x train columns
            x = pd.DataFrame(x)
            
            if feature_scores.any():
                for score, col in sorted(zip(feature_scores, x.columns), reverse=True)[:n]:
                    #print(f_name, round(score, 4))
                    high_score_features.append(col)
                    x_ = x[high_score_features]
        
        else:
            pass
        
        
        
        return x_
    
    
    
    
    
    #takes no arguments and returns a dict of the __init__ parameters of the estimator, together with their values
    def get_params(self, deep = True):
        
        par = {
            "Est": self.estimator,
            "Kernel": self.k,
            "C": self.C,
            "Rand num": self.random_number
        } 
        
        return par
        
            
            
        
    def return_params(self):
        count = len(self.get_params().keys())
        count = int(count)
        
        params = [f"{param}:{value}" for param, value in sorted(self.get_params(deep=True).items())[:count]]
                
        return params
    
    
    
    
    def data_split(self):
        
        x_ = self.selectK()
        x = x_.copy()
        
        
        y = self.label_encoding()
        
        
        
    
        self.n_samples, self.n_features = x.shape
    
        
        x_train, x_test, y_train, y_test = train_test_split(x, 
                                                        y, 
                                                        test_size=s.TEST_SIZE, 
                                                        random_state=self.random_number, 
                                                        shuffle = True,
                                                        stratify=y
                                                    )
        
        return x_train, x_test, y_train, y_test
    
    
    
    def scale(self):
    
        x_train, x_test, _, _ = self.data_split() 
        scaler = StandardScaler()
        
        
        
        x_train = scaler.fit_transform(x_train.values.reshape(-1,self.n_features))
        x_test = scaler.transform(x_test.values.reshape(-1,self.n_features))
        
        
        return x_train, x_test
        
    
    
    
    
    def fit(self):
        
        #linear hyperparams
        linear_params = {
           "kernel": "linear",
            "C": 1e3
        }
        
        #poly hyperparams
        poly_params = {
            "kernel": "poly",
            "C": 1e3,
            "degree": np.random.randint(2,3) #low inclusive, high exclusive
        }
        
        #rbf hyperparams
        rbf_params = {
            "kernel": "rbf",
            "C": 1e3,
            "gamma": 0.1
        }
        
        
        
        
        #the below func must run before fit, without parenthesis
        if callable(self.scale):
        
            #import from here 
            x_train, x_test = self.scale()
        
        
        
            #pass an an np.ndarray
            x_train = np.array(x_train)
        
        
            if isinstance(x_train, np.ndarray):
            
                #do not import x_train, x_test because I have modified it
                _, _, y_train, y_test = self.data_split()
            
            
                #checks if kernel exists
                if self.k in self.kernels.keys():
                
                    estimator = SVC(kernel = self.kernels[self.k])
                    model = estimator.fit(x_train, y_train)
            
            
                    y_pred = model.predict(x_test)
                    self.accuracy = accuracy_score(y_pred, y_test)
            
            
                else:
                    print(f"{self.k} required to be in {self.kernel.keys()}")
                    
                    if self.k == "linear":
                        SVC(C = linear_params["C"], kernel = linear_params["kernel"])
                    
                    elif self.k == "poly":
                        SVC(C = poly_params["C"], kernel = poly_params["kernel"], degree = poly_params["degree"])
                        
                    else:
                        SVC(C = rbf_params["C"], kernel = rbf_params["kernel"], gamma=rbf_params["gamma"])
                    
                    
    
            else:
                pass
        
        else:
            pass
        
        
        
        return self.accuracy
    
    
    
    
    
    def accuracy_score(self):
        
        
        x_train, x_test = self.scale()
        _, _, y_train, y_test = self.data_split()
        
        
        if isinstance(x_train, np.ndarray):
            
            #creates dict
            classifiers = self.create_dict()
            
            for i in list(self.kernels.keys())[:len(self.kernels.keys())]:
                classifiers[i] = SVC(kernel=i)
        
        
        
        
            accr = self.create_dict()
        
            for algorithm, classifier in classifiers.items():
                classifier.fit(x_train, y_train)
                y_pred = classifier.predict(x_test)
                accr[algorithm] = accuracy_score(y_pred, y_test)
            
        
                
            for algorithm, accuracy in sorted(accr.items())[:len(accr.keys())]:
            
                #notice must be 2x %% in order to show % in print
                print("%s Accuracy: %.2f%%" % (algorithm, 100*accuracy))
            
        
        
        return accr
    
    
    
        
        
        
        
    def initialise_params(self):
        """
        :param arg1: description
        :type arg1: int
        :return: initialise params
        :rtype: int
        """
        
        
        _, n_features = x.shape
        w = np.zeros(n_features)
        b = 0
        return w, b


# In[64]:


if __name__ == "__main__":
    m = s(estimator="SVC", k = "linear", C = 1000, gamma = 1, random_number=None)
    m.__str__()
    m.instance_attributes()
    #m.create_arr()
    #m.load_data()
    #m.datetime_index()
    #m.mis_vals()
    #m.x_y_()
    #m.label_encoding()
    #m.data_split()
    #m.scale()
    #m.selectK()
    #m.get_params()
    #m.return_params()
    #m.feature_selection()
    #m.fit()
    #m.accuracy_score()
    #m.create_tuple()


# In[65]:


print(m)


# In[51]:


m.__dict__.keys()


# In[52]:


m.instance_attributes()


# In[53]:


m.cls_attr()


# In[66]:


m.datetime_index()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def gridSearchCV(self):
        
        x_train, x_test = self.scale()
        _, _, y_train, y_test = self.data_split()
        
        
        param_grid = {
            "C": [0.1,1,10,100,1000,1000],
            "gamma": [1,0.1,0.01,0.001],
            "kernel": ["linear", "poly", "rbf"]
        }
        
        search = GridSearchCV(estimator = SVC(), 
                             param_grid = param_grid, 
                             cv = 5, #int, specify number of folds in StratifiedKfold
                             verbose = 4, # control verbosity, the higher the more messages
                             refit = True, # refit an estimator using the best found params on data
                             scoring = "accuracy")
        
        search.fit(x_train, y_train)
        #y_pred = search.predict(x_test)
        
        #print("Test Accuracy: {}.".format(accuracy_score(y_test, y_pred)))

