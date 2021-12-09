#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import sys
np.set_printoptions(threshold=sys.maxsize)


from datetime import datetime
import seaborn as sns
import pandas as pd
import math

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (normalize,
                                   StandardScaler,
                                   MinMaxScaler,
                                   LabelEncoder,
                                   OneHotEncoder
                                  )
from sklearn.decomposition import PCA


from sklearn.feature_selection import (VarianceThreshold,
                                       SelectKBest, 
                                       mutual_info_classif,
                                       RFE,
                                       SelectFromModel,
                                       SequentialFeatureSelector
                                      )

from sklearn.svm import SVC
from sklearn.ensemble import (AdaBoostClassifier, 
                              BaggingClassifier, 
                              StackingClassifier
                             )


from sklearn.metrics import (confusion_matrix,
                             classification_report,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score
                            )


from sklearn.utils import shuffle

from sklearn.model_selection import (train_test_split,
                                     GridSearchCV,
                                     cross_validate
                                    )

from sklearn.utils.estimator_checks import check_estimator

from sklearn.utils.validation import (check_array, 
                                      check_is_fitted, 
                                      check_X_y,
                                      _check_sample_weight
                                     )

from sklearn.base import (BaseEstimator, ClassifierMixin)

import warnings
warnings.filterwarnings("ignore")




import module


# In[ ]:


np.random.randint(0,100,size=None)


# In[ ]:


class svm(BaseEstimator, ClassifierMixin):

    #constants are uppercase
    NUM = 1e-3
    
    
    def __init__(self,
                 estimator:str = "SVC",
                 kernel_type:str = "poly",
                 C:int = 10,
                 n_iters:int = 1e3,
                 random_number:int = None,
                 test_size = 0.8):
        
        
        
        
        self.estimator = estimator
        self.kernel_type = kernel_type                     
        self.C = C 
        self.n_iters = n_iters
        self.random_number = random_number if random_number is not None else np.random.randint(0,100,size=None)
        self.test_size = test_size
        
        
        #pozor nizsie zadej func self.kernel_linear
        #self.kernel_type must exactly match the key in dict a potom zadef func below
        
        self.kernels = {
            "linear": self.kernel_linear,
            "rbf": self.kernel_rbf,
            "poly": self.kernel_poly
        }
        
        
    
    def dec(original):
        def wrap(*args,**kwargs):
            print(f"wrap executed before {original.__name__}")
            
            
            
            return original(*args,**kwargs)
        return wrap
    
    
    
    
    
    
    
    def check_attr(self):
        ins = None
        
        #check class instance
        if hasattr(svm,"NUM"):
            ins = svm.NUM
        return ins
    
    
    @dec
    def __str__(self):
        
        
        #returns the name of the estimator
        return self.estimator.__str__()
            
    
    
    def create_arr(self):
        self.arr = []
        
        return self.arr
    
    
    
    def create_dict(self):
        self.dict = {}
        
        return self.dict
    
    
    def load_data(self):
        path = "C:/Users/mpalovic/Desktop"
        ticker = "gspc"
        file_name = "ta.{}".format(str(ticker)) + ".csv"
        data = pd.read_csv(filepath_or_buffer = "{}/{}".format(path, file_name), parse_dates=["Date"], sep = ",")
        df = pd.DataFrame(data)
        
        df.sample(n=5)
        
        
        #choose dtypes
        df.astype(
            {"Volume":float,"On Balance Volume":float}
        ).dtypes

        df.select_dtypes(np.number)
        
        
        return df
    
    
    
    def datetime_index(self):
    
        df = self.load_data()
    
        df["m"] = df["Date"].dt.month
        df["d"] = df["Date"].dt.day

        #datetime index cannot be used as an input to a machine learning model
        #cyclical feature encoding for the date column
        df["cos"] = np.cos(2*math.pi*df["Date"].dt.month/df["Date"].dt.month.max())
        df["sin"] = np.sin(2*math.pi*df["Date"].dt.month/df["Date"].dt.month.max())
        df.drop(labels="Date", axis = 1, inplace = True)
        
        return df
    
    
    
    
    
    
    def mis_vals(self):
        df = self.datetime_index()
        
        i = SimpleImputer(missing_values = np.nan, strategy = "mean")
        df = pd.DataFrame(i.fit_transform(df), columns = df.columns)
        
        return df
    
    
    
    
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
    
    
    
    
    
    
    
    def feature_selection(self, num:float = 1.0*10e-3):
        
        
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
            high_var_cols = [i for i,e in enumerate(selector.get_support(indices = False))]
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
            "Kernel": self.kernel_type,
            "C": self.C,
            "n_iters": self.n_iters,
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
                                                        test_size=self.test_size, 
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
        
        #import from here 
        x_train, x_test = self.scale()
        
        
        
        #pass an an np.ndarray
        x_train = np.array(x_train)
        
        
        if isinstance(x_train, np.ndarray):
            
            #do not import x_train, x_test because I have modified it
            _, _, y_train, y_test = self.data_split()
            
            
            
            estimator = SVC(kernel = self.kernels[self.kernel_type])
            
            
            model = estimator.fit(x_train, y_train)
            
            
            y_pred = model.predict(x_test)
            self.accuracy = accuracy_score(y_pred, y_test)
        
        
        #for _ in self.iters
        
        
        return self.accuracy
    
    
    
    
    
    def acrc(self):
        
        x_train, x_test = self.scale()
        _, _, y_train, y_test = self.data_split()
        
        
        if isinstance(x_train, np.ndarray):
            #creates dict
            classifiers = self.create_dict()
        
        
            list_classif = ["linear", "rbf", "poly"]
            
            for i in list_classif:
                classifiers[i] = SVC(kernel=i)
        
        
        
        
            accuracies = self.create_dict()
        
            for algorithm, classifier in classifiers.items():
                classifier.fit(x_train, y_train)
                y_pred = classifier.predict(x_test)
                accuracies[algorithm] = accuracy_score(y_pred, y_test)
            
        
                
            for algorithm, accuracy in sorted(accuracies.items())[:len(accuracies.keys())]:
            
                #notice must be 2x %% in order to show % in print
                print("%s Accuracy: %.2f%%" % (algorithm, 100*accuracy))
            
        
        
        return accuracies
    
    
    
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
        
        
        
        
        
        
        
    def initialise_params(self):
        _, n_features = x.shape
        w = np.zeros(n_features)
        b = 0
        return w, b
    
    


# In[ ]:


if __name__ == "__main__":
    model = svm(estimator="SVC", kernel_type = "linear", C = 1000, n_iters=1000)
    
    #class attribute check
    model.check_attr()
    
    #return the name of the estimator
    model.__str__()
    
    #creates an empty list
    model.create_arr()
    
    #loda data
    model.load_data()
    
    #remove datetime index as scikit cannot work with date type
    model.datetime_index()
    
    #input missing values as mean
    model.mis_vals()
    
    #create x,y
    model.x_y_()
    
    #y label encoding
    model.label_encoding()
    
    #train test split
    model.data_split()
    
    #scale features to common scale
    model.scale()
    
    #select columns
    model.selectK()
    
    #returns init vals as a dict
    model.get_params()
    
    ##returns init vals as a list
    model.return_params()
    
    #select columns
    model.feature_selection()
    
    #fit
    model.fit()
    
    #model accuracy
    model.acrc()
    
    #hyperparams
    model.gridSearchCV()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




