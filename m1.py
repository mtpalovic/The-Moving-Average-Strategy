#!/usr/bin/env python
# coding: utf-8

# In[48]:


import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import sys
np.set_printoptions(threshold=sys.maxsize)


from datetime import datetime
import seaborn as sns
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import math

#feature preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (normalize,
                                   StandardScaler,
                                   MinMaxScaler,
                                   LabelEncoder,
                                   OneHotEncoder
                                  )
from sklearn.decomposition import PCA

#feature selection
from sklearn.feature_selection import (VarianceThreshold,
                                       SelectKBest, 
                                       mutual_info_classif,
                                       RFE,
                                       SelectFromModel,
                                       SequentialFeatureSelector
                                      )
#model selection
from sklearn.svm import SVC
from sklearn.ensemble import (AdaBoostClassifier, 
                              BaggingClassifier, 
                              StackingClassifier
                             )

#model evaluation
from sklearn.metrics import (confusion_matrix,
                             classification_report,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score
                            )

from sklearn.utils import shuffle
from sklearn.utils.multiclass import unique_labels
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
from sklearn.base import (BaseEstimator, TransformerMixin)

import warnings
warnings.filterwarnings("ignore")

import module


# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


np.random.randint(0,100,size=None)


# In[81]:


class svm():

    #constants are uppercase
    NUM = 1e-3
    
    
    def __init__(self,
                 estimator:str = "SVC",
                 kernel:str = "linear",
                 C:int = 1e3,
                 n_iters:int = 1e3,
                 random_number:int = None):
        
        
        
        
        self.estimator = estimator
        self.kernel = kernel                     #self._kernel_type(kernel, **kwargs)
        self.C = C 
        self.n_iters = n_iters
        self.random_number = random_number if random_number is not None else np.random.randint(0,100,size=None)
                
    def __repr__(self):
        return self.estimator.__repr__()
    
    def __str__(self):
        return self.estimator.__str__()
    
    
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
        pred = (df.shift(-14)["Close"] >= df["Close"]) #bool, if price in 14 days bigger, returns 1
        df.drop("Close", 1, inplace=True)
        pred = pred.iloc[:-14]
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
    
    
    
    
    def data_split(self):
        x,y = self.x_y_()
    
        self.n_samples, self.n_features = x.shape
    
        
        x_train, x_test, y_train, y_test = train_test_split(x, 
                                                        y, 
                                                        test_size=0.7, 
                                                        random_state=42, 
                                                        shuffle = True,
                                                        stratify=y
                                                           )
        
        return x_train, x_test, y_train, y_test
    
    
    
    def scale(self):
    
        x_train, x_test, y_train, y_test = self.data_split() 
        
        scaler = StandardScaler()
        
        
        
        x_train = scaler.fit_transform(x_train.values.reshape(-1,self.n_features))
        x_test = scaler.transform(x_test.values.reshape(-1,self.n_features))
        
        
        return x_train, x_test, y_train, y_test
    
    
    
    def feature_selection(self, num:float = 1.0*10e-3):
        
        if num:
            x_train, x_test, y_train, y_test = self.scale()

            c = []    
    
            scaler = MinMaxScaler()
            x_train = scaler.fit_transform(x_train)
        
        
            #actual variance of cols
            x_train.var()

            #columns with variance higher than threshold will remain in dataframe
            #i want to remove low variance columns
            #TRUE low variance, FALSE high variance
            selector = VarianceThreshold(num)
            selector.fit_transform(x_train)
            selector.get_support()
            high_var_cols = [i for i,e in enumerate(selector.get_support(indices = False)) if e]
            high_var_cols
            #returns cols with above threshold variance, that is what I want


            x_train = pd.DataFrame(x_train)
            if isinstance(x_train, pd.DataFrame):
                c = [x for z,x in enumerate(x_train.columns) if z not in high_var_cols]

            x_train = np.array(
                x_train[x_train.columns[~x_train.columns.isin(c)]])
            
        else:
            pass
        
        #if num call this func, else call self.scale()
        return x_train if num else self.scale()
    
    
    
    def selectK(self, n:int=8):
        if n:
            x_train, x_test, y_train, y_test = self.scale()
        
            n = int(n)
    
            high_score_features = []
            feature_scores = mutual_info_classif(x_train,y_train, random_state=self.random_number)
            
            
            
            
            #must be because of x train columns
            x_train = pd.DataFrame(x_train)
            
            if feature_scores.any():
                for score, col in sorted(zip(feature_scores, x_train.columns), reverse=True)[:n]:
                    #print(f_name, round(score, 4))
                    high_score_features.append(col)
                    x_train_ = x_train[high_score_features]
        
        else:
            pass
        
        
        
        return x_train_


# In[82]:


if __name__ == "__main__":
    model = svm(estimator="SVC", kernel = "linear", C = 1000, n_iters=1000)
    
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
    model.feature_selection()
    
    #select columns
    model.selectK()


# In[84]:


y = model.label_encoding()
y

mask = 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#assign a function from another module to some variable if used often
add_fund_from_another_module = module.add
add_fund_from_another_module(10,10)

a, *_, c  = (1,2,3,4,5)
c


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#svm fit method

#if not isinstance(x_train, type(None)):


x_train_transformed = selectK(x_train)


x_train, x_test, y_train, y_test = d()
y_train


x_train_transformed = np.array(x_train_transformed)
y_train = np.array(y_train)


model = SVC()
    

m = model.fit(x_train,y_train)
y_pred = model.predict(x_test)
    
accuracy_score(y_pred, y_test)

model.feature_names_in


# In[ ]:





# In[ ]:


def _kernel_type(self, kernel:str = None, **kwargs):
        """Kernel is a hyperparameter and is selected by the researcher."""
        
        if self.estimator == "SVC":
        
            if kernel:
                if isinstance(kernel, str) and kernel is not None:
                
                    if kernel == "linear":
                        def linear_kernel():
                            return lambda X,y: np.dot(X,y.T)
        
                    if kernel == "poly":
                        def _polynomial_kernel(bias = 0, power = 2):
                            return lambda X,y: (self.gamma * np.dot(X,y)+bias)**power
        
                    if kernel == "rbf":
                        def _rbf_kernel():
                            return lambda X_i,y_i: np.exp(-self.gamma * np.dot(X_i-y_i, X_i-y_i))  
        
            kernel_mapping = {
                "linear": _linear_kernel,
                "poly": _polynomial_kernel,
                "rbf": _rbf_kernel
            }
            
            if kernel not in {None,"linear", "poly", "rbf"}:
                raise ValueError(f"{self.kernel} kernel not recognised.")

        return 


# In[ ]:


def predict(self, X_test:np.array, y = None):
        """Step 7: Model prediction."""
        if isinstance(X_test, np.array) and not y:
            check_is_fitted(self, msg="is_fitted")          
        
        
        
        
        
        
        y_pred = self.base_regressor.predict(X_test)
        
        if self.n:
            linear_output = np.sign(np.dot(np.array(n_features), self.w) + self.b)
        
        
        return y_pred if self.n is False else linear output
        
        
    
    def evaluation(self):
        """The following function evaluates how well the model performs on the test data. Step 6: model evaluation."""
        print("\nThe Classifier Accuracy Score is {:.2f}\n".format(clf.score(X_test, y_test)))
        
    def get_params(self, deep = True):
        """
        The below function returns parameter values.
        """
        return {
            "C": self.C,
            "kernel": self.kernel
            "epsilon": self.epsilon
        }
    
    def set_params(self, **params):
        for param, val in params.items():
            setattr(self, param, val)
        return self    
    
    def gridSearchCV(self):
        param_grid = {
            "C": [1,10,100,1000,10000],
            "gamma": [1,0.1,0.01,0.001,0.00001],
            "kernel": ["linear", "poly", "rbf"],
            "class_weight": ["balanced", None]
        }
        
        search = GridSearchCV(estimator = svm, 
                             param_grid = param_grid, 
                             cv = 5, # determines cross-validation splitting strategy, int, specify number of folds in StratifiedKfold
                             verbose = 1, # control verbosity, the higher the more messages
                             refit = True, # refit an estimator using the best found params on data
                             scoring = accuracy)
        
        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)
        
        print("Test Accuracy: {}.".format(accuracy_score(y_test, y_pred)))
        print("Best Params: {}.".format(model.best_params_))


# In[ ]:


def main(a,b,q):
    if q:
        e = None
        def plus(a,b):
            e = a+b
            return "Ahoj volam sa martin"
    else:
        E = None
        def minus(a,b):
            E = a-b
            return E
    return plus(a,b) if q is not None else minus(a,b)


# In[ ]:


main(40,10, q = "ok")


# In[ ]:





# In[ ]:


def color_neg_vals(val):
    color = "red" if val > 1400 else "black"
    return "color:%s"%color

# style with pandas dataframe
#df["Close"].to_frame().style.applymap(color_neg_vals)

#string, notice col name
#df.query("Close > 4695")

def outliers(df, n_std:int=3):
    
    df = pd.DataFrame(df)
    df_ = df.copy()
    
    
    for col in range(len(df_.columns)):
        
        mean, std = np.mean(df_[col]), np.std(df_[col])
        mean = float(mean)
        std = float(std)

        cut_off = std*n_std
        lower, upper = mean - cut_off, mean + cut_off
        outliers = [x for x in df_[col] if x < lower or x > upper]
        print(f"{col}: {len(outliers)}")
        df_ = [x for x in df[col] if x not in outliers]
    
    
    return df_


# In[ ]:



df.mask(df.isna(),0)

mask = df["Close"] >4200
df["Close"][mask]


# In[ ]:


def outliers(self, num:int=3):
        df = self.datetime_index()
        
        df = pd.DataFrame(df)
        df_ = df.copy()


        for col in df.columns:
            mean, std = np.mean(df[col])
            std = np.std(df[col])
            mean = float(mean)
            std = float(std)

            cut_off = std*num
            lower, upper = mean - cut_off, mean + cut_off
            
            outliers = [x for x in df[col] if x < lower or x > upper]
            
            print(f"{col}: {len(outliers)}")
            
            df = [x for x in df[col] if x not in outliers]
    
        return df


# In[ ]:


def dimensionality_reduction(var_retained:float = .95, r = None):
        if r:
            x_train, x_test, y_train, y_test = scale()
            pca = PCA(var_retained)
        
            x_train = pca.fit_transform(x_train)
            x_test = pca.transform(x_test)
                                                                    
            explained_variance = pca.explained_variance_ratio_

    return explained_variance

