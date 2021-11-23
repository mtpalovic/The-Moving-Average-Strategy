#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from datetime import datetime
import seaborn as sns
import pandas as pd
import math

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (StandardScaler,
                                   MinMaxScaler,
                                   Normalizer,
                                   LabelEncoder,
                                   OneHotEncoder
                                  )

from sklearn.decomposition import PCA

from sklearn.feature_selection import (VarianceThreshold,
                                       SelectKBest, mutual_info_classif, chi2,
                                       RFE,
                                       SelectFromModel,
                                       SequentialFeatureSelector
                                      )

from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, StackingClassifier)

from sklearn.utils import shuffle
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import SVC

from sklearn.model_selection import (train_test_split,
                                     GridSearchCV,
                                     cross_validate
                                    )

from sklearn.metrics import (confusion_matrix,
                             classification_report,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score
                            )

from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import (check_array, 
                                      check_is_fitted, 
                                      check_X_y,
                                      _check_sample_weight
                                     )

from sklearn.base import (BaseEstimator, 
                          TransformerMixin
                         )

import warnings
warnings.filterwarnings("ignore")


# In[20]:


path = "C:/Users/mpalovic/Desktop"
ticker = "gspc"
file_name = "ta.{}".format(str(ticker)) + ".csv"
df = pd.read_csv(filepath_or_buffer = "{}/{}".format(path, file_name), parse_dates=["Date"], sep = ",")

df["y"] = df["Date"].dt.year
df["m"] = df["Date"].dt.month
df["d"] = df["Date"].dt.day

#datetime index cannot be used as an input to a machine learning model
#cyclical feature encoding for the date column
df["cos"] = np.cos(2*math.pi*df["Date"].dt.month/df["Date"].dt.month.max())
df["sin"] = np.sin(2*math.pi*df["Date"].dt.month/df["Date"].dt.month.max())
df.drop(labels="Date", axis = 1, inplace = True)

#missing values are replaced with mean
i = SimpleImputer(missing_values = np.nan, strategy = "mean")
df = pd.DataFrame(i.fit_transform(df), columns = df.columns)

x = pd.DataFrame(df.loc[:,df.columns != "Close"])
y = df.iloc[:,1]
x.sample(10)


# In[6]:


df.columns
df.columns.to_list()
df.dtypes
df.info()
df.drop_duplicates()
df.shape
df.describe()
df.isnull().sum()
df.index
df.nunique()
df.sample(n=5)
df.corr()
df.astype(
    {"Volume":float,"On Balance Volume":float}).dtypes
df.sort_values(by="RSI", ascending=False)
df.size
df.select_dtypes("object").head(5)
df.mask(df.isna(),0)

mask = df["Close"] >4200
df["Close"][mask]


# In[ ]:


x.to_numpy()
y = np.array(np.where(df["Close"] > df["Close"].shift(-7),1,0))


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42, shuffle = True,stratify=y)
clf = SVC(C=1000.0, kernel = "rbf")
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
y_pred.tolist()


# In[24]:


df = pd.DataFrame(df)
df.sample(100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


class svm(BaseEstimator, TransformerMixin):

    def __init__(self,
                 estimator:str = "SVC",
                 kernel:str = "linear",
                 C:int = 1e3,
                 random_number:int = None):
        
        
        self.estimator = estimator
        self.kernel = self._kernel_type(kernel, **kwargs)
        
        self.C = C if C is not None
        self.random_number = random_number if random_number is not None else np.random.randint(0,100,size=1)
        
        
        
        
        
        
    
    
    
    def decorator_function(original_function):
        def wrapper_function(*args, **kwargs):
            print("executed before {} (original function).".format(original_function.__name__))
            return original_function(*args, **kwargs)
        return wrapper_function
    
    
    
    
    
    
    
    
    
    
    
    def s(self, x:np.array, y:np.array):
        
        if not isinstance(x, np.array):
            x = np.array(x, dtype=float64)
        
        elif:
            
            
            
            
            
            
            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 43)
        
        
        
        
        
        
        
        
        else:
            raise ValueError(
                "x must be of type np.array"
            )
        
        
        
        
        if x_train.shape[0] = x_test.shape[0]:
            print(f"\n{x_test.shape[0]} obs in test set less than {x_train.shape[0]} obs in train set.")

        
        

         

        
        
        
        
        n_features = x.shape[1]
        x_train, x_test = x_train.values.reshape(-1, n_features), x_test.values.reshape(-1, n_features)
    
        return x_train, x_test, y_train, y_test
    
    
    
    
    
    def feature_scaling(x):
        
        n_samples, n_features = x.shape
        x_train, x_test = self.train_test_split()
        
        scaler = StandardScaler()
        
        x_train = scaler.fit_transform(x_train.values.reshape(-1,n_features))
        x_test = scaler.transform(x_test.values.reshape(-1,n_features))
        
        return x_train, x_test
    
    
    
    def dimensionality_reduction(self, var_retained:float = .95, np:bool = False):
        
        x_train, x_test = feature_scaling()
        pca = PCA(var_retained)
        
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)
        
        if len(x_train.shape[1]) == len(n_features) and isinstance(x_train, np.float64):
            raise ValueError(
                "%d n_features in X_train_pca is not less than %d n_features in X_train." 
                    (len(X_train_pca.shape[1]), len(n_features))
            )
                                                                    
        explained_variance = pca.explained_variance_ratio_

        return X_train, X_test
    
    
    
    
    
    
    def feature_selection(switch:bool):
        if switch:
            x_transformed = None
            def variance_threshold(x_train):
                if isinstance(x, pd.DataFrame):
                    x = x.loc[:, x.columns!="Date"]
                    x.select_dtypes(include = np.number)
        
                    scaler = MinMaxScaler(feature_range = (0,1))
                    x_scaled = scaler.fit_transform(x)

                    selector = VarianceThreshold(.03)
                    x_transformed = selector.fit_transform(x_scaled)
                    x_transformed = pd.DataFrame(x_transformed)
    
                    print(f"Original n_features: {x.shape[1]}.")
                    print("Transformed n_features: {}.".format(x_transformed.shape[1]))
    
                    dropped = [col for col in x.columns if col not in x.columns[selector.get_support()]]
                    dropped_list = [features for features in dropped]
                    x_transformed.drop(dropped, axis = 1, inplace = True)
        
                return x_transformed
        
        else:
            x_transformed = None
            def selectKBest(x, n=3):
                if isinstance(x, pd.DataFrame):
                    x = x.loc[:, x.columns!="Date"]
                    x.select_dtypes(include = np.number)

                    lab_enc = LabelEncoder()
                    y_enc = lab_enc.fit_transform(y)

                    selector = SelectKBest(score_func = mutual_info_classif, k = n)
                    selector.fit(x,y_enc)

                    a = [col for col in x.columns if col in x.columns[selector.get_support(True)]]
                    x_transformed = x.loc[:,x.columns.isin(a)]
        
                    print("Original {}.".format(x.shape))
                    print("Transformed {}.".format(x_transformed.shape))
    
                    #returns index where true
                    z = {x.columns.get_loc(c): c for index,c in enumerate(x.columns[selector.get_support(True)])}
    
                return x_transformed
           
        return variance_threshold() if switch is not None else selectKBest()
    
    
    
    
    


    def fit(self, x_train, y_train):
        
        x_train, x_test = split()
        if x_train:
        
            n_samples, n_features = X.shape
        
            # if gamma is not specified in init, it is specified as
            if not self.gamma:
                self.gamma = 1/(n_features*X.var())
        
            if X_train.shape[1] != n_features:
                raise ValueError("{} != {}".format(X_train.shape[1], self.n_features))
            elif X_train.shape[1] < 2:
                raise ValueError("cannot fit model with {} features.".format(X_train.shape[1]))
            
        
            # Checks X and y for consistent length, enforces X to be 2D and y 1D. 
            # By default, X is checked to be non-empty and containing only finite values. 
            # Standard input checks are also applied to y, such as checking that y does not have np.nan or np.inf targets.
            X, y = check_X_y(X, 
                             y, 
                             force_all_finite=False) # accepts np.nan in X
        
        
            # By default, the input is checked to be a non-empty 2D array containing only finite values.
            X = check_array(X, ensure_2d=True, ensure_min_samples=1, ensure_min_features=1)
            
            
            
            
            self.est = self._estimator(self.estimator)
            self.est.fit(X_train, y_train)
            
            
            
            
            
            
            
            
            if self.n:
                y_train_pca_ = np.where(y_train_pca <= 0, -1,1)
                n_samples, n_features = train.shape
                
                self.w = np.zeros(n_features) #each feature has to have some weight, in linear regression each beta parameter is a feature that has some weight
                self.b = 0

                if isinstance(self.C, type(None)):
                    raise ValueError(
                        "Regularisation parameter %s not defined in the constructor method." (str(self.C))
                    )
                else: 
                    lambda_param = 1 / self.C
            
            
            
                self.n_iters = int(self.n_iters)
                if self.n_iters <= 0:
                    raise ValueError(
                        f"{self.n_iters} is less than zero."
                    )
        
        
        
            #gradient descent
            for _ in range(1, self.n_iters):
                for index, x_i in enumerate(X.train_pca):
                    condition = y_train_pca_[index] * (np.dot(x_i, self.w) - self.b) >= 1
                    if condition:
                        # update params if condition true
                        # parameter = parameter - (self.learning rate * gradient (derivative of cost function))
                        self.w = self.w - (self.learn_rate * (2 * (1/self.C) * self.w))
                    else:
                        self.w = self.w - (self.learn_rate * ((2 * (1/self.C) * self.w) - np.dot(x_i, y_train_pca_[index])))
                        self.b = self.b - (self.learn_rate * y_train_pca_[index])
                        
        return something if self.n is not None else


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


for i in x.select_dtypes(include = np.number):
    if np.any(np.isnan(x[i])) == True:
        print("ok")


# In[ ]:


def label_encoding(self, y:np.ndarray):
        l = LabelEncoder()
        y = l.fit_transform(y)
        return y
    
if np:
            # calculate cov matrix
            cov_matrix = np.cov(X_train.T)
            eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
            
            
            # calculating explained variance on each component
            var_expl = [i/(sum(eigen_values))*100 for i in eigen_values]
            
            # identifying components that explain at least 95% variance
            cum_var_expl = np.cumsum(var_expl)


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
        
    def main():
        f = svm()
        
        f.load_data()
        f.split()
        f.feature_scaling()
        f.dimensionality_reduction()
        f.variance_threshold()
        f.selectKBest()
        
        f.fit()
        f.predict()
        
    if __name__ = "__main__":
        main()


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


@decorator_function
    def missing_vals(x, threshold = 0.6):
        threshold = float(threshold)
        for i in x.columns:
            if isinstance(x,pd.DataFrame):
                if float((x[i].isnull().sum()/x[i].shape[0])*100) > threshold:
                    x_transformed = x.drop(labels = i, axis = 1)
        return x_transformed
    

