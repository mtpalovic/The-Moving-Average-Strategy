#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# initial learn rate decay


# In[3]:


import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')

from datetime import datetime

import seaborn as sns

import pandas as pd
pd.set_option("display.max_columns", None)

from sklearn.preprocessing import (StandardScaler,
                                   Normalizer,
                                   LabelEncoder
                                  )

from sklearn.decomposition import PCA

from sklearn.feature_selection import (VarianceThreshold,
                                       SelectKBest, mutual_info_regression, mutual_info_classif,
                                       RFE,
                                       SelectFromModel
                                      )

from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, StackingClassifier)

from sklearn.utils import shuffle
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import SVR
from sklearn.model_selection import (train_test_split,
                                     GridSearchCV
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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


class svm(BaseEstimator, TransformerMixin):
    
    test_size = 0.2
    m_vals = 0.05
    
    def __init__(self,
                 estimator = "SVC"
                 
                 kernel:Optional[str] = "rbf",
                 C:int = 1e3, # to represent 1000 (1 times 10 to the third power)
                 
                 learn_rate:float = 0.001,
                 tol:float = 0.05,
                 batch_size:int = 1,
                 n_epochs:int = 1000,
                 decay:float = 1, 
                 random_number:Optional[float] = None,
                 visualisation = True,
                 
                 n = False # works only for linearly separable data and is like a switch
                ):
        
        self.estimator = estimator
        self.kernel = self._kernel_type(kernel, **kwargs)
        
        self.C = C
        self.learn_rate = learn_rate
        self.tol = tol
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.decay = decay
        self.random_number = random_number if random_number is not None else np.random.randint(0,100,size=1)
        
        self.n = n
        
        # initialise model params so I have to come up with them later
        self.w = False
        self.b = False
        
        
        self.visualisation = visualisation
        self.colors = {1:"r", -1:"b"}
        if self.visualisation:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
            
            
    def decorator_function(original_function):
        def wrapper_function(*args, **kwargs):
            print("executed before {} (original function).".format(original_function.__name__))
            return original_function(*args, **kwargs)
        return wrapper_function
            
    
             
        
    @staticmethod
    # static method has no self parameter and knows nothing about a class
    def missing_values(X:np.ndarray, threshold:int = None, verbose = False):
        """Step 0: Data pre-processing. Missing values."""
        if threshold:
            m_vals_pct = ([len(isnull(X[:,i])) for i in range (X.shape[1])]/X.shape[0])
            to_del = [i for i,v in enumerate(m_vals_pct) if v >= threshold]
            if verbose:
                print("del {}.".format(to_del))
            new_cols = [i for i in range(X.shape[1]) if i not in to_del]
            X = X[:,new_cols]
        return X
    
    
    
    
    
    def _estimator(self, n:str)
        est = {
            "linear": LinearRegression(),
            "svc": SVC(kernel=self.kernel, C = self.C)
        }
    
        return est[n]
    
    

        
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
           
        
        
        
        
        
    def loss_function(self):
        """The below returns the loss function for the stochastic gradient descent."""
        # The input to this function is the predicted output and the actual output.
        pass
        
        
        
        
        
        
        
    def label_encoding(self, y:np.ndarray):
        l = LabelEncoder()
        y = l.fit_transform(y)
        return y
        
        
    def train_test_split(self, X_pca:np.ndarray, y:np.ndarray):    
        """Step 3: Data pre-processing. Split the dataset into train and test set."""
        
        X = X.values.reshape(-1,1)
        y = y.values.reshape(-1,1)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = svm.test_size, random_state = self.random_number, shuffle = True, stratify = y)
        
        X_train.shape, y_train.shape
        X_test.shape, y_test.shape
        
        # Convert to float64
        X_train, X_test = np.array(X_train,dtype = np.float64), np.array(X_test,dtype = np.float64) 
        
        # reshape
        n_samples, n_features = X.shape
        X_train, X_test = X_train.reshape(-1, n_features), X_test.reshape(-1, n_features) 
        
        
        return X_train, X_test, y_train, y_test
    
    
    
    
    
    @decorator_function
    def feature_scaling(self):
        """Step 4: Data pre-processing. Feature scaling."""
        
        X_train, X_test = train_test_split()
        
        # Standardization is the process of scaling data so that they have a mean value of 0 and a standard deviation of 1
        scaler = StandardScaler()
        
        # fit_transform on X_train but transform on X_test
        X_train = scaler.fit_transform(X_train.values.reshape(-1,1))
        X_test = scaler.transform(X_test.values.reshape(-1,1))
        
        
    
        
    def feature_extraction(self, var_retained:float = .95, np:bool = False):
        """Step 5: Data pre-processing. Principal Component Analysis."""
        pca = PCA(var_retained) # choose n_components such that 95% variance remains explained
        
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        if len(X_train_pca.shape[1]) = len(X_train.shape[1]) and isinstance(X_train_pca, np.float64):
            raise ValueError(
                "%d n_features in X_train_pca is not less than %d n_features in X_train." 
                    (len(X_train_pca.shape[1]), len(X_train.shape[1]))
            )
                                                                    
        explained_variance = pca.explained_variance_ratio_
        
        if np:
            # calculate cov matrix
            cov_matrix = np.cov(X_train.T)
            eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
            
            
            # calculating explained variance on each component
            var_expl = [i/(sum(eigen_values))*100 for i in eigen_values]
            
            # identifying components that explain at least 95% variance
            cum_var_expl = np.cumsum(var_expl)
            
    
        return X_train, X_test
    
    
    
    
    
    def fit(self, X_train:Union[np.ndarray, pd.DataFrame], y_train:Union[np.ndarray, pd.DataFrame], 
            sample_weight = 0.1):
        """Step 6: Model-fitting."""
        
        if isinstance(X, pd.DataFrame):
        
            X_train, y_train = train_test_split()
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
            n_samples, n_features = X.shape
            self.w = np.zeros(n_features)
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
                    "{} must be greater than zero".format(self.n_iters)
                )
        
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
        
        
        
        
        
        
        return self if self.n is False else 
        
        
        

        
        
        
        
        
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
        gspc = pd.read_csv("^GSPC.csv")
        
        
        
        
        m = svm()
        
        m.train_test_split()
        m.feature_scaling()
        m.feature_extraction()
        
        m.fit()
        m.predict()
        
        
        
        
   


# In[ ]:





# In[ ]:





# In[4]:


path = "C:/Users/mpalovic/Desktop"
file_name = "ta.gspc" + ".csv"

custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
df = pd.read_csv(filepath_or_buffer = "{}/{}".format(path, file_name), 
                 sep = ",", 
                 index_col=None,                 
                 parse_dates=["Date"], 
                 date_parser=custom_date_parser)
df = pd.DataFrame(df)

x = df.loc[:, df.columns!="Close"]
y = df.iloc[:,1]


# In[5]:


df


# In[1]:


df


# In[ ]:





# In[ ]:





# In[ ]:





# In[65]:





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





# In[63]:





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


date = datetime.date(2016,7,11)


# In[ ]:


@staticmethod
def is_workday(d):
    if d.weekday() ==5 or d.weekday() ==6:
        return False
    return True


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


is_workday(date)


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


#original_function = decorator_function(original_function)
#original_function()

# if np.all(np.abs(diff) <= tol):
# break  

