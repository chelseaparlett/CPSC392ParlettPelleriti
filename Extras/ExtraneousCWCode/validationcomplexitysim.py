import warnings
warnings.filterwarnings('ignore')

# data imports
import pandas as pd
import numpy as np
from plotnine import *

# modeling imports
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV # Linear Regression Model
from sklearn.preprocessing import StandardScaler #Z-score variables
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error #model evaluation
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut

# pipeline imports
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer

%matplotlib inline


# example of how to time code
start = time.perf_counter()

for i in range(0,100000):
    10 + 10


stop = time.perf_counter()
print("it took", stop-start, "seconds to run this code.")

# read in data

p = pd.read_csv("https://raw.githubusercontent.com/cmparlettpelleriti/CPSC392ParlettPelleriti/master/Data/PopDivas_data.csv")
print(p.shape)
p.head()

# Define functions that can run our model validation simulation
def TTSSim(X,y, contin):
    start = time.perf_counter()

    ###

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

    z = make_column_transformer((StandardScaler(), contin),
                                remainder = "passthrough")

    lr = LinearRegression()

    pipe = Pipeline([("zscore", z),
                    ("linearregression", lr)])

    pipe.fit(X_train,y_train)            

    ###
    stop = time.perf_counter()
    
    return(stop-start)

def KFSim(X,y, contin):
    start = time.perf_counter()

    ###
    kf = KFold(5)

    z = make_column_transformer((StandardScaler(), contin),
                            remainder = "passthrough")

    lr = LinearRegression()

    pipe = Pipeline([("zscore", z),
                    ("linearregression", lr)])

    for train,test in kf.split(X):
        X_train = X.iloc[train]
        X_test = X.iloc[test]
        y_train = y[train]
        y_test = y[test]


        pipe.fit(X_train, y_train)

    ###
    stop = time.perf_counter()
    
    return(stop-start)
    
def LOOSim(X,y, contin):
    start = time.perf_counter()

    ###
    loo = LeaveOneOut()

    z = make_column_transformer((StandardScaler(), contin),
                            remainder = "passthrough")

    lr = LinearRegression()

    pipe = Pipeline([("zscore", z),
                    ("linearregression", lr)])

    for train,test in loo.split(X):
        X_train = X.iloc[train]
        X_test = X.iloc[test]
        y_train = y[train]
        y_test = y[test]
        
        pipe.fit(X_train, y_train)

    ###
    stop = time.perf_counter()
    
    return(stop - start)

# Train Test Split Sim
tts = [TTSSim(X,y, contin) for i in range(0,250)]

(ggplot(pd.DataFrame({"x": tts}), aes(x = "x")) +
 geom_histogram(fill = "white", color = "black") + theme_bw() +
 geom_vline(xintercept = np.mean(tts), color = "red", linetype = "dashed"))

# KFOLD Sim
kff = [KFSim(X,y, contin) for i in range(0,250)]

(ggplot(pd.DataFrame({"x": kff}), aes(x = "x")) +
 geom_histogram(fill = "white", color = "black") + theme_bw() +
 geom_vline(xintercept = np.mean(kff), color = "red", linetype = "dashed"))

# Leave One Out Sim
loo = [LOOSim(X,y,contin) for i in range(0,25)]

(ggplot(pd.DataFrame({"x": loo}), aes(x = "x")) +
 geom_histogram(fill = "white", color = "black") + theme_bw()+
 geom_vline(xintercept = np.mean(loo), color = "red", linetype = "dashed"))