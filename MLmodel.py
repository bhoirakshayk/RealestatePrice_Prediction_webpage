import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df =pd.read_csv("Real_estates.csv")
x = df.iloc[:, :-2]
y = df.iloc[:, -2]

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.3, random_state=1)

def mymodel(model):
    model.fit(xtrain, ytrain)
    return model

def makeprediction():
    linreg = LinearRegression()
    model = mymodel(linreg)
    return model
    
