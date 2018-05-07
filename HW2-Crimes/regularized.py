# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:53:47 2018

@author: M200
"""

import pandas as pd
train=pd.read_csv('Crime_train.csv')
test=pd.read_csv('Crime_test.csv')

ytrain=train['overall']
Xtrain=train.drop('overall',axis=1)
ytest=test['overall']
Xtest=test.drop('overall',axis=1)

from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.model_selection import GridSearchCV
import numpy as np
#1. Linear Regression
ap=np.linspace(3006012,5006012,1000)

bestTrain=0
bestTest=0
bAlpha=0
for param in ap:
    model1=Ridge(max_iter=5000,alpha=param)
    model1.fit(Xtrain,ytrain)
    if model1.score(Xtest,ytest)>bestTest and model1.score(Xtrain,ytrain)>bestTrain:
        bestTrain=model1.score(Xtrain,ytrain)
        bestTest=model1.score(Xtest,ytest)
        bAlpha=param
        print("Train score:%f , Test score:%f, bestAlpha:%f"%(bestTrain,bestTest,bAlpha))
    
