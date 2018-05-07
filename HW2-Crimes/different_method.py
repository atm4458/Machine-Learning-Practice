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

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,f_regression
#1. Linear Regression
model1=LinearRegression()
model1.fit(Xtrain,ytrain)

#2. StandardScaler + Linear Regression
sst=StandardScaler()
Xtrain2=sst.fit_transform(Xtrain)
Xtest2=sst.transform(Xtest)
model2=LinearRegression()
model2.fit(Xtrain2,ytrain)


#3. PCA + Linear regression()
tPCA=PCA(n_components=1)
Xtrain3=tPCA.fit_transform(Xtrain)
Xtest3=tPCA.transform(Xtest)
model3=LinearRegression()
model3.fit(Xtrain3,ytrain)


#4. PCA + Linear regression()
fsk=SelectKBest(score_func=f_regression,k=1)
fsk.fit(Xtrain,ytrain)
Xtrain4=fsk.transform(Xtrain)
Xtest4=fsk.transform(Xtest)
model4=LinearRegression()
model4.fit(Xtrain4,ytrain)


#. Select by coef_ + Linear regression()
Xtrain5=Xtrain.iloc[:,0].values.reshape(-1,1)
Xtest5=Xtest.iloc[:,0].values.reshape(-1,1)
model5=LinearRegression()
model5.fit(Xtrain5,ytrain)



print('Score1:'+ str(model1.score(Xtest,ytest)*100) + '\nScore2:' +  str(model2.score(Xtest2,ytest)*100)
               + '\nScore3:' +  str(model3.score(Xtest3,ytest)*100)+ '\nScore4:' +  str(model4.score(Xtest4,ytest)*100)
               + '\nScore5:' +  str(model4.score(Xtest5,ytest)*100))

print(model1.coef_)
print(model2.coef_)
print(model3.coef_)
print(model4.coef_)