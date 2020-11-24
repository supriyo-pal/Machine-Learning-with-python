# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:16:21 2020

@author: Supriyo
"""

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

digits=load_digits()

df=pd.DataFrame(digits.data,digits.target)

df.to_csv('digits.csv',index=False)

df['target']=digits.target

X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis='columns'), df.target, test_size=0.52)

rbf_model=SVC(kernel='rbf')
rbf_model.fit(X_train,y_train)
rbf_model.score(X_test,y_test)



linear_model=SVC(kernel='linear')
linear_model.fit(X_train,y_train)
linear_model.score(X_test,y_test)