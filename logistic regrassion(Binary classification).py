# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 23:49:45 2020

@author: Supriyo
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("insurance_data.csv")

plt.scatter(df.age,df.bought_insurance,marker="+",color="red")
plt.xlabel("Age",fontsize=10)
plt.ylabel("Insurance bought or not",fontsize=10)

x_train,x_test,y_train,y_test=train_test_split(df[['age']],df.bought_insurance,train_size=0.9)

model=LogisticRegression()
model.fit(x_train,y_train)

model.predict(x_test)

