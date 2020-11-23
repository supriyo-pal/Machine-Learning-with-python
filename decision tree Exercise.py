# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:18:57 2020

@author: Supriyo
"""

import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

df=pd.read_csv("titanic.csv")
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)


inputs=df.drop('Survived',axis='columns')
target=df.Survived

mean_of_age=(inputs.Age).mean()

inputs.Age.fillna(math.floor(mean_of_age),inplace=True)

inputs.Sex=inputs.Sex.map({'male':1,'female':2})
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)

model=tree.DecisionTreeClassifier()
model.fit(X_train,y_train)

model.score(X_test,y_test)
plt.scatter(inputs.Sex,inputs.Age)
plt.xlabel('SEX',fontsize=10)
plt.ylabel('AGE',fontsize=10)
