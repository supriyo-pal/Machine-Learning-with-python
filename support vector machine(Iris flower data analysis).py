# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:39:08 2020

@author: Supriyo
"""

import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris=load_iris()

df=pd.DataFrame(iris.data,columns=iris.feature_names)

df['target']=iris.target

df[df.target==1].head()
df[df.target==2].head()

df['flower_name']=df.target.apply(lambda x:iris.target_names[x])

df.to_csv("new_flower_data.csv",index=False)

df_fl1=df[df.target==0]
df_fl2=df[df.target==1]
df_fl3=df[df.target==2]

plt.scatter(df_fl1['sepal length (cm)'],df_fl1['sepal width (cm)'],marker="+")
plt.scatter(df_fl2['sepal length (cm)'],df_fl2['sepal width (cm)'],marker=".")
plt.scatter(df_fl3['sepal length (cm)'],df_fl3['sepal width (cm)'],marker="*")
plt.xlabel('sepal length (cm)',fontsize=15)
plt.ylabel('sepal width (cm)',fontsize=15)

plt.scatter(df_fl1['petal length (cm)'],df_fl1['petal width (cm)'],marker="+")
plt.scatter(df_fl2['petal length (cm)'],df_fl2['petal width (cm)'],marker=".")
plt.scatter(df_fl3['petal length (cm)'],df_fl3['petal width (cm)'],marker="*")
plt.xlabel('petal length (cm)',fontsize=15)
plt.ylabel('petal width (cm)',fontsize=15)


X=df.drop(['target','flower_name'],axis='columns')
Y=df.target

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)


model=SVC(C=10)

model.fit(x_train,y_train)

model.score(x_test,y_test)

























