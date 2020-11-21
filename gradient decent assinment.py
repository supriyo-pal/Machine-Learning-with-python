# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 12:40:00 2020

@author: Supriyo
"""

import  numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
import math

def predict_using_sklearn():
    df=pd.read_csv("test_scores.csv")
    reg=LinearRegression()
    reg.fit(df[['math']],df.cs)
    return reg.coef_, reg.intercept_

def gradient_decent(x,y):
   m_current=0
   b_current=0
   iteration=10
   
   n=len(x)
   learning_rate=0.0002
   
   cost_previous=0
   
   for i in range(iteration):
       y_predicted=m_current*x+b_current
       
       cost=(1/n)*sum([val**2 for val in y-y_predicted])
       md= -(2/n)*sum(x*(y-y_predicted))
       bd= -(2/n)*sum(y-y_predicted)
       
       m_current=m_current-learning_rate*md
       b_current=b_current-learning_rate*bd
       
       if math.isclose(cost,cost_previous,rel_tol=1e-20):
           break
       cost_previous=cost
       print("m{} , b{}, cost{} , iteration{}, ".format(m_current,b_current,cost,i))
   return m_current,b_current

if __name__=="__main__":
    df=pd.read_csv("test_scores.csv")
    x=np.array(df.math)
    y=np.array(df.cs)
    
    m,b=gradient_decent(x,y)
    print("Using gradscent function : coef{}  intercept{}".format(m,b))
    
    m_sklearn,b_sklearn=predict_using_sklearn()
    print("Using Sklearn: coef{} Intercept{}".format(m_sklearn,b_sklearn))
    
    reg=LinearRegression()
    reg.fit(df[['math']],df.cs)
    plt.scatter(df.math,df.cs,color='red')
    
    plt.xlabel('MATH',fontsize=15)
    plt.ylabel('CS',fontsize=15)
    plt.plot(df.math,reg.predict(df[['math']]),color='Blue')