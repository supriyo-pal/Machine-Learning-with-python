# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:32:20 2020

@author: Supriyo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lin
import math as m
df=pd.read_excel('multivariate1.xlsx')

median_bedrooms=m.floor(df.bedrooms.median())

df.bedrooms=df.bedrooms.fillna(median_bedrooms)

reg=lin.LinearRegression()
X=df[['myarea','bedrooms','age']]
Y=df.price
reg.fit(X,Y)

reg.predict([[3000,3,40]])

reg.predict([[2500,4,5]])
