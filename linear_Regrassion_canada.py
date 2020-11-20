# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:12:32 2020

@author: Supriyo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lin

df=pd.read_csv('canada_data.csv') 

plt.scatter(df.year,df.income,color='red')
plt.xlabel('Years',fontsize=15)
plt.ylabel('Income Per Capita in US $',fontsize=15)


reg=lin.LinearRegression()
reg.fit(df[['year']],df.income)

reg.predict([[2020]])
plt.plot(df.year,reg.predict(df[['year']]),color='Blue')
