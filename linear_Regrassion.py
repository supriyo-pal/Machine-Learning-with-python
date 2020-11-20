# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:16:30 2020

@author: Supriyo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lin 

#############
#reding the exel file
df=pd.read_excel("home_prices.xlsx")

plt.xlabel('Area in sq ft')
plt.ylabel('Price in US $')
plt.scatter(df.area,df.price,color='red')


reg=lin.LinearRegression()
reg.fit(df[['area']],df.price)


reg.predict(3500)
reg.coef_
reg.intercept_

reg.predict([[3000]])
d=pd.read_excel('areas.xlsx')
p=reg.predict(d)

d['prices']=p
d.to_excel('prediction.xlsx',index=False)

plt.xlabel('Area in sq ft',fontsize=20)
plt.ylabel('Price in US $',fontsize=20)
plt.scatter(df.area,df.price,color='red',marker='+')
plt.plot(df.area,reg.predict(df[['area']]),color='Blue')

