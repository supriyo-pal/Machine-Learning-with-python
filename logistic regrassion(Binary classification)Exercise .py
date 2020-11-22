# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 21:00:10 2020

@author: Supriyo
"""
import pandas as pd
from matplotlib import pyplot as plt

df=pd.read_csv("hr_data.csv")
left_job=df[df.left==1] #left is the columns in the dataset

retained_job=df[df.left==0]

fg=df.groupby('left').mean()
#fg.to_csv('Left and retained data set.csv')
'''
**Satisfaction Level**: Satisfaction level seems to be relatively low (0.44) in employees leaving the firm vs the retained ones (0.66)
**Average Monthly Hours**: Average monthly hours are higher in employees leaving the firm (199 vs 207)
**Promotion Last 5 Years**: Employees who are given promotion are likely to be retained at firm
'''
#impact of salary employee mentioned

pd.crosstab(df.salary,df.left).plot(kind='bar')

#depatment wise employee retaintion rate
pd.crosstab(df.Department,df.left).plot(kind="bar")

'''

From the data analysis so far we can conclude that we will use following variables as independant variables in our model
**Satisfaction Level**
**Average Monthly Hours**
**Promotion Last 5 Years**
**Salary**

'''
subdf=df[['satisfaction_level','average_montly_hours','average_montly_hours','salary']]
'''
Salary has all text data. It needs to be converted to numbers and we will use dummy variable for that. 
Check my one hot encoding tutorial to understand purpose behind dummy variables.
'''

salary_dummies=pd.get_dummies(subdf.salary,prefix="salary")
df_with_dummies=pd.concat([subdf,salary_dummies],axis="columns")


df_with_dummies.drop('salary',axis="columns",inplace=True)