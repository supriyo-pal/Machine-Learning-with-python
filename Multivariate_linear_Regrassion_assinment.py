# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 20:31:48 2020

@author: Supriyo
"""

import pandas as pd
import numpy as np
import sklearn.linear_model as lin
import word2number.w2n as w
import math

df=pd.read_csv('hiring.csv')
df.experience=df.experience.fillna("Zero")

df.experience=df.experience.apply(w.word_to_num)

median_test_score=math.floor(df['test_score(out of 10)'].mean())
df['test_score(out of 10)'].fillna(median_test_score,inplace=True)

reg=lin.LinearRegression()
X=df[['experience','test_score(out of 10)','interview_score(out of 10)']]
Y=df[['salary($)']]
reg.fit(X,Y)


reg.predict([[2,9,6]])
reg.predict([[12,10,10]])

