# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 22:51:34 2020

@author: Supriyo
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df=pd.read_csv("salaries.csv")

inputs=df.drop('salary_more_then_100k',axis='columns')
target=df['salary_more_then_100k']

le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()

inputs['company_n']=le_company.fit_transform(inputs['company'])
inputs['job_n']=le_company.fit_transform(inputs['job'])
inputs['degree_n']=le_company.fit_transform(inputs['degree'])

inputs_n=inputs.drop(['company','job','degree'],axis='columns')

model=tree.DecisionTreeClassifier()
model.fit(inputs_n,target)

model.score(inputs_n,target)

model.predict([[2,2,1]])
model.predict([[2,0,1]])
