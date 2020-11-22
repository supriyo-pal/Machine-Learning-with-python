# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:26:22 2020

@author: Supriyo
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn

digits=load_digits()

plt.gray()
# for i in range(5):
#     plt.matshow(digits.images[i])
    
x_train,x_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.2)

model=LogisticRegression()
model.fit(x_train,y_train)

model.score(x_test,y_test)

plt.matshow(digits.images[67])
digits.target[67]
model.predict([digits.data[67]])

model.predict(digits.data[0:5])

y_predicted=model.predict(x_test)

cm=confusion_matrix(y_test,y_predicted)

plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel("Prdicted")
plt.ylabel("Truth")