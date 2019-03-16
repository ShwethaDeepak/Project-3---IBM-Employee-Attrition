#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 19:37:21 2018

@author: swetu
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df_train = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
df_train.columns
label = df_train.iloc[:,1]
label = pd.get_dummies(label,drop_first=True)

df_train = df_train.drop(columns = 'Attrition')

df_train = pd.get_dummies(df_train,columns = ['BusinessTravel','OverTime','Department','EducationField','Gender','JobRole','MaritalStatus','Over18'])
df_train.dtypes

X_train,X_test,y_train,y_test = train_test_split(df_train,label,test_size = 0.25,random_state = 0)
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
train_accuracy = 'Train Accuracy:',regressor.score(X_train,y_train)
test_accuracy = 'Test Accuracy:',regressor.score(X_test,y_test)
print(train_accuracy)
print(test_accuracy)

 ###*************using decision tree and bagging classifier
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth = 3 ,min_samples_leaf = 0.16,random_state = 1)
bc = BaggingClassifier(base_estimator = dt, n_estimators = 300,oob_score = True,random_state=1)
bc.fit(X_train,y_train)
dt.fit(X_train,y_train)
y_pred1 = bc.predict(X_test)
y_pred2 = dt.predict(X_test)
print('Decision Classifier Accuracy:', accuracy_score(y_test,y_pred2))
print("Bagging Classifier Accuracy:",accuracy_score(y_test,y_pred1))


###************using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 25, random_state = 2)
rf.fit(X_train,y_train)
y_pred3 = rf.predict(X_test)

print('RandomForestClassifier Accuracy:', accuracy_score(y_test,y_pred3))

###  
import matplotlib.pyplot as plt
importances = pd.Series(data = rf.feature_importances_,index = X_train.columns)

importances_sorted = importances.sort_values()
importances_sorted.plot(kind = 'barh',color = 'red')
plt.title("features importance")
plt.show()

print(importances_sorted)


# Building model with features Selection

features = pd.DataFrame(df_train,columns= ['Age','MonthlyIncome','TotalWorkingYears','EmployeeNumber','DailyRate','DistanceFromHome','MonthlyRate','HourlyRate','OverTime_No','YearsAtCompany','StockOptionLevel','PercentSalaryHike','EnvironmentSatisfaction',])
label1 = label
features.dtypes
features.isnull().sum()

X_train,X_test,y_train,y_test = train_test_split(features,label1,test_size = 0.25,random_state = 0)
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
train_accuracy = 'Train Accuracy:',regressor.score(X_train,y_train)
test_accuracy = 'Test Accuracy:',regressor.score(X_test,y_test)
print(train_accuracy)
print(test_accuracy)

 ###*************using decision tree and bagging classifier
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth = 3 ,min_samples_leaf = 0.16,random_state = 1)
bc = BaggingClassifier(base_estimator = dt, n_estimators = 300,oob_score = True,random_state=1)
bc.fit(X_train,y_train)
dt.fit(X_train,y_train)
y_pred1 = bc.predict(X_test)
y_pred2 = dt.predict(X_test)
print('Decision Classifier Accuracy:', accuracy_score(y_test,y_pred2))
print("Bagging Classifier Accuracy:",accuracy_score(y_test,y_pred1))


###************using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 25, random_state = 2)
rf.fit(X_train,y_train)
y_pred3 = rf.predict(X_test)

print('RandomForestClassifier Accuracy:', accuracy_score(y_test,y_pred3))













































