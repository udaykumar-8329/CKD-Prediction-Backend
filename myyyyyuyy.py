# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 13:34:29 2020

@author: anush
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams 
from sklearn.linear_model import LogisticRegression# Import logisticregression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split # used for splitting training and testing data
from sklearn.preprocessing import StandardScaler # used for feature scaling
from sklearn.tree import DecisionTreeClassifier# Import decision tree
from sklearn.naive_bayes import GaussianNB# Import naive_bayes
from sklearn.neighbors import KNeighborsClassifier# Import knn
from sklearn.ensemble import RandomForestClassifier# Import random forest
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
dataset1 = pd.read_csv('mycsvdata.csv')
k=dataset1.isna().sum()
print(k)

dataset1.drop('id',axis=1,inplace = True)     
#Data Mapping
dataset1['classification'] = dataset1['classification'].map({'ckd':1,'\tckd':1,'notckd':0,'\tnotckd':0})
dataset1['htn'] = dataset1['htn'].map({'yes':1,'no':0,'\tyes':1,'\tno':0})
dataset1['dm'] = dataset1['dm'].map({'yes':1,'no':0,'\tyes':1,'\tno':0})
dataset1['cad'] = dataset1['cad'].map({'yes':1,'\tno':0,'no':0,'\tyes':1})
dataset1['appet'] = dataset1['appet'].map({'good':1,'poor':0})
dataset1['ane'] = dataset1['ane'].map({'yes':1,'no':0,'\tyes':1,'\tno':0})
dataset1['pe'] = dataset1['pe'].map({'yes':1,'no':0,'\tyes':1,'\tno':0})
dataset1['ba'] = dataset1['ba'].map({'present':1,'notpresent':0})
dataset1['pcc'] = dataset1['pcc'].map({'present':1,'notpresent':0})
dataset1['pc'] = dataset1['pc'].map({'abnormal':1,'normal':0})
dataset1['rbc'] = dataset1['rbc'].map({'abnormal':1,'normal':0})
print(len(dataset1.columns))

dataset1.fillna(round(dataset1.mean()),inplace=True)
C = dataset1.iloc[:, :-1].values # attributes to determine dependent variable / Class
D = dataset1.iloc[:, -1].values# dependent variable / Class
X_train1, X_test1, y_train1, y_test1 = train_test_split(C,D, test_size = 0.20, random_state = 43)

# Feature Scaling
sc_X1 = StandardScaler()
X_train1 = sc_X1.fit_transform(X_train1)
X_test1 = sc_X1.transform(X_test1)
#List of model predicted values
Models1 = list()
Models1

 
#Build a Logistic regression
LR1=LogisticRegression()

# Train the model
LR1.fit(X_train1, y_train1)

#Predict outcome on testing dataset, assign prediction outcome to y_pred

y_pred_LR1 = LR1.predict(X_test1) 

Models1.append(y_pred_LR1)
print()
print()
print("Logistic Regression")
print("-"*30)
print("Accuracy percentage:"+"{:.2f}".format(accuracy_score(y_test1,y_pred_LR1)*100))#accuracy score
print("Confusion Matrix:")
l=confusion_matrix(y_test1,y_pred_LR1)
print(l)
tp=l[0][0]
fn=l[0][1]
fp=l[1][0]
tn=l[1][1]
err=(fp+fn)/(tp+fn+fp+tn)
print("Basic Evaluation Measures")
print("Error Rate:",err)
print("Accuracy:",1-err)
print("Sensitivity (Recall or True positive rate):",tp/(tp+fn))
print("Specificity (True negative rate):",tn/(tn+fp))
print("Precision (Positive predictive value)",tp/tp+fp)
print("False positive rate:",fp/tn+fp)
print()
print()
# Build a decision tree model
dtree=DecisionTreeClassifier()

# Train the model
dtree.fit(X_train1, y_train1)
  
#Predicting ourtcome on dataset

y_pred_dtree = dtree.predict(X_test1)

Models1.append(y_pred_dtree)
print("Decision Tree")
print("-"*30)
print("Accuracy percentage:"+"{:.2f}".format(accuracy_score(y_test1,y_pred_dtree)*100))#accuracy score
print("Confusion Matrix:")
l=confusion_matrix(y_test1,y_pred_dtree)
print(l)
tp=l[0][0]
fn=l[0][1]
fp=l[1][0]
tn=l[1][1]
err=(fp+fn)/(tp+fn+fp+tn)
print("Basic Evaluation Measures")
print("Error Rate:",err)
print("Accuracy:",1-err)
print("Sensitivity (Recall or True positive rate):",tp/(tp+fn))
print("Specificity (True negative rate):",tn/(tn+fp))
print("Precision (Positive predictive value)",tp/tp+fp)
print("False positive rate:",fp/tn+fp)
print()
print()
#Build a Gaussian Naive Bayes model
GNB = GaussianNB()

#Train a model
GNB.fit(X_train1, y_train1) 

#Predicting outcome on dataset
y_pred_NB = GNB.predict(X_test1)

Models1.append(y_pred_NB)
print("SVR")
print("-"*30)
print("Accuracy percentage:"+"{:.2f}".format(accuracy_score(y_test1,y_pred_NB)*100))#accuracy score
print("Confusion Matrix:")
l=confusion_matrix(y_test1,y_pred_NB)
print(l)
tp=l[0][0]
fn=l[0][1]
fp=l[1][0]
tn=l[1][1]
err=(fp+fn)/(tp+fn+fp+tn)
print("Basic Evaluation Measures")
print("Error Rate:",err)
print("Accuracy:",1-err)
print("Sensitivity (Recall or True positive rate):",tp/(tp+fn))
print("Specificity (True negative rate):",tn/(tn+fp))
print("Precision (Positive predictive value)",tp/tp+fp)
print("False positive rate:",fp/tn+fp)
print()
print()
#Build a KNeighborsClassifier
knn=KNeighborsClassifier()

#Train a Model
knn.fit(X_train1, y_train1)

#Predicting outcome on dataset
y_pred_KN = knn.predict(X_test1)

Models1.append(y_pred_KN)
print("KNN")
print("-"*30)
print("Accuracy percentage:"+"{:.2f}".format(accuracy_score(y_test1,y_pred_KN)*100))#accuracy score
print("Confusion Matrix:")
l=confusion_matrix(y_test1,y_pred_KN)
print(l)
tp=l[0][0]
fn=l[0][1]
fp=l[1][0]
tn=l[1][1]
err=(fp+fn)/(tp+fn+fp+tn)
print("Basic Evaluation Measures")
print("Error Rate:",err)
print("Accuracy:",1-err)
print("Sensitivity (Recall or True positive rate):",tp/(tp+fn))
print("Specificity (True negative rate):",tn/(tn+fp))
print("Precision (Positive predictive value)",tp/tp+fp)
print("False positive rate:",fp/tn+fp)
print()
print()
# Build A Random Forest Model
RanFor=RandomForestClassifier()
# Train the model
RanFor.fit(X_train1, y_train1)
#Predict outcome on testing dataset, assign prediction outcome to y_pred

y_pred_RanFor = RanFor.predict(X_test1) 

Models1.append(y_pred_RanFor)
#accuracy score
print("Random Forest")
print("-"*30)
print("Accuracy percentage:"+"{:.2f}".format(accuracy_score(y_test1,y_pred_RanFor)*100))#accuracy score
print("Confusion Matrix:")
l=confusion_matrix(y_test1,y_pred_RanFor)
print(l)
tp=l[0][0]
fn=l[0][1]
fp=l[1][0]
tn=l[1][1]
err=(fp+fn)/(tp+fn+fp+tn)
print("Basic Evaluation Measures")
print("Error Rate:",err)
print("Accuracy:",1-err)
print("Sensitivity (Recall or True positive rate):",tp/(tp+fn))
print("Specificity (True negative rate):",tn/(tn+fp))
print("Precision (Positive predictive value)",tp/tp+fp)
print("False positive rate:",fp/tn+fp)
print()
print()
# Build A SVM Model
SVM=SVC()
# Train the model
SVM.fit(X_train1, y_train1)
#Predict outcome on testing dataset, assign prediction outcome to y_pred

y_pred_Svm = SVM.predict(X_test1) 

Models1.append(y_pred_Svm)
#accuracy score
print("SVM")
print("-"*30)
print("Accuracy percentage:"+"{:.2f}".format(accuracy_score(y_test1,y_pred_Svm)*100))#accuracy score
print("Confusion Matrix:")
l=confusion_matrix(y_test1,y_pred_Svm)
print(l)
tp=l[0][0]
fn=l[0][1]
fp=l[1][0]
tn=l[1][1]
err=(fp+fn)/(tp+fn+fp+tn)
print("Basic Evaluation Measures")
print("Error Rate:",err)
print("Accuracy:",1-err)
print("Sensitivity (Recall or True positive rate):",tp/(tp+fn))
print("Specificity (True negative rate):",tn/(tn+fp))
print("Precision (Positive predictive value)",tp/tp+fp)
print("False positive rate:",fp/tn+fp)
print()
print()
from sklearn.linear_model import SGDClassifier
SVM=SGDClassifier()
# Train the model
SVM.fit(X_train1, y_train1)
#Predict outcome on testing dataset, assign prediction outcome to y_pred

y_pred_Svm = SVM.predict(X_test1) 

Models1.append(y_pred_Svm)
#accuracy score
print("SGDClassifier")
print("-"*30)
print("Accuracy percentage:"+"{:.2f}".format(accuracy_score(y_test1,y_pred_Svm)*100))
l=confusion_matrix(y_test1,y_pred_Svm)
print(l)
tp=l[0][0]
fn=l[0][1]
fp=l[1][0]
tn=l[1][1]
err=(fp+fn)/(tp+fn+fp+tn)
print("Basic Evaluation Measures")
print("Error Rate:",err)
print("Accuracy:",1-err)
print("Sensitivity (Recall or True positive rate):",tp/(tp+fn))
print("Specificity (True negative rate):",tn/(tn+fp))
print("Precision (Positive predictive value)",tp/tp+fp)
print("False positive rate:",fp/tn+fp)
print()  








