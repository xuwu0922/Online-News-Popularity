# Online news popularity
# created by: Xu Wu

import os
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from time import time

from sklearn.metrics import roc_auc_score




# read .csv from provided dataset
csv_filename="OnlineNewsPopularity.csv"
# df=pd.read_csv(csv_filename,index_col=0)
df=pd.read_csv(csv_filename)
# handle goal attrubte to binary
popular = df.shares >= 1400	
unpopular = df.shares < 1400
df.loc[popular,'shares'] = 1
df.loc[unpopular,'shares'] = 0


features=list(df.columns[2:60])

# split dataset to 60% training and 40% testing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(df[features], df['shares'], test_size=0.4, random_state=0)

# print X_train.shape, y_train.shape


# Decision Tree accuracy and time elapsed caculation
t0=time()
print "DecisionTree"
dt = DecisionTreeClassifier(min_samples_split=20,random_state=99)
# dt = DecisionTreeClassifier(min_samples_split=20,max_depth=5,random_state=99)
clf_dt=dt.fit(X_train,y_train)
print "Acurracy: ", clf_dt.score(X_test,y_test)
t1=time()
print "time elapsed: ", t1-t0

# cross validation for DT
tt0=time()
print "cross result========"
scores = cross_validation.cross_val_score(dt, df[features], df['shares'], cv=5)
print scores
print scores.mean()
tt1=time()
print "time elapsed: ", tt1-tt0
print "\n"


# Random Forest accuracy and time elapsed caculation
t2=time()
print "RandomForest"
rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
clf_rf = rf.fit(X_train,y_train)
print "Acurracy: ", clf_rf.score(X_test,y_test)
t3=time()
print "time elapsed: ", t3-t2

#cross validation for RF
tt2=time()
print "cross result========"
scores = cross_validation.cross_val_score(rf, df[features], df['shares'], cv=5)
print scores
print scores.mean()
tt3=time()
print "time elapsed: ", tt3-tt2
print "\n"

# Naive Bayes accuracy and time elapsed caculation
t4=time()
print "NaiveBayes"
nb = BernoulliNB()
clf_nb=nb.fit(X_train,y_train)
print "Acurracy: ", clf_nb.score(X_test,y_test)
t5=time()
print "time elapsed: ", t5-t4

# cross-validation for NB
tt4=time()
print "cross result========"
scores = cross_validation.cross_val_score(nb, df[features], df['shares'], cv=5)
print scores
print scores.mean()
tt5=time()
print "time elapsed: ", tt5-tt4
print "\n"



#KNN accuracy and time elapsed caculation
t6=time()
print "KNN"
# knn = KNeighborsClassifier(n_neighbors=3)
knn = KNeighborsClassifier()
clf_knn=knn.fit(X_train, y_train)
print "Acurracy: ", clf_knn.score(X_test,y_test) 
t7=time()
print "time elapsed: ", t7-t6

# cross validation for KNN
tt6=time()
print "cross result========"
scores = cross_validation.cross_val_score(knn, df[features], df['shares'], cv=5)
print scores
print scores.mean()
tt7=time()
print "time elapsed: ", tt7-tt6
print "\n"
