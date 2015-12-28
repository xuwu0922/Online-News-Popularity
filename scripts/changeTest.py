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
import xlsxwriter



# load dataset to padas dataframe
csv_filename="OnlineNewsPopularity.csv"

df=pd.read_csv(csv_filename)

popular = df.shares >= 1400	
unpopular = df.shares < 1400

df.loc[popular,'shares'] = 1
df.loc[unpopular,'shares'] = 0

# split original dataset into 60% training and 40% testing
features=list(df.columns[2:60])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(df[features], df['shares'], test_size=0.4, random_state=0)

# open one ouput excel file and two worksheets
workbook = xlsxwriter.Workbook('changeTest_output.xlsx')

worksheet = workbook.add_worksheet()
worksheet.write("A1","scala_index")
worksheet.write("B1","DecisionTree")
worksheet.write("C1","KNN")
worksheet.write("D1","RandomForest")
worksheet.write("E1","NaiveBayes")

worksheet2 = workbook.add_worksheet()
worksheet2.write("A1","scala_index")
worksheet2.write("B1","DecisionTree")
worksheet2.write("C1","KNN")
worksheet2.write("D1","RandomForest")
worksheet2.write("E1","NaiveBayes")

# increasingly add size of testing set 5% of orginal, keep training size unchanged
for i in range(0,100,5):
	X_rest, X_test_part, y_rest, y_test_part= cross_validation.train_test_split(X_test, y_test, test_size=0.049+i/100.0, random_state=0)
	print "====================== loop: ", i 
	t0=time()
	print "DecisionTree"
	dt = DecisionTreeClassifier(min_samples_split=20,random_state=99)
# dt = DecisionTreeClassifier(min_samples_split=20,max_depth=5,random_state=99)
	clf_dt=dt.fit(X_train,y_train)
	score_dt=clf_dt.score(X_test_part,y_test_part)
	print "Acurracy: ", score_dt
	t1=time()
	dur_dt=t1-t0
	print "time elapsed: ", dur_dt	
	print "\n"

	t6=time()
	print "KNN"
# knn = KNeighborsClassifier(n_neighbors=3)
	knn = KNeighborsClassifier()
	clf_knn=knn.fit(X_train, y_train)
	score_knn=clf_knn.score(X_test_part,y_test_part) 
	print "Acurracy: ", score_knn 
	t7=time()
	dur_knn=t7-t6
	print "time elapsed: ", dur_knn
	print "\n"

	t2=time()
	print "RandomForest"
	rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
	clf_rf = rf.fit(X_train,y_train)
	score_rf=clf_rf.score(X_test_part,y_test_part)
	print "Acurracy: ", score_rf
	t3=time()
	dur_rf=t3-t2
	print "time elapsed: ", dur_rf
	print "\n"

	t4=time()
	print "NaiveBayes"
	nb = BernoulliNB()
	clf_nb=nb.fit(X_train,y_train)
	score_nb=clf_nb.score(X_test_part,y_test_part)
	print "Acurracy: ", score_nb
	t5=time()
	dur_nb=t5-t4
	print "time elapsed: ", dur_nb

# write result data to excel file
	list1=[]
	list2=[]

	list1.append(i/100.0+0.05)
	list1.append(score_dt)
	list1.append(score_knn)
	list1.append(score_rf)
	list1.append(score_nb)

	list2.append(i/100.0+0.05)
	list2.append(dur_dt)
	list2.append(dur_knn)
	list2.append(dur_rf)
	list2.append(dur_nb)

	for col in range(len(list1)):
		worksheet.write(i/5+1,col,list1[col])
		worksheet2.write(i/5+1,col,list2[col])

workbook.close()



