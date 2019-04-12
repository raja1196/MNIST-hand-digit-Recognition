# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 11:12:27 2019

@author: rajar
"""

import gzip
import numpy as np
import csv
from sklearn.metrics import confusion_matrix

#Defining Training and testing image sizes
img_size=28
img_full=img_size*img_size
len_train=60000
len_test=10000

#Defining a function to extract features
def loadDataset(file_path, len_images, isLabel, image_size = img_size):
    f = gzip.open(file_path,'r')
    if isLabel:
        f.read(8)
    else:
        f.read(16)
    buf = f.read(image_size * image_size * len_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    if isLabel:
        return data.reshape(len_images,1)
    return data.reshape(len_images , image_size * image_size)

#Loading dataset
X_train = loadDataset('train-images-idx3-ubyte.gz',len_train, False)
Y_train =  loadDataset('train-labels-idx1-ubyte.gz',len_train, True)
X_test =  loadDataset('t10k-images-idx3-ubyte.gz',len_test,False)
Y_test =  loadDataset('t10k-labels-idx1-ubyte.gz',len_test,True)


#Importing sklearn library for LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(solver='lbfgs')

#Training classifier
classifier_lr.fit(X_train, Y_train)
score_lrTrain = classifier_lr.score(X_train, Y_train)

#Estimating Prediction
y_lrpred = classifier_lr.predict(X_test)
score_lrTest= classifier_lr.score(X_test,Y_test)
cm = confusion_matrix(Y_test, y_lrpred)

#Saving as one hot encoding
y_lrpred=y_lrpred.astype(int)
lr = open('D:\\hw\\mnist\\lr.csv','w', newline='') 
w = csv.writer(lr)
for i in y_lrpred:
    x = [0 for _ in range(10)]
    x[i] = 1
    w.writerow(x)
lr.close()

#Importing sklearn for Random forest classifier
from sklearn.ensemble import RandomForestClassifier
#with hyper-parameters tuned
classifier_rf = RandomForestClassifier(criterion = 'entropy',bootstrap= True, max_depth=70, max_features='auto', min_samples_leaf= 4, min_samples_split= 10, n_estimators= 400)

#Training
classifier_rf.fit(X_train, Y_train)
score_rfTrain= classifier_rf.score(X_train, Y_train)

#Prediction
y_rfpred = classifier_rf.predict(X_test)
score_rfTest = classifier_rf.score(X_test, Y_test)
cm = confusion_matrix(Y_test,y_rfpred)

#saving RF classification as one-hot submission
y_rfpred=y_rfpred.astype(int)
rf = open('D:\\hw\\mnist\\rf.csv','w', newline='') 
w = csv.writer(rf)
for i in y_lrpred:
    x = [0 for _ in range(10)]
    x[i] = 1
    w.writerow(x)
rf.close()

#writing the name of the author
import csv

name=open("D:\\hw\\mnist\\name.csv","w")
n=csv.writer(name,lineterminator='\n\n')
n1=["rthiruv","rmishra4","vnagara2"]
n.writerow(n1)  
name.close()