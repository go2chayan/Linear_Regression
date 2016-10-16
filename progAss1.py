# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:39:44 2014
Programming Assignment 1: Linear Regression
@author: Md. Iftekhar Tanveer (itanveer@cs.rochester.edu)
"""
import random
import numpy as np

trainN = 348
testN = 42
devN = 45 

# Read each line and convert into 2D array
dictionary = {'democrat':1,'republican':-1,'y':1,'n':-1,'?':0}
input_file = open('voting2.dat')
lines = input_file.readlines()
allData = [line.strip().split(',') for line in lines if (\
    line.startswith('republican') or \
    line.startswith('democrat'))]
for i in range(np.size(allData,0)):
    for j in range(np.size(allData,1)):
        allData[i][j] = dictionary[allData[i][j]]
allData = np.array(allData)

accSum = 0
lamda = 0 # TA instructed that we don't need dev set. So I didn't tune lamda.
          # I just set it to zero

# This loop will randomly sample the dataset and calculate accuracy 
# 100 times and come up with an average accuracy
for i in range(0,99):
    # Create training, development and test set by random sampling
    allIdx = set(range(np.size(allData,0)))
    trainingSet = random.sample(allIdx,trainN)
    testSet = random.sample(allIdx.difference(trainingSet),testN)
    X_tr = np.hstack((allData[trainingSet,1:],np.ones([trainN,1])))
    Y_tr = allData[trainingSet,0][None].T
    X_te = np.hstack((allData[testSet,1:],np.ones([testN,1])))
    Y_te = allData[testSet,0][None].T
            
    # Model building on training set using Least Square Regression
    w = np.linalg.inv(X_tr.T.dot(X_tr) + lamda*np.eye(np.size(X_tr,1))). \
    dot(X_tr.T).dot(Y_tr)
    
    # Apply on test data and calculate the final accuracy
    Y_te_hat = np.sign(X_te.dot(w))
    accSum = accSum + sum(Y_te_hat == Y_te)/float(testN)

print 'Average Accuracy =', accSum/100, '(may change in each run due to', \
    'random sample)'
    