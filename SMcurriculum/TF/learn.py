#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
file_dir= os.path.dirname(os.path.realpath(__file__))
os.chdir(file_dir)

import tensorflow as tf
import data_preprocessing as dp

import numpy as np
import graph_constructor as gc
Xd = np.loadtxt('X_train.txt')
yd = np.loadtxt('y_train.txt')
ind = np.random.permutation(Xd.shape[0])
Xd = Xd[ind]
yd = yd[ind]

Xtest = np.loadtxt('X_test.txt')
ytest = np.loadtxt('y_test.txt')

Xd = dp.normalizer(Xd)
Xtest = dp.normalizer(Xtest)

n_class = int(np.amax(yd))

yd -= 1
ytest -= 1
Xtrain, ytrain, Xval, yval = dp.spliter(Xd,yd,1.0)

ytrain.astype(int), yval.astype(int)

###############################################################################
n_feat = Xd.shape[1]   
    
def train_and_validate(algorithm):

    if algorithm == 'knn':
        model = gc.KNN(n_feat)
        dists = model.distance_matrix()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())        
        feed_dict = {model.X_train: Xtrain, 
                     model.X_test: Xtest}
        dists = sess.run(dists,feed_dict=feed_dict)

        preds = model.predict(dists,ytest,ytrain,k=9)
        num_correct = np.sum(preds == ytest)
        accuracy = float(num_correct) / ytest.shape[0]
        print('accuracy on test set: ',accuracy)          

            
    elif algorithm == 'logistic':
        tf.reset_default_graph()
        model = gc.Logistic(n_feat,n_class)
        sess = tf.Session()       
        model.SGD_train(sess,Xd, yd, print_epoch = False)
        model.predict(sess,Xtest, ytest)

    elif algorithm == '2-layer':
        tf.reset_default_graph()
        model = gc.Two_layer(n_feat,n_class)
        sess = tf.Session()     
        model.SGD_train(sess,Xd, yd, print_epoch = False)
        model.predict(sess,Xtest, ytest)
       
print("="*40)
print('Training with knn')        
train_and_validate('knn')    
print("="*40)
print('Training with logistic regression') 
train_and_validate('logistic')
print("="*40)
print('Training with a two-layer NN')     
train_and_validate('2-layer') 