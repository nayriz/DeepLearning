#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt

class KNN(object):
    
    def __init__(self, n_feat):
        self.X_train = tf.placeholder(tf.float32,shape = [None,n_feat])
        self.X_test = tf.placeholder(tf.float32,shape = [None,n_feat])

    def distance_matrix(self):
        
        A2 = tf.reduce_sum(self.X_train**2,axis=1)
        B2 = tf.reduce_sum(self.X_test**2,axis=1) 
        AB = tf.matmul(self.X_test,tf.transpose(self.X_train))
        
        dists = tf.sqrt( tf.expand_dims(B2,axis=1) + A2 - 2*AB)
        
        self.dists = dists
        
        return dists

    def predict(self,dists,labels,y_train,k=1):
        
        num_test = labels.shape[0]
        preds = np.zeros(num_test)
        for i in range(num_test):
          sort_dists = np.argsort(dists[i,:])     
          closest_y = y_train[sort_dists[0:k,]]   
          unique, counts = np.unique(closest_y, return_counts=True)
          
          #this is to get the smaller label if there is a tie
          m = np.amax(counts)
          arg_ms = (counts == m)
          preds[i] = int(min(unique[arg_ms]))   
          

        return preds
    
class Logistic(object):
    
    def __init__(self, n_feat,n_class):
        self.X_train = tf.placeholder(tf.float32,shape = [None,n_feat])
        self.y_train = tf.placeholder(tf.int64,shape = [None]) 
        self.n_feat = n_feat
        self.n_class = n_class
        
    def scores(self):
            weight_initializer = tf.contrib.layers.xavier_initializer()
            W = tf.Variable(weight_initializer((self.n_feat,self.n_class)))       
            b = tf.Variable(tf.zeros((self.n_class)))   
            logits = tf.matmul(self.X_train,W) + b
            
            self.logits = logits
            
            return logits
            
    def loss(self,logits):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,labels=self.y_train)
        
        mean_loss = tf.reduce_mean(losses)  
        
        return mean_loss
        
    def train_op(self,mean_loss):

        
        optimizer = tf.train.AdamOptimizer(1e-3)
        train_op = optimizer.minimize(mean_loss)    
        
        return train_op
    
    def SGD(self, Xd, yd):

        correct_prediction = tf.equal(tf.argmax(self.logits,1), self.y_train)
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)
        variables = [self.mean_loss,correct_prediction,self.train_op]
    
        for e in range(self.epochs):
            correct = 0
            losses = []
            for i in range(int(math.ceil(Xd.shape[0]/self.batch_size))):
                start_idx = (i*self.batch_size)%Xd.shape[0]
                idx = train_indicies[start_idx:start_idx+self.batch_size]
                feed_dict = {self.X_train: Xd[idx,:], self.y_train: yd[idx]}
                actual_batch_size = yd[idx].shape[0]
                loss, corr, _ = self.sess.run(variables,feed_dict=feed_dict)
                losses.append(loss*actual_batch_size)
                correct += np.sum(corr)
    
            total_correct = correct/Xd.shape[0]
            total_loss = np.sum(losses)/Xd.shape[0]
            
            if self.print_epoch:
                print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
                      .format(total_loss,total_correct,e+1))
            if self.plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e+1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()


    def SGD_train(self,sess,Xd, yd, epochs=20, batch_size=64, plot_losses=False, print_epoch = False):
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.plot_losses = plot_losses
        self.print_epoch = print_epoch
        self.sess = sess
        self.scores()
        self.mean_loss = self.loss(self.logits)
        self.train_op = self.train_op(self.mean_loss)
    
        sess.run(tf.global_variables_initializer())        
        self.SGD(Xd, yd)        
        
    def predict(self, sess,Xd, yd):
   
        correct_prediction = tf.equal(tf.argmax(self.logits,1), self.y_train)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        indicies = np.arange(Xd.shape[0])
        variables = [self.mean_loss,correct_prediction,accuracy]
        
        correct = 0
        losses = []
        
        for i in range(int(math.ceil(Xd.shape[0]/self.batch_size))):
        
                start_idx = (i*self.batch_size)%Xd.shape[0]
                idx = indicies[start_idx:start_idx+self.batch_size]
                feed_dict = {self.X_train: Xd[idx,:], self.y_train: yd[idx]}
                actual_batch_size = yd[idx].shape[0]
                loss, corr, _ = self.sess.run(variables,feed_dict=feed_dict)
                losses.append(loss*actual_batch_size)
                correct += np.sum(corr)
        
        total_correct = correct/Xd.shape[0]
        print("accuracy on test set: ",total_correct)
 
class Two_layer(object):
    
    def __init__(self, n_feat,n_class):
        self.X_train = tf.placeholder(tf.float32,shape = [None,n_feat])
        self.y_train = tf.placeholder(tf.int64,shape = [None]) 
        self.n_feat = n_feat
        self.n_class = n_class
        
    def scores(self):
        d1 = tf.layers.dense(self.X_train, units=1024, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=d1, units=self.n_class)
        self.logits = logits
            
        return logits
            
    def loss(self,logits):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,labels=self.y_train)
        
        mean_loss = tf.reduce_mean(losses)  
        
        return mean_loss
        
    def train_op(self,mean_loss):

        
        optimizer = tf.train.AdamOptimizer(1e-3)
        train_op = optimizer.minimize(mean_loss)    
        
        return train_op
    
    def SGD(self, Xd, yd):

        correct_prediction = tf.equal(tf.argmax(self.logits,1), self.y_train)
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)
        variables = [self.mean_loss,correct_prediction,self.train_op]
    
        for e in range(self.epochs):
            correct = 0
            losses = []
            for i in range(int(math.ceil(Xd.shape[0]/self.batch_size))):
                start_idx = (i*self.batch_size)%Xd.shape[0]
                idx = train_indicies[start_idx:start_idx+self.batch_size]
                feed_dict = {self.X_train: Xd[idx,:], self.y_train: yd[idx]}
                actual_batch_size = yd[idx].shape[0]
                loss, corr, _ = self.sess.run(variables,feed_dict=feed_dict)
                losses.append(loss*actual_batch_size)
                correct += np.sum(corr)
    
            total_correct = correct/Xd.shape[0]
            total_loss = np.sum(losses)/Xd.shape[0]
            
            if self.print_epoch:
                print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
                      .format(total_loss,total_correct,e+1))
            if self.plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e+1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()


    def SGD_train(self,sess,Xd, yd, epochs=20, batch_size=64, plot_losses=False, print_epoch = False):
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.plot_losses = plot_losses
        self.print_epoch = print_epoch
        self.sess = sess
        self.scores()
        self.mean_loss = self.loss(self.logits)
        self.train_op = self.train_op(self.mean_loss)
    
        sess.run(tf.global_variables_initializer())        
        self.SGD(Xd, yd)        
        
    def predict(self, sess,Xd, yd):
   
        correct_prediction = tf.equal(tf.argmax(self.logits,1), self.y_train)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        indicies = np.arange(Xd.shape[0])
        variables = [self.mean_loss,correct_prediction,accuracy]
        
        correct = 0
        losses = []
        
        for i in range(int(math.ceil(Xd.shape[0]/self.batch_size))):
        
                start_idx = (i*self.batch_size)%Xd.shape[0]
                idx = indicies[start_idx:start_idx+self.batch_size]
                feed_dict = {self.X_train: Xd[idx,:], self.y_train: yd[idx]}
                actual_batch_size = yd[idx].shape[0]
                loss, corr, _ = self.sess.run(variables,feed_dict=feed_dict)
                losses.append(loss*actual_batch_size)
                correct += np.sum(corr)
        
        total_correct = correct/Xd.shape[0]
        print("accuracy on test set: ",total_correct)    