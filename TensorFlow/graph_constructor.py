import math
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import matplotlib.pyplot as plt

class CNN(object):
    
    ''' 
    This class defines a classic CNN model.
    It can accommodate various classic architectures.
    '''
    
    def __init__(self,**kwargs):
        
        for key, value in kwargs.items():
            setattr(self, key, value) 
        
        self.Xpholder = tf.placeholder(tf.float32, [None, self.img_height, 
                                                    self.img_width,  
                                                    self.num_channels])
        self.ypholder = tf.placeholder(tf.int64, [None])
        self.is_training = tf.placeholder(tf.bool)

    def cnn_scores(self,X,is_training):
        
        '''
        Define a CNN architecture and perform a forward pass.
        Softmax is NOT applied.
        '''      
        
        hw0 = self.img_height
        ck = 3 
        p = math.floor(ck/2)
        cs = 1
        pk = 2
        hwc1 = math.floor((hw0 - ck + 2*p)/cs+1)
        hwc2 = math.floor((hwc1 - ck + 2*p)/cs+1)
        nConv = 2 
        hwe = int(hwc2/pk**nConv)
            
        conv1 = layers.conv2d(X, 32, ck, activation_fn=tf.nn.relu, padding='SAME')    
        pool1 = tf.layers.max_pooling2d(conv1,pk,pk) 
        conv2 = layers.conv2d(pool1, 64, ck, activation_fn=tf.nn.relu, padding='SAME')   
        pool2 = tf.layers.max_pooling2d(conv2,pk,pk) 
        pool2_flat =  tf.reshape(pool2, [-1, hwe*hwe*64])
        dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        do1 = tf.layers.dropout(inputs=dense1, rate=.5, training=is_training)    
        scores = tf.layers.dense(inputs=do1, units=10)
        
        return scores
    
    def loss(self,y_true,scores):
        
        '''
        Calculate the mean loss.
        Softmax is applied here.
        '''
        
        mean_loss = tf.losses.softmax_cross_entropy(tf.one_hot(y_true,10),scores)        
        
        return mean_loss
    
    def optimize(self,mean_loss):
       
        ''' 
        Define one training step of the optimization problem.
        '''
        
        optimizer = tf.train.AdamOptimizer(1e-4)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.control_dependencies(extra_update_ops):
            train_step = optimizer.minimize(mean_loss)    
    
        return train_step
    
    def build(self):
        
        '''
        Build the model.
        '''
        
        self.scores = self.cnn_scores(self.Xpholder,self.is_training)
        self.mean_loss = self.loss(self.ypholder,self.scores)
        self.train_step = self.optimize(self.mean_loss)

class SGD(object):

    def __init__(self,**kwargs):

        for key, value in kwargs.items():
            setattr(self, key, value)
        


    def forward_pass(self,session,X,y,indices, training=None):
        
        '''
        Perform one forwad pass of the model.
        This can be for training or predicting/testing.
        The accuracy over the whole training set, and the losses for each
        mini_batch are are also returned.
        '''
        
        correct = 0        
        losses = []
        
        self.correct_prediction = tf.equal(tf.argmax(self.scores,1), self.ypholder)
        
        for i in range(int(math.ceil(X.shape[0]/self.batch_size))):  
    
            start_idx = (i*self.batch_size)%X.shape[0]
            idx = indices[start_idx:start_idx+self.batch_size]             
            actual_batch_size = y[idx].shape[0]  
            feed_dict = {self.Xpholder: X[idx,:],self.ypholder: y[idx]}
            if training is not None:
                feed_dict[self.is_training] = True         
                loss, corr, _ = session.run([self.mean_loss,self.correct_prediction,training],feed_dict=feed_dict) 
                losses.append(loss)
                if self.print_every is not None:
                    if i % self.print_every == 0:   
                        print("iter {0}: with minibatch training loss = {1:.2g} and accuracy of {2:.2g}"\
                              .format(i,loss,np.sum(corr)/actual_batch_size))                           
            else:
                feed_dict[self.is_training] = False    
                loss, corr = session.run([self.mean_loss,self.correct_prediction],feed_dict=feed_dict)             
            correct += np.sum(corr)               
    
           
        accuracy = correct/X.shape[0]
        
        return accuracy, losses
    
    def train(self,session, X_train, y_train,X_val=None, y_val=None, plot_losses=False, pred_train = True):
        
        '''
        Train the model.
        You can choose to output the accuracy for the training and/or the
        validation set if you supply one.
        '''

        train_indices = np.arange(X_train.shape[0])
        
        val_indices = np.arange(X_val.shape[0])
        
        for e in range(self.num_epoch):
            
           print('Epoch',e+1)
            
           np.random.shuffle(train_indices) 
           train_accuracy, losses = self.forward_pass(session,X_train,y_train,
                                               train_indices,training = self.train_step)

           if plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e+1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()

           if pred_train:
                check_accuracy, _ = self.forward_pass(session,X_train,y_train,
                                                       train_indices)
                print("overall train accuracy {0:.3g}"\
                      .format(train_accuracy))   

           if y_val is not None:
               val_accuracy, _ = self.forward_pass(session,X_val,y_val,                                               val_indices)
               print("validation accuracy {0:.3g}"\
                      .format(val_accuracy)) 
                
           print("="*60)            
