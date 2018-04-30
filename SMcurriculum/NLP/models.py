# Copyright (c) 2017 Yazabi Predictive Inc.

#################################### MIT License ####################################
#                                                                                   #
# Permission is hereby granted, free of charge, to any person obtaining a copy      #
# of this software and associated documentation files (the "Software"), to deal     #
# in the Software without restriction, including without limitation the rights      #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell         #
# copies of the Software, and to permit persons to whom the Software is             #
# furnished to do so, subject to the following conditions:                          #
#                                                                                   #
# The above copyright notice and this permission notice shall be included in all    #
# copies or substantial portions of the Software.                                   #
#                                                                                   #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE       #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER            #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,     #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     #
# SOFTWARE.                                                                         #
#                                                                                   #
#####################################################################################

# This module contains a class template for building machine learning models
# on the IMDB movie review dataset. The code is heavily commented to allow you to follow along easily.

# Please report any bugs you find using @yazabi, or email us at contact@yazabi.com.
from __future__ import print_function
from __future__ import generators
from __future__ import division

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, GlobalMaxPooling1D, LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding

class LSATextClassifier(object):

	def __init__(self, embedding_matrix=None, additional_parameters=None):
		"""Initialize the classifier with an (optional) embedding_matrix
		and/or any other parameters."""
		
		print("="*20)
		print('Initializing the model.')
		
		self.embedding_matrix = embedding_matrix
		self.solver = LogisticRegression()
		self.stage = None
		self.normalizer = Normalizer()
		
		if additional_parameters is not None:
			self.N_FEATURES = additional_parameters['N_FEATURES']
		
		print('Model initialized! \n')

	def build(self, model_parameters=None):
		"""Build the model/graph."""

		self.tfidf = TfidfVectorizer( ngram_range = (1,3 ),sublinear_tf=True, use_idf=True)		
		self.lsa = TruncatedSVD(n_components = self.N_FEATURES, n_iter = 10)
		
		print('Model built! \n')
		
	def SVDmatrix(self,X):
		
		'''
		X_train ~= Xk = Uk x Sk X Vk^T))
		
		The logistic regression is trained on X = Uk x Sk^T	
		The predictions are made on X' = X_test x Vk
		(see http://scikit-learn.org/stable/modules/decomposition.html) )
		
		'''
	
	
		if self.stage == 'train':
			
			print('Preparing the training reduced SVD matrix.')
			print('Obtaining TF-IDF features.')			
			# obtain  matrix of TF-IDF features
			tfidf = self.tfidf.fit_transform(X) 	
			
			print('Normalizing the matrix.')
			# normalize the matrix
			tfidf_norm = self.normalizer.fit_transform(tfidf)

			print('Obtaining the reduced order matrix')			
			# obtain the reduced order Xk_train = Uk x Sk^T	
			Xsvd = self.lsa.fit_transform(tfidf_norm) 			

		elif self.stage == 'test':			

			print('Preparing the test reduced SVD matrix.')
			
			# obtain the TF-IDF features corresponding to the 
			# TF-IDF matrix of the training set
			feats = self.tfidf.transform(X) 

			# normalize the matrix
			feats_norm = self.normalizer.fit_transform(feats)
			
			# obtain the reduced order Xk_test = X_test x Vk 				
			Xsvd = self.lsa.transform(feats_norm)
		
		return Xsvd
	
	def train(self, train_data, train_labels, additional_parameters=None):
		"""Train the model on the training data."""
		print("="*20)
		print('Training the model.')
		
		self.stage = 'train'

		X = LSATextClassifier.SVDmatrix(self,train_data)
		y = train_labels 		
		
		print('Applying logistic regression.')
		self.solver.fit(X,y)
		print('Training complete. \n')


	def evaluate(self, test_data, test_labels, additional_parameters=None):
		"""Evaluate the model on the test data.

		returns:
		:accuracy: the model's accuracy classifying the test data.
		"""
		
		print("="*20)
		print('Testing the model.')
		
		self.stage = 'test'
		
		X = LSATextClassifier.SVDmatrix(self,test_data)
		accuracy = self.solver.score(X,test_labels)

		return accuracy 
	
	def predict(self, review):
		"""Predict the sentiment of an unlabelled review.

		returns: the predicted label of :review:
		"""
		self.stage = 'test'

		print('Predicting.')
		X = LSATextClassifier.SVDmatrix(self,[review])	
		pred = self.solver.predict(X)
		
		return pred



class CNNTextClassifier(object):

	def __init__(self, embedding_matrix=None, additional_parameters=None):

		self.embedding_matrix = embedding_matrix
		if additional_parameters is not None:
			self.MAX_SEQ_LENGTH = additional_parameters['MAX_SEQ_LENGTH']
			self.NUM_WORDS = additional_parameters['NUM_WORDS']
			self.EMBEDDING_SIZE = additional_parameters['EMBEDDING_SIZE']
			self.INDEX_FROM = additional_parameters['INDEX_FROM']
			
	def build(self, model_parameters=None):
		
		# NECESSARY LAYERS FOR ALL MODELS
		model = Sequential()
		model.add(Embedding(self.NUM_WORDS, self.EMBEDDING_SIZE, input_length=self.MAX_SEQ_LENGTH))

###############################################################################		
		'''Dr. Jason Brownlee model
		see https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/		
		accuracy on test > .85
		overfits because of no regularization/dropout
		'''
#		model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
#		model.add(MaxPooling1D(pool_size=2))
#		model.add(Flatten())
#		model.add(Dense(250, activation='relu'))
#		model.add(Dense(1, activation='sigmoid'))
#		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		
###############################################################################
		''' Francois Chollet, official keras model
		see https://github.com/fchollet/keras/blob/master/examples/imdb_cnn.py
		accuracy on test > .87
		'''
		model.add(Dropout(0.2))
		model.add(Conv1D(250,3,padding='valid',activation='relu',strides=1))
		model.add(GlobalMaxPooling1D())
		model.add(Dense(250))
		model.add(Dropout(0.2))
		model.add(Activation('relu'))
		model.add(Dense(1))
		model.add(Activation('sigmoid'))
		model.compile(loss='binary_crossentropy',optimizer='adam',
metrics=['accuracy'])	
		
###############################################################################		
		
		self.model = model

	def train(self, train_data, train_labels, batch_size=50, num_epochs=5, additional_parameters=None):
		
		self.model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size, verbose=2)


	def evaluate(self, test_data, test_labels, additional_parameters=None):

		scores = self.model.evaluate(test_data, test_labels, verbose=0)
		accuracy = scores[1]
		
		return accuracy

	def predict(self, review):

		if type(review) is np.ndarray:
			preds =  np.round(self.model.predict(review))			
		elif isinstance(review, str):	
			tokenized_review = tf.keras.preprocessing.text.text_to_word_sequence(review)			
			word2ind = keras.datasets.imdb.get_word_index()			
			integer_review = [word2ind[w]+self.INDEX_FROM for w in tokenized_review]	
			integer_review = np.array(integer_review)
			
############################################################################### 
			# SANITY CHECK
#			word_to_id = keras.datasets.imdb.get_word_index()
#			word_to_id = {k:(v+self.INDEX_FROM) for k,v in word_to_id.items()}
#			word_to_id["<PAD>"] = 0
#			word_to_id["<START>"] = 1
#			word_to_id["<UNK>"] = 2			
#			id_to_word = {value:key for key,value in word_to_id.items()}
#			print(' '.join(id_to_word[id] for id in integer_review ))
# see https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset for explanations			
###############################################################################	
			
			x = keras.preprocessing.sequence.pad_sequences([integer_review], maxlen=self.MAX_SEQ_LENGTH)
			
			preds = np.round(self.model.predict(x))[0]			
		else:
			raise ValueError('This type of input is not supported in this code.')
	
		return preds
			
			
class RNNTextClassifier(object):

	def __init__(self, embedding_matrix=None, additional_parameters=None):

		self.embedding_matrix = embedding_matrix
		if additional_parameters is not None:
			self.MAX_SEQ_LENGTH = additional_parameters['MAX_SEQ_LENGTH']
			self.NUM_WORDS = additional_parameters['NUM_WORDS']
			self.EMBEDDING_SIZE = additional_parameters['EMBEDDING_SIZE']
			self.INDEX_FROM = additional_parameters['INDEX_FROM']
			
	def build(self, model_parameters=None):
		
		# NECESSARY LAYERS FOR ALL MODELS
		model = Sequential()
		model.add(Embedding(self.NUM_WORDS, self.EMBEDDING_SIZE, input_length=self.MAX_SEQ_LENGTH))

###############################################################################
		''' Francois Chollet, official keras model
		see https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py
		accuracy on test > .87
		'''
		model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy',
		              optimizer='adam',
		metrics=['accuracy'])
				
###############################################################################		
		
		self.model = model

	def train(self, train_data, train_labels, batch_size=50, num_epochs=5, additional_parameters=None):
		
		self.model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size, verbose=2)


	def evaluate(self, test_data, test_labels, additional_parameters=None):

		scores = self.model.evaluate(test_data, test_labels, verbose=0)
		accuracy = scores[1]
		
		return accuracy

	def predict(self, review):

		if type(review) is np.ndarray:
			preds =  np.round(self.model.predict(review))			
		elif isinstance(review, str):	
			tokenized_review = tf.keras.preprocessing.text.text_to_word_sequence(review)			
			word2ind = keras.datasets.imdb.get_word_index()			
			integer_review = [word2ind[w]+self.INDEX_FROM for w in tokenized_review]	
			integer_review = np.array(integer_review)
			
############################################################################### 
			# SANITY CHECK
#			word_to_id = keras.datasets.imdb.get_word_index()
#			word_to_id = {k:(v+self.INDEX_FROM) for k,v in word_to_id.items()}
#			word_to_id["<PAD>"] = 0
#			word_to_id["<START>"] = 1
#			word_to_id["<UNK>"] = 2			
#			id_to_word = {value:key for key,value in word_to_id.items()}
#			print(' '.join(id_to_word[id] for id in integer_review ))
# see https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset for explanations			
###############################################################################	
			
			x = keras.preprocessing.sequence.pad_sequences([integer_review], maxlen=self.MAX_SEQ_LENGTH)
			
			preds = np.round(self.model.predict(x))[0]			
		else:
			raise ValueError('This type of input is not supported in this code.')
	
		return preds