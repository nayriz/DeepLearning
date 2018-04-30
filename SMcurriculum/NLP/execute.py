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

# This script contains a loose template for building and training machine learning models
# on the imdb movie review dataset

# Please report any bugs you find using @yazabi, or email us at contact@yazabi.com.
from __future__ import print_function
from __future__ import generators
from __future__ import division

import sys
import data_preprocessing as dp
from models import LSATextClassifier
from models import CNNTextClassifier
from models import RNNTextClassifier

import keras 
params = {}
if __name__ == "__main__":

	use_model = sys.argv[1]
	if use_model is None:
		print('Specify model to be used. LSATextClassifier, RNNTextClassifier or CNNTextClassifier')
		sys.exit()

	# build and train model
	if use_model == 'LSATextClassifier':
		params['N_FEATURES'] = 30
		X_train, X_test, y_train, y_test = dp.load_imdb_data()
		model = LSATextClassifier(additional_parameters=params)
		model.build()
		model.train(X_train, y_train)

	elif use_model == 'RNNTextClassifier':

		BATCH_SIZE = 32
		NUM_EPOCHS = 2
		
		params['MAX_SEQ_LENGTH'] = 80
		params['NUM_WORDS'] = 20000
		params['EMBEDDING_SIZE'] = 50
		params['INDEX_FROM'] = 3
		
		(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=params['NUM_WORDS'], index_from=params['INDEX_FROM'])
		X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=params['MAX_SEQ_LENGTH'])
		X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=params['MAX_SEQ_LENGTH'])		
			
		model = RNNTextClassifier(additional_parameters=params)
		model.build()
		model.train(X_train, y_train, BATCH_SIZE, NUM_EPOCHS)

	elif use_model == 'CNNTextClassifier':

		BATCH_SIZE = 32
		NUM_EPOCHS = 10
		
		params['MAX_SEQ_LENGTH'] = 400
		params['NUM_WORDS'] = 5000
		params['EMBEDDING_SIZE'] = 50
		params['INDEX_FROM'] = 3		
		
		(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=params['NUM_WORDS'], index_from=params['INDEX_FROM'])
		X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=params['MAX_SEQ_LENGTH'])
		X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=params['MAX_SEQ_LENGTH'])		
		
		model = CNNTextClassifier(additional_parameters=params)
		model.build()
		model.train(X_train, y_train, BATCH_SIZE, NUM_EPOCHS)

	# evaluate model
	accuracy = model.evaluate(X_test, y_test)
	print('Test accuracy: ', accuracy)

	# predict
	neg_review = 'This movie was the worst thing I have ever watched.'
	pos_review = 'This was the greatest thing. I really liked it.'
	neg_pred = model.predict(neg_review)
	pos_pred = model.predict(pos_review)
	
	# REPORT BUG
	print('Prediction on negative review: ', neg_pred)
	print('Prediction on positive review: ', pos_pred)
