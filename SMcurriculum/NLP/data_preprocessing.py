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

# This module contains the function signatures for data preprocessing on the IMDB
# movie review dataset. The code is heavily commented to allow you to follow along easily.

# Please report any bugs you find using @yazabi, or email us at contact@yazabi.com.
from __future__ import print_function
from __future__ import generators
from __future__ import division


import os
dir_dp = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_dp)

import keras
# useful packages
import nltk
import gensim
import sklearn
import numpy as np
from operator import itemgetter
stopset = set(nltk.corpus.stopwords.words('english'))


def load_imdb_data(one_hot_labels=True):

    # inspired from https://www.youtube.com/watch?v=gnNvGUROWjY

    print('Loading and preprocessing data.')

    Train = sklearn.datasets.load_files(
        dir_dp + '/aclImdb/train/', categories=['neg', 'pos'])
    print('Training data loaded.')

    X_train = Train.data
    y_train = Train.target

    Test = sklearn.datasets.load_files(
        dir_dp + '/aclImdb/test/', categories=['neg', 'pos'])
    print('Testing data loaded.')
    X_test = Test.data
    y_test = Test.target

    X_train = [s.replace(b'<br />', b' ') for s in X_train]
    X_test = [s.replace(b'<br />', b' ') for s in X_test]

    print('Loading and preprocessing completed.')

    return X_train, X_test, y_train, y_test


def tokenize(text):

    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.decode("utf8"))

    # Note; gensim also has its own tokenizer
    #tokens = gensim.utils.simple_tokenize(text.decode("utf8"))

    tokens = [token for token in tokens if token not in stopset]

    return tokens


def make_embedding_matrix(texts, size):
    #    """Create an embedding matrix from a list of text samples.
    #    Hint: gensim
    #
    #    params:
    #    :texts: a list of text samples containing the vocabulary words.
    #    :size: the size of the word-vectors.
    #
    #    returns:
    #    :embedding_matrix: a dictionary mapping words to word-vectors (embeddings).
    #    """
    print('Tokenizing the texts.')

    TOKENS = []
    for text in texts:

        TOKENS.append(tokenize(text))

    print('Creating embedding matrix.')

    min_count = 5
    # NOTE: a big list should be passed to Word2Vec, not a  list of lists!!!
    embedding_matrix = gensim.models.Word2Vec(TOKENS, size=size,
                                              min_count=min_count)

    embedding_matrix.min_count = min_count

    embedding_matrix.save('embed')

    return embedding_matrix


def load_embedding_matrix(filepath):

    embedding_matrix = gensim.models.Word2Vec.load(filepath)

    return embedding_matrix


def to_word_vectors(tokenized_samples, embedding_matrix, max_seq_length):

    num_samples = len(tokenized_samples)
    TWV = np.zeros((num_samples, max_seq_length, embedding_matrix.vector_size))

    for i, tokens in enumerate(tokenized_samples):
        for j, word in enumerate(tokens):
            if word in embedding_matrix.wv and j < max_seq_length:
                TWV[i, :, :] = embedding_matrix.wv[word]

    return TWV


def generate_batches(data, labels, batch_size, embedding_matrix=None):
	
	# inspired from https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python

	n_text = len(labels)
	
	IND = np.arange(n_text)
	np.random.shuffle(IND)
	
	for step in range(0, n_text - batch_size + 1, batch_size):
		indices = IND[step, step + batch_size]
		
		
		if embedding_matrix is not None:	
			TOKENS = []
			for text in data[indices]:
				TOKENS.append(tokenize(text))
		
			batch_data = to_word_vectors(TOKENS, embedding_matrix, 20)
			
		else:
			data[indices]
			
		batch_labels = labels[indices]
			
		yield batch_data, batch_labels