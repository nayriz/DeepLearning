import sklearn.preprocessing
import numpy as np

def normalizer(X):

	X = sklearn.preprocessing.scale(X)

	# SANITY check
	i = np.random.randint(0,X.shape[1])
	
	if abs(np.std(X[:,i]) - 1.0) < abs(np.std(X[i,:]) - 1.0):
		print('Normalizer sanity check passed.')
	else:
		raise Exception("It's possible you got rows and columns mixed up. Double check!")
	
	return X

def spliter(X,y,p):
	
	num_train = int(X.shape[0]*p)
	
	Xtrain = X[:num_train]
	ytrain = y[:num_train]
	Xval = X[num_train:]
	yval = y[num_train:]
	
	return Xtrain, ytrain, Xval, yval
