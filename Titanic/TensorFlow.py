from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd
import tensorflow as tf
import itertools
import numpy as np
from sklearn.metrics import accuracy_score
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

file_dir= os.path.dirname(os.path.realpath(__file__))
os.chdir(file_dir)

import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

########################################################################
############### SUPER IMPORTANT ########################################
############### !!! DANGER!!! ##########################################


LABEL = "Survived" # do NOT put [ ] around "Survived" !!!
 

###########################################################################

coltrain = pd.read_csv("train.csv", nrows=1).columns
train_set = pd.read_csv("train.csv", skipinitialspace=True,
                           skiprows=1, names = coltrain)

coltest = pd.read_csv("test.csv", nrows=1).columns
test_set = pd.read_csv("test.csv", skipinitialspace=True,
                           skiprows=1, names = coltest)

coltarget = pd.read_csv("gender_submission.csv", nrows=1).columns
test_target = pd.read_csv("gender_submission.csv", skipinitialspace=True,
                           skiprows=1, names = coltarget)


def preprocess(data):
 
  data = pd.get_dummies(data, columns=["Embarked","Pclass"])
  clean_gender = {"Sex":{"male": 0, "female": 1}}  
  data.replace(clean_gender, inplace=True)
  data['Age'].fillna(-1,inplace=True )
  return data

train_set = preprocess(train_set)
test_set = preprocess(test_set)


FEATURES = ['Sex', 'Age', 'SibSp', 'Parch','Fare', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_S']

# START USING TENSORFLOW
sess = tf.Session()
tf.set_random_seed(1234)

feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

def get_input_fn(data_set,data_target, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y=pd.Series(data_target[LABEL].values), num_epochs=num_epochs,
        shuffle=shuffle)

# Build a fully connected DNN
model = tf.estimator.DNNClassifier(feature_columns=feature_cols,
                                        hidden_units=[10,10,10],
    dropout=.2,
    optimizer='Adagrad',
    activation_fn = tf.sigmoid)

# Train
model.train(input_fn=get_input_fn(train_set,train_set, num_epochs=1000, shuffle=True), steps=len(train_set))

print('\n= FINISHED TRAINING =')

# ACCURACY ON TRAINING SET

y = model.predict(
      input_fn=
      get_input_fn(train_set,
                   train_set, 
                   num_epochs=1, 
                   shuffle=False),
                  )

preds = list(np.argmax(p["probabilities"]) for p in itertools.islice(y, len(train_set)))

pred = np.array(preds)

labels = pd.Series(train_set[LABEL].values)

ascore = accuracy_score(labels,pred)

print('train set accuracy',ascore)



############################################
y = model.predict(
      input_fn=
      get_input_fn(test_set,
                   test_target, 
                   num_epochs=1, 
                   shuffle=False),
                  )

preds = list(np.argmax(p["probabilities"]) for p in itertools.islice(y, len(test_set)))

pred = np.array(preds)

labels = pd.Series(test_target[LABEL].values)

ascore = accuracy_score(labels,pred)

print('test set accuracy',ascore)