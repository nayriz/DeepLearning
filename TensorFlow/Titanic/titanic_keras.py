from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
import os
import pandas as pd
file_dir= os.path.dirname(os.path.realpath(__file__))
os.chdir(file_dir)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

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

train_ft = pd.DataFrame({k: train_set[k].values for k in FEATURES})
train_labels = pd.Series(train_set[LABEL].values)

train_ft = train_ft.values
train_labels = train_labels.values


test_ft = pd.DataFrame({k: test_set[k].values for k in FEATURES})
test_labels = pd.Series(test_target[LABEL].values)

test_ft = test_ft.values
test_labels = test_labels.values

####################
# start using KERAS

# BUILD MODEL
model = Sequential()

activation = 'sigmoid'
h_layers = [10,10,10]

for i in range(len(h_layers)):
  
  if i == 0:
    model.add(Dense(h_layers[i], input_dim=len(FEATURES), activation='sigmoid'))
  else:
    model.add(Dense(h_layers[i], activation=activation))

model.add(Dense(1, activation=activation))

model.compile(loss='binary_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy',metrics.binary_accuracy])

# TRAIN
model.fit(train_ft, train_labels,
          epochs=500,
          batch_size=10,
          verbose=0)

# CHECK ACCURACY ON TRAINING SET
train_preds = model.predict(train_ft)
score = model.evaluate(train_ft, train_labels, verbose=0)
print('Train accuracy:', score[1])

# CHECK ACCURACY ON TEST SET
score_test = model.evaluate(test_ft, test_labels, verbose=0)
print('Test accuracy:', score_test[1])
