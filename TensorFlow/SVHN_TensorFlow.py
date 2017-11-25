import os
file_dir= os.path.dirname(os.path.realpath(__file__))
os.chdir(file_dir)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import graph_constructor as gc
import data_preprocessing as dp

###############################################################################


# LOADING THE DATA
from scipy.io import loadmat
train = loadmat('data/train_32x32.mat')
X_train = train['X']
y_train = train['y'][:,0]
y_train[y_train==10] = 0
X_train = X_train.transpose((3,0,1,2))

test = loadmat('data/test_32x32.mat')
X_test = test['X']
y_test = test['y'][:,0]
y_test[y_test==10] = 0
X_test = X_test.transpose((3,0,1,2))

# EXPECTED IMAGE SIZE
img_height = 32
img_width = 32

# PROCESSING THE DATA AND HAVING A LOOK AT AN EXAMPLE
rand_int = np.random.randint(X_train.shape[0])
rand_img = X_train[rand_int,:,:,:]
plt.imshow(rand_img)
plt.show()

method = 'grey_scale'
X_train = dp.prepocess(X_train,method=method)
X_test = dp.prepocess(X_test,method=method)
X_train, X_test = dp.normalize(X_train,X_test)

if X_train.shape[3] == 3:
    rand_img = X_train[rand_int,:,:,:]-np.amin(X_train[rand_int,:,:,:])
    rand_img = rand_img/np.amax(rand_img)
else:
    rand_img = X_train[rand_int,:,:,0]
plt.imshow(rand_img)
plt.show()


###############################################################################

# BUILDING TENSORFLOW MODEL
tf.reset_default_graph()
model = gc.CNN(img_height=img_height,img_width=img_width,num_channels=X_train.shape[3])
model.build()

# VARIOUS PARAMETERS
BATCH_SIZE = 128
NUM_EPOCH = 20
NUM_PRINT = 5
NUM_IT = int(X_train.shape[0]/BATCH_SIZE)
PRINT_INT = 100
NUM_PRINT = 5
PRINT_EVERY = int(int(NUM_IT/NUM_PRINT )/PRINT_INT)*PRINT_INT

if PRINT_EVERY == 0:
    PRINT_EVERY = None
    print("\nPRINTING INTERVAL too large compared to batch size. \nWon't be printing during epoch loop. ")

# BUILDIN SOLVER
solver = gc.SGD(batch_size = BATCH_SIZE, num_epoch = NUM_EPOCH, print_every = PRINT_EVERY)

# MAKE SOLVER ACQUIRE ALL ATTRIBUTES OF THE MODEL
solver.__dict__.update(model.__dict__)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

print('TRAINING')
print("="*60)      
print("="*60) 
solver.train(sess,X_train,y_train,X_test,y_test,True,True)
