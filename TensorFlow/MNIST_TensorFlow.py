import os
file_dir= os.path.dirname(os.path.realpath(__file__))
os.chdir(file_dir)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import graph_constructor as gc

###############################################################################
X_train = np.load('data/X_train_MNIST.npy')
y_train = np.load('data/y_train_MNIST.npy')
X_test = np.load('data/X_test_MNIST.npy')    
y_test = np.load('data/y_test_MNIST.npy')    

img_height = 28 
img_width = 28

###############################################################################

# PROCESSING THE DATA AND HAVING A LOOK AT AN EXAMPLE
rand_int = np.random.randint(X_train.shape[0])
rand_img = X_train[rand_int,:,:,0]
plt.imshow(rand_img)
plt.show()

###############################################################################

# BUILDING TENSORFLOW MODEL
tf.reset_default_graph()
model = gc.CNN(img_height=img_height,img_width=img_width,num_channels=X_train.shape[3])
model.build()

# VARIOUS PARAMETERS
BATCH_SIZE = 64
NUM_EPOCH = 5
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

