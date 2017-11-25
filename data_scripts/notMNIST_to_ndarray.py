###############################################################################
# This file:
# 1. Downloads the notNOTMNIST data from the original repositor and saves it in
# the folder where the file itself is located.

# 2. Creates 4 numpy arrays containing the data and training set and the
# corresponding labels. Invalid files are discared.

# 3. Saves the numoy arrays as npy files in the folder where the file itself
# is located
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile

file_dir= os.path.dirname(os.path.realpath(__file__))
os.chdir(file_dir)
image_size = 28

###############################################################################
def progress_bar(blocknum, blocksize, totalsize):
    
    '''
    Helper function that creates a progress bar when downloading the data.
    '''
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))

###############################################################################
# DOWNLOADING THE DATA

print('='*20)        
from six.moves.urllib.request import urlretrieve

if not os.path.exists('notMNIST_large.tar.gz'):
    print('Downloading the training data.')
    url = 'http://yaroslavvb.com/upload/notMNIST/'
    train_file_name, _ = urlretrieve(url+'notMNIST_large.tar.gz',
                                     'notMNIST_large.tar.gz',
                                     reporthook=progress_bar)
if not os.path.exists('notMNIST_small.tar.gz'):
    print('Downloading the testing data.')
    test_file_name, _ = urlretrieve(url+'notMNIST_small.tar.gz',
                                    'notMNIST_small.tar.gz',
                                     reporthook=progress_bar)

################################################################################
print('='*20)        
print('Processing the training set.')

# EXTRACTING THE TRAINING SET TO A TAR FILE
if not os.path.exists('notMNIST_large'):
    print('Untarring.')
    tar = tarfile.open('notMNIST_large.tar.gz')
    tar.extractall()
    tar.close()

SUBDIR = os.listdir('notMNIST_large') 

n_images = 0

for subdir in SUBDIR:
    class_dir = 'notMNIST_large/'+ subdir 
    image_files = os.listdir(class_dir)
    n_images += len(image_files)
   
image_set = None
label_set = None

# The labels will be modified so that A corresponds to 0 ... J to 9
ord_max = ord('J')

n__processed_img = 0 
print('Processing each image.')

for subdir in SUBDIR:
    
    class_dir = 'notMNIST_large/'+ subdir 
    image_files = os.listdir(class_dir)
    class_dataset = np.ndarray(shape=(len(image_files), image_size, image_size))
    del_ind = []
    
    print('Processing class ',subdir)
        
    for image_num, image in enumerate(image_files):
        try:
            image_dir = os.path.join(class_dir,image)
            image_data = plt.imread(image_dir).astype(float)
            class_dataset[image_num, :, :] = image_data
        except:
            del_ind.append(image_num)
            
    cleaned_class = np.delete(class_dataset, del_ind,axis = 0)

    if image_set is None:
        image_set = cleaned_class
        label_set = (ord_max-ord(subdir))*np.ones(cleaned_class.shape[0])
    else:    
        labels = (ord_max-ord(subdir))*np.ones(cleaned_class.shape[0])
        image_set = np.concatenate((image_set,cleaned_class))
        label_set = np.concatenate((label_set,labels))
        
    n__processed_img += len(image_files)    
    print(subdir,' class completed ',n__processed_img/n_images*100,'%')
    
np.save('X_train',image_set)
np.save('y_train',label_set)

###############################################################################
print('='*20)
print('Processing the test set.')

# EXTRACTING THE TEST SET TO A TAR FILE
if not os.path.exists('notMNIST_small'):
    tar = tarfile.open('notMNIST_small.tar.gz')
    tar.extractall()
    tar.close()

SUBDIR = os.listdir('notMNIST_small') 

image_set = None
label_set = None

# The labels will be modified so that A corresponds to 0 ... J to 9
ord_max = ord('J')

for subdir in SUBDIR:
    
    class_dir = 'notMNIST_small/'+ subdir 
    image_files = os.listdir(class_dir)
    class_dataset = np.ndarray(shape=(len(image_files), image_size, image_size))
    del_ind = []
    
    for image_num, image in enumerate(image_files):
        try:
            image_dir = os.path.join(class_dir,image)
            image_data = plt.imread(image_dir).astype(float)
            class_dataset[image_num, :, :] = image_data
        except:
            del_ind.append(image_num)
            
    cleaned_class = np.delete(class_dataset, del_ind,axis = 0)

    if image_set is None:
        image_set = cleaned_class
        label_set = (ord_max-ord(subdir))*np.ones(cleaned_class.shape[0])
    else:    
        labels = (ord_max-ord(subdir))*np.ones(cleaned_class.shape[0])
        image_set = np.concatenate((image_set,cleaned_class))
        label_set = np.concatenate((label_set,labels))
        
    print(subdir,' set completed')
    
np.save('X_test',image_set)
np.save('y_test',label_set)
