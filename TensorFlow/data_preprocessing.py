import numpy as np

def prepocess(X,method=None):
    
    ''' Preprocessing method. Choose between:
    1. 'average': takes the average of the three channels for each images
    2. 'red': keeps only the the 1st channel of each image
    3. 'green': keeps only the the 2nd channel of each image
    4. 'blue': keeps only the the 3rd channel of each image
    5. 'grey scale': computes the gray scale values for each images

    or do nothing
    '''
    
    if method is not None:
               
        if method == 'average':
            
            X = np.mean(X,axis=3)
    
        if method == 'red':
            
            X = X[:,:,:,0]
    
        if method == 'green':
            
            X = X[:,:,:,1]
              
            
        if method == 'blue':
    
            X = X[:,:,:,2]
    
        if method == 'blue':
    
            X = X[:,:,:,2]
             
        if method == 'grey_scale':
            
            X = np.dot(X, [0.2989, 0.5870, 0.1140])
            
        X = np.expand_dims(X,axis = 3)
        
    else:
        print('No channel preprocessing was applied.')
        
    return X

def normalize(X_train,X_test,X_val = None):
    
        Xmean = np.mean(X_train,axis=0)    
        Xstd = np.std(X_train,axis=0)
        
        X_train = (X_train - Xmean[np.newaxis,:,:,:])/Xstd[np.newaxis,:,:,:]
        X_test = (X_test - Xmean[np.newaxis,:,:,:])/Xstd[np.newaxis,:,:,:]
        
        if X_val is not None:
            X_val = (X_val - Xmean[np.newaxis,:,:,:])/Xstd[np.newaxis,:,:,:]        
            
            print('Data normalized.')
            return X_train, X_test, X_val
        else:
           print('Data normalized.')
           return X_train, X_test